import json
import os
import threading
import time
from datetime import datetime
from typing import Dict, Union, Optional, List, Any
from cogbot.memory.ontology import get_facet_spec
from cogbot.memory_structures import ClaimNode, EpisodeNode, InsightNode, NoteNode, EntityNode
from cogbot import config

class MemoryGraph:
    """
    장기 기억(LTM) 그래프 데이터베이스 (Append-only Log Optimized)
    - 역할: Node/Edge의 CRUD 및 파일 입출력 (Business Logic 없음)
    - 저장소: ltm_graph.json (구조), ltm_embeddings.json (벡터)
    
    [Optimization Strategy]
    - Write: 변경 사항(Delta)을 'ltm_delta.jsonl' 파일 끝에 추가 (O(1))
    - Read: 시작 시 'Snapshot(JSON)' 로드 후 'Delta(JSONL)'를 리플레이하여 복원
    - Compaction: 주기적으로 Delta를 Snapshot에 병합하여 로그 파일 크기 관리
    """
    
    def __init__(self, graph_path: str = None, embeddings_path: str = None):
        self.graph_path = graph_path or config.LTM_GRAPH_PATH          # Snapshot
        self.embeddings_path = embeddings_path or config.LTM_EMBEDDINGS_PATH
        self.delta_path = self.graph_path.replace(".json", "_delta.jsonl") # Log File
        
        # Thread Safety
        self._lock = threading.RLock()
        
        # In-Memory Storage
        self.episodes: Dict[str, EpisodeNode] = {}
        self.insights: Dict[str, InsightNode] = {}
        self.notes: Dict[str, NoteNode] = {}
        self.claims: Dict[str, ClaimNode] = {}
        self.entities: Dict[str, EntityNode] = {}
        self._embeddings_cache: Dict[str, list] = {}
        self._boundary_persistence_migration_needed = False
        
        # 초기 로드 (Snapshot + Replay)
        self._load_all()

    # =================================================================
    # Public Methods (Thread-Safe & Logged)
    # =================================================================
    
    def add_episode(self, content: str, user_id: str, emotion: str, 
                    embedding: Optional[list] = None) -> EpisodeNode:
        with self._lock:
            node = EpisodeNode(
                content=content, 
                timestamp=time.time(),
                emotion_tag=emotion, 
                user_id=user_id,
                embedding=None
            )
            
            # Temporal Edge Logic
            last_id = list(self.episodes.keys())[-1] if self.episodes else None
            
            # 1. 메모리 업데이트
            self.episodes[node.node_id] = node
            if embedding:
                self._embeddings_cache[node.node_id] = embedding

            # 2. 로그 기록 (Node 생성)
            self._append_log("UPSERT_NODE", {"category": "episodes", "data": node.to_dict()})
            
            # 3. 로그 기록 (Embedding)
            if embedding:
                self._append_log("UPSERT_EMBEDDING", {"node_id": node.node_id, "vector": embedding})

            # 4. 로그 기록 (Temporal Edge)
            if last_id:
                self.connect_nodes(last_id, node.node_id, config.TEMPORAL_EDGE_FORWARD)
                self.connect_nodes(node.node_id, last_id, config.TEMPORAL_EDGE_BACKWARD)
                
            return node

    def add_or_update_insight(self, summary: str, confidence: float = 0.5, 
                              embedding: Optional[list] = None) -> InsightNode:
        with self._lock:
            now = time.time()
            normalized = self._normalize_text(summary)
            node = None
            for existing in self.insights.values():
                if self._normalize_text(existing.summary) == normalized:
                    node = existing
                    break

            if node:
                node.summary = summary
                node.confidence = max(node.confidence, confidence)
                node.last_updated = now
            else:
                node = InsightNode(
                    summary=summary,
                    confidence=confidence,
                    last_updated=now,
                    embedding=None
                )
                self.insights[node.node_id] = node

            if embedding:
                self._embeddings_cache[node.node_id] = embedding
            
            # 로그 기록
            self._append_log("UPSERT_NODE", {"category": "insights", "data": node.to_dict()})
            if embedding:
                self._append_log("UPSERT_EMBEDDING", {"node_id": node.node_id, "vector": embedding})
                
            return node

    def add_or_update_note(self, summary: str, note_type: str = "narrative",
                           tags: Optional[List[str]] = None, confidence: float = 0.5,
                           related_entity_ids: Optional[List[str]] = None,
                           evidence_episode_ids: Optional[List[str]] = None,
                           embedding: Optional[list] = None) -> NoteNode:
        with self._lock:
            tags = list(dict.fromkeys(tags or []))
            related_entity_ids = list(dict.fromkeys(related_entity_ids or []))
            evidence_episode_ids = list(dict.fromkeys(evidence_episode_ids or []))
            normalized = self._normalize_text(summary)

            node = None
            for existing in self.notes.values():
                if existing.note_type != note_type:
                    continue
                if self._normalize_text(existing.summary) == normalized:
                    node = existing
                    break

            if node:
                node.summary = summary
                node.confidence = max(node.confidence, confidence)
                node.tags = list(dict.fromkeys(node.tags + tags))
                node.related_entity_ids = list(dict.fromkeys(node.related_entity_ids + related_entity_ids))
                node.evidence_episode_ids = list(dict.fromkeys(node.evidence_episode_ids + evidence_episode_ids))
            else:
                node = NoteNode(
                    note_type=note_type,
                    summary=summary,
                    tags=tags,
                    confidence=confidence,
                    related_entity_ids=related_entity_ids,
                    evidence_episode_ids=evidence_episode_ids,
                    embedding=None,
                )
                self.notes[node.node_id] = node

            self._append_log("UPSERT_NODE", {"category": "notes", "data": node.to_dict()})
            if embedding:
                self._embeddings_cache[node.node_id] = embedding
                self._append_log("UPSERT_EMBEDDING", {"node_id": node.node_id, "vector": embedding})

            return node

    def upsert_claim(self, subject_id: str, facet: str, value: Optional[Dict[str, Any]] = None,
                     qualifiers: Optional[Dict[str, Any]] = None, nl_summary: str = "",
                     source_type: str = "explicit", confidence: float = 0.5,
                     status: str = "active", valid_from: Optional[float] = None,
                     valid_to: Optional[float] = None, last_confirmed_at: Optional[float] = None,
                     evidence_episode_ids: Optional[List[str]] = None,
                     sensitivity: Optional[str] = None, scope: str = "user_private",
                     embedding: Optional[list] = None) -> ClaimNode:
        with self._lock:
            value = value or {}
            qualifiers = qualifiers or {}
            scope = self._normalize_claim_scope(scope)
            qualifiers = self._normalize_claim_qualifiers(subject_id, value, qualifiers, scope)
            evidence_episode_ids = list(dict.fromkeys(evidence_episode_ids or []))
            spec = get_facet_spec(facet)
            merge_key = self._build_claim_merge_key(subject_id, facet, value, qualifiers)
            now = time.time()
            effective_last_confirmed = last_confirmed_at if last_confirmed_at is not None else now
            normalized_valid_from, normalized_valid_to = self._normalize_claim_window(
                value, qualifiers, valid_from, valid_to
            )

            matching = [
                claim for claim in self.claims.values()
                if claim.subject_id == subject_id and claim.facet == facet and claim.merge_key == merge_key
            ]
            if spec.merge_policy == "interval":
                matching = self._dedupe_claims(matching + self._find_interval_matches(
                    subject_id, facet, value, qualifiers, normalized_valid_from, normalized_valid_to
                ))

            active_matching = [claim for claim in matching if claim.status == "active"]

            node = self._select_existing_claim(matching, active_matching, spec.merge_policy, status)
            if spec.merge_policy == "replace" and status == "active":
                self._supersede_claims([
                    claim for claim in self.claims.values()
                    if claim.subject_id == subject_id and claim.facet == facet and claim.status == "active"
                    and claim.merge_key != merge_key
                    and (not node or claim.node_id != node.node_id)
                ])
            elif spec.merge_policy in {"statusful", "sticky", "hypothesis"}:
                self._supersede_claims([
                    claim for claim in active_matching
                    if node and claim.node_id != node.node_id
                ])
            elif spec.merge_policy == "multi_active":
                self._supersede_claims([
                    claim for claim in active_matching
                    if node and claim.node_id != node.node_id
                ])
            elif spec.merge_policy == "interval" and status == "active":
                self._supersede_claims([
                    claim for claim in active_matching
                    if node and claim.node_id != node.node_id
                ])

            if node:
                node.merge_key = merge_key
                node.value = self._merge_dict(node.value, value, spec.merge_policy)
                node.qualifiers = self._merge_dict(node.qualifiers, qualifiers, spec.merge_policy)
                node.nl_summary = nl_summary or node.nl_summary
                node.source_type = source_type or node.source_type
                node.confidence = max(node.confidence, confidence)
                node.status = status or node.status
                if spec.merge_policy == "interval" and status == "active":
                    node.valid_from = self._merge_interval_start(node.valid_from, normalized_valid_from)
                    node.valid_to = self._merge_interval_end(node.valid_to, normalized_valid_to)
                    self._update_interval_payload(node.value, node.valid_from, node.valid_to)
                else:
                    node.valid_from = normalized_valid_from if normalized_valid_from is not None else node.valid_from
                    node.valid_to = normalized_valid_to if normalized_valid_to is not None else node.valid_to
                node.last_confirmed_at = effective_last_confirmed
                node.evidence_episode_ids = list(dict.fromkeys(node.evidence_episode_ids + evidence_episode_ids))
                node.sensitivity = sensitivity or node.sensitivity
                node.scope = scope or node.scope
            else:
                node = ClaimNode(
                    subject_id=subject_id,
                    facet=facet,
                    merge_key=merge_key,
                    value=value,
                    qualifiers=qualifiers,
                    nl_summary=nl_summary,
                    source_type=source_type,
                    confidence=confidence,
                    status=status,
                    valid_from=normalized_valid_from,
                    valid_to=normalized_valid_to,
                    last_confirmed_at=effective_last_confirmed,
                    evidence_episode_ids=evidence_episode_ids,
                    sensitivity=sensitivity or spec.default_sensitivity,
                    scope=scope,
                    embedding=None,
                )
                if spec.merge_policy == "interval":
                    self._update_interval_payload(node.value, node.valid_from, node.valid_to)
                self.claims[node.node_id] = node

            self._append_log("UPSERT_NODE", {"category": "claims", "data": self._claim_to_public_graph_payload(node)})
            if embedding:
                self._embeddings_cache[node.node_id] = embedding
                self._append_log("UPSERT_EMBEDDING", {"node_id": node.node_id, "vector": embedding})

            return node
    
    def get_or_create_user(self, user_id: str, nickname: str) -> EntityNode:
        with self._lock:
            target_node = None
            is_new = False
            
            # 검색
            for node in self.entities.values():
                if node.user_id == str(user_id):
                    target_node = node
                    break
            
            # 생성
            if not target_node:
                target_node = EntityNode(user_id=str(user_id), nickname=nickname)
                self.entities[target_node.node_id] = target_node
                is_new = True
            
            # 업데이트 (닉네임 변경 or 신규)
            if is_new or (nickname and target_node.nickname != nickname):
                if nickname: target_node.nickname = nickname
                # 로그 기록 (상태 변경 시에만)
                self._append_log("UPSERT_NODE", {"category": "entities", "data": target_node.to_dict()})
                
            return target_node

    def connect_nodes(self, source_id: str, target_id: str, weight: float = 1.0):
        with self._lock:
            # 1. 메모리 업데이트
            source = self.get_node(source_id)
            if source:
                source.edges[target_id] = weight
                
            # 2. 로그 기록
            self._append_log("ADD_EDGE", {"source": source_id, "target": target_id, "weight": weight})

    def link_user_to_episode(self, user_id_str: str, episode_node_id: str):
        """Helper: 유저 ID로 Entity를 찾아 에피소드와 연결"""
        with self._lock:
            # Entity 찾기 (없으면 생성)
            # 여기서는 닉네임을 모르므로 빈 문자열. 기존 닉네임 유지.
            user_node = self.get_or_create_user(user_id_str, "")
            
            # 양방향 연결
            self.connect_nodes(user_node.node_id, episode_node_id, 1.0)
            self.connect_nodes(episode_node_id, user_node.node_id, 1.0)

    def update_affinity(self, user_id: str, delta: float):
        with self._lock:
            node = self.get_or_create_user(user_id, "")
            node.affinity = max(0.0, min(100.0, node.affinity + delta))
            # 로그 기록
            self._append_log("UPSERT_NODE", {"category": "entities", "data": node.to_dict()})

    def get_node(self, node_id: str) -> Union[EpisodeNode, InsightNode, NoteNode, ClaimNode, EntityNode, None]:
        with self._lock:
            return self.episodes.get(node_id) or \
                   self.insights.get(node_id) or \
                   self.notes.get(node_id) or \
                   self.claims.get(node_id) or \
                   self.entities.get(node_id)
    
    def get_all_nodes(self) -> List[Any]:
        with self._lock:
            all_nodes = []
            for store in [self.episodes, self.insights, self.notes, self.claims, self.entities]:
                for node in store.values():
                    # 임베딩 임시 주입 (검색용)
                    node.embedding = self._embeddings_cache.get(node.node_id)
                    all_nodes.append(node)
            return all_nodes

    def _normalize_text(self, text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    def _normalize_merge_value(self, value: Any) -> str:
        if isinstance(value, list):
            return ",".join(sorted(self._normalize_merge_value(item) for item in value))
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        if value is None:
            return ""
        return str(value).strip().lower()

    def _build_claim_merge_key(self, subject_id: str, facet: str, value: Dict[str, Any],
                               qualifiers: Dict[str, Any]) -> str:
        spec = get_facet_spec(facet)
        if not spec.key_fields:
            return f"{subject_id}|{facet}"

        parts = [subject_id, facet]
        for field in spec.key_fields:
            field_value = value.get(field)
            if field_value is None:
                field_value = qualifiers.get(field)
            parts.append(f"{field}={self._normalize_merge_value(field_value)}")
        return "|".join(parts)

    def _merge_dict(self, original: Dict[str, Any], incoming: Dict[str, Any], merge_policy: str) -> Dict[str, Any]:
        if merge_policy == "set_union":
            merged = dict(original)
            for key, value in incoming.items():
                existing = merged.get(key)
                if isinstance(existing, list) or isinstance(value, list):
                    existing_list = existing if isinstance(existing, list) else ([existing] if existing else [])
                    value_list = value if isinstance(value, list) else ([value] if value else [])
                    merged[key] = list(dict.fromkeys(existing_list + value_list))
                elif existing and value and existing != value:
                    merged[key] = list(dict.fromkeys([existing, value]))
                elif value is not None:
                    merged[key] = value
            return merged

        merged = dict(original)
        for key, value in incoming.items():
            if value is not None:
                merged[key] = value
        return merged

    def _normalize_claim_window(self, value: Dict[str, Any], qualifiers: Dict[str, Any],
                                valid_from: Optional[float], valid_to: Optional[float]) -> tuple[Optional[float], Optional[float]]:
        start = self._coerce_timestamp(
            valid_from
            if valid_from is not None else value.get("start_at") or qualifiers.get("start_at")
        )
        end = self._coerce_timestamp(
            valid_to
            if valid_to is not None else value.get("end_at") or qualifiers.get("end_at")
        )
        return start, end

    def _coerce_timestamp(self, raw_value: Any) -> Optional[float]:
        if raw_value is None or raw_value == "":
            return None
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
        if isinstance(raw_value, str):
            try:
                return datetime.fromisoformat(raw_value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return None
        return None

    def _find_interval_matches(self, subject_id: str, facet: str, value: Dict[str, Any],
                               qualifiers: Dict[str, Any], valid_from: Optional[float],
                               valid_to: Optional[float]) -> List[ClaimNode]:
        title = self._normalize_merge_value(value.get("title") or qualifiers.get("title"))
        if not title:
            return []

        matches: List[ClaimNode] = []
        for claim in self.claims.values():
            if claim.subject_id != subject_id or claim.facet != facet:
                continue
            existing_title = self._normalize_merge_value(
                claim.value.get("title") or claim.qualifiers.get("title")
            )
            if existing_title != title:
                continue

            existing_start, existing_end = self._normalize_claim_window(
                claim.value, claim.qualifiers, claim.valid_from, claim.valid_to
            )
            if self._intervals_overlap(existing_start, existing_end, valid_from, valid_to):
                matches.append(claim)
        return matches

    def _intervals_overlap(self, start_a: Optional[float], end_a: Optional[float],
                           start_b: Optional[float], end_b: Optional[float]) -> bool:
        if start_a is None or start_b is None:
            return False
        resolved_end_a = end_a if end_a is not None else start_a
        resolved_end_b = end_b if end_b is not None else start_b
        return start_a <= resolved_end_b and start_b <= resolved_end_a

    def _dedupe_claims(self, claims: List[ClaimNode]) -> List[ClaimNode]:
        seen = set()
        result = []
        for claim in claims:
            if claim.node_id in seen:
                continue
            seen.add(claim.node_id)
            result.append(claim)
        return result

    def _select_existing_claim(self, matching: List[ClaimNode], active_matching: List[ClaimNode],
                               merge_policy: str, incoming_status: str) -> Optional[ClaimNode]:
        ordered_active = sorted(active_matching, key=self._claim_sort_key, reverse=True)
        if ordered_active:
            return ordered_active[0]
        if merge_policy in {"multi_active", "interval"} or incoming_status != "active":
            ordered_matching = sorted(matching, key=self._claim_sort_key, reverse=True)
            if ordered_matching:
                return ordered_matching[0]
        return None

    def _claim_sort_key(self, claim: ClaimNode) -> tuple[float, float]:
        return (claim.last_confirmed_at or 0.0, claim.valid_to or 0.0)

    def _merge_interval_start(self, existing: Optional[float], incoming: Optional[float]) -> Optional[float]:
        if existing is None:
            return incoming
        if incoming is None:
            return existing
        return min(existing, incoming)

    def _merge_interval_end(self, existing: Optional[float], incoming: Optional[float]) -> Optional[float]:
        if existing is None:
            return incoming
        if incoming is None:
            return existing
        return max(existing, incoming)

    def _update_interval_payload(self, payload: Dict[str, Any], valid_from: Optional[float], valid_to: Optional[float]):
        if valid_from is not None:
            payload["start_at"] = valid_from
        if valid_to is not None:
            payload["end_at"] = valid_to

    def _supersede_claims(self, claims: List[ClaimNode]):
        for claim in claims:
            if claim.status == "active":
                claim.status = "superseded"
                claim.last_confirmed_at = time.time()
                self._append_log("UPSERT_NODE", {"category": "claims", "data": self._claim_to_public_graph_payload(claim)})

    def _normalize_claim_scope(self, scope: Optional[str]) -> str:
        normalized = str(scope or "user_private").strip().lower()
        if normalized == "shared":
            return "participants"
        return normalized or "user_private"

    def _normalize_claim_qualifiers(self, subject_id: str, value: Dict[str, Any],
                                    qualifiers: Dict[str, Any], scope: str,
                                    inferred_audience_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        normalized = dict(qualifiers or {})
        if scope != "participants":
            return normalized

        audience_ids = [
            str(audience_id)
            for audience_id in normalized.get("audience_ids", [])
            if audience_id
        ]
        if not audience_ids:
            audience_ids.append(str(subject_id))
            target_entity_id = value.get("target_entity_id")
            if target_entity_id:
                audience_ids.append(str(target_entity_id))
            for audience_id in inferred_audience_ids or []:
                if audience_id:
                    audience_ids.append(str(audience_id))
        normalized["audience_ids"] = list(dict.fromkeys(audience_ids))
        return normalized

    def _claim_to_public_graph_payload(self, claim: ClaimNode) -> Dict[str, Any]:
        data = claim.to_graph_dict()
        data.pop("embedding", None)
        return data

    def _sanitize_persisted_claim_data(self, data: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
        raw_data = dict(data or {})
        if raw_data.get("facet") != "boundary.rule":
            return raw_data, False

        sanitized = ClaimNode(**raw_data).to_graph_dict()
        sanitized.pop("embedding", None)
        raw_without_embedding = dict(raw_data)
        raw_without_embedding.pop("embedding", None)
        changed = json.dumps(raw_without_embedding, ensure_ascii=False, sort_keys=True) != json.dumps(
            sanitized,
            ensure_ascii=False,
            sort_keys=True,
        )
        return sanitized, changed

    # =================================================================
    # Persistence Engine (Snapshot & Delta)
    # =================================================================

    def checkpoint(self):
        """
        [Compaction]
        현재 메모리 상태를 Snapshot(JSON)으로 덤프하고, Delta Log를 비웁니다.
        봇 종료 시나 주기적으로 호출합니다.
        """
        with self._lock:
            print("💾 Checkpointing LTM... (Compacting logs)")
            self._save_snapshot() # 전체 덤프
            self._clear_log()     # 로그 비우기
            print("✅ Checkpoint Complete.")
    
    def save_all(self):
        """Alias for checkpoint - saves current state to disk"""
        self.checkpoint()
    
    def compact(self):
        """Alias for checkpoint - compacts logs into snapshot"""
        self.checkpoint()

    def _append_log(self, action: str, payload: Dict):
        """Delta Log 파일에 변경 사항 한 줄 추가 (Append Only)"""
        entry = {
            "timestamp": time.time(),
            "action": action,
            "payload": payload
        }
        try:
            with open(self.delta_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"❌ Log Write Error: {e}")

    def _load_all(self):
        """1. Snapshot 로드 -> 2. Delta Log 리플레이"""
        # 1. Load Snapshot
        self._load_snapshot()
        
        # 2. Replay Log
        if os.path.exists(self.delta_path):
            print(f"🔄 Replaying Delta Log: {self.delta_path}")
            try:
                with open(self.delta_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            entry = json.loads(line)
                            self._apply_log_entry(entry)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"❌ Log Replay Error: {e}")
        self._migrate_legacy_claim_scopes()
        self._migrate_sensitive_boundary_claims_in_persistence()

    def _apply_log_entry(self, entry: Dict):
        """로그 한 줄을 메모리에 반영"""
        action = entry.get("action")
        payload = entry.get("payload", {})
        
        if action == "UPSERT_NODE":
            cat = payload.get("category")
            data = payload.get("data")
            if not data:
                return
            node_id = data.get("node_id")
            
            # 객체 복원
            if cat == "episodes":
                node = EpisodeNode(**data)
                self.episodes[node_id] = node
            elif cat == "insights":
                node = InsightNode(**data)
                self.insights[node_id] = node
            elif cat == "notes":
                node = NoteNode(**data)
                self.notes[node_id] = node
            elif cat == "claims":
                sanitized_data, changed = self._sanitize_persisted_claim_data(data)
                if changed:
                    self._boundary_persistence_migration_needed = True
                node = ClaimNode(**sanitized_data)
                node.scope = self._normalize_claim_scope(node.scope)
                self.claims[node_id] = node
            elif cat == "entities":
                node = EntityNode(**data)
                self.entities[node_id] = node

        elif action == "UPSERT_EMBEDDING":
            node_id = payload.get("node_id")
            vector = payload.get("vector")
            self._embeddings_cache[node_id] = vector

        elif action == "ADD_EDGE":
            src = payload.get("source")
            tgt = payload.get("target")
            weight = payload.get("weight", 1.0)
            node = self.get_node(src)
            if node:
                node.edges[tgt] = weight

    def _save_snapshot(self):
        """메모리 전체를 JSON 파일로 저장 (Overwrite)"""
        data = {
            "episodes": {k: v.to_dict() for k, v in self.episodes.items()},
            "insights": {k: v.to_dict() for k, v in self.insights.items()},
            "notes": {k: v.to_dict() for k, v in self.notes.items()},
            "claims": {k: self._claim_to_public_graph_payload(v) for k, v in self.claims.items()},
            "entities": {k: v.to_dict() for k, v in self.entities.items()}
        }
        # 임베딩 필드 제거
        for cat in data.values():
            for node in cat.values():
                node.pop("embedding", None)

        try:
            # Graph Snapshot
            with open(self.graph_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Embeddings Snapshot (임베딩은 로그로 관리하기엔 너무 크니 스냅샷 위주로 관리해도 됨)
            # 여기서는 편의상 임베딩도 함께 덤프
            with open(self.embeddings_path, "w", encoding="utf-8") as f:
                json.dump(self._embeddings_cache, f)
        except Exception as e:
            print(f"❌ Snapshot Save Error: {e}")

    def _clear_log(self):
        """로그 파일 초기화"""
        open(self.delta_path, "w").close()

    def _load_snapshot(self):
        """기존 JSON 파일 로드"""
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                for k, v in data.get("episodes", {}).items():
                    self.episodes[k] = EpisodeNode(**v)
                for k, v in data.get("insights", {}).items():
                    self.insights[k] = InsightNode(**v)
                for k, v in data.get("notes", {}).items():
                    self.notes[k] = NoteNode(**v)
                for k, v in data.get("claims", {}).items():
                    sanitized_data, changed = self._sanitize_persisted_claim_data(v)
                    if changed:
                        self._boundary_persistence_migration_needed = True
                    claim = ClaimNode(**sanitized_data)
                    claim.scope = self._normalize_claim_scope(claim.scope)
                    self.claims[k] = claim
                for k, v in data.get("entities", {}).items():
                    self.entities[k] = EntityNode(**v)
            except Exception as e:
                print(f"❌ Graph Snapshot Load Error: {e}")
                
        if os.path.exists(self.embeddings_path):
            try:
                with open(self.embeddings_path, "r", encoding="utf-8") as f:
                    self._embeddings_cache = json.load(f)
            except Exception as e:
                print(f"❌ Embeddings Snapshot Load Error: {e}")

    def _migrate_legacy_claim_scopes(self):
        entity_ids_by_node = {
            node_id: entity.user_id
            for node_id, entity in self.entities.items()
        }
        for claim in self.claims.values():
            inferred_audience_ids = [
                entity_ids_by_node[edge_id]
                for edge_id in claim.edges
                if edge_id in entity_ids_by_node
            ]
            claim.scope = self._normalize_claim_scope(claim.scope)
            claim.qualifiers = self._normalize_claim_qualifiers(
                claim.subject_id,
                claim.value,
                claim.qualifiers,
                claim.scope,
                inferred_audience_ids=inferred_audience_ids,
            )

    def _migrate_sensitive_boundary_claims_in_persistence(self):
        if not self._boundary_persistence_migration_needed:
            return
        print("🔐 Scrubbing sensitive boundary payloads from graph persistence...")
        self._save_snapshot()
        self._clear_log()
        self._boundary_persistence_migration_needed = False
