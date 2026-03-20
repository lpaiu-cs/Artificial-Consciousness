import time
import threading
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

import config
from memory.canonical_store import CanonicalMemoryStore
from memory.ontology import Facet, FACET_SPECS, get_facet_spec
from memory_structures import MemoryObject
from modules.ltm_graph import MemoryGraph
from api_client import UnifiedAPIClient

class ReflectionHandler:
    """
    [Background Process]
    STM에서 방출(Eviction)된 기억들을 주기적으로 가져와 LTM에 저장합니다.
    단순 저장이 아니라, LLM을 통해 '성찰(Reflection)'하여 구조화된 데이터로 변환합니다.
    """

    def __init__(self, graph_db: MemoryGraph, api_client: UnifiedAPIClient,
                 canonical_store: CanonicalMemoryStore = None):
        self.graph = graph_db
        self.api = api_client
        self.canonical_store = canonical_store
        self.stop_event = threading.Event()
        self.thread = None
        self._pending_memories: List[MemoryObject] = []
        self._pending_lock = threading.Lock()

    def start_background_loop(self, eviction_buffer: List[MemoryObject], interval: int = 30):
        """STM의 eviction_buffer를 주기적으로 감시하는 데몬 스레드 시작"""
        if self.thread and self.thread.is_alive():
            return

        def loop():
            while not self.stop_event.is_set():
                if self.stop_event.wait(interval):
                    break
                batch = []
                if eviction_buffer:
                    batch.extend(eviction_buffer[:])
                    eviction_buffer.clear()
                batch.extend(self._drain_pending_memories())

                if not batch:
                    continue

                try:
                    self._process_batch(batch)
                except Exception as e:
                    logging.error(f"❌ Reflection Error: {e}")

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        print("🌙 Reflection Handler Started (Background)")

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()

    def submit_memories(self, memories: List[MemoryObject]):
        """즉시 반영이 필요한 assistant/user 발화를 reflection 큐에 추가한다."""
        if not memories:
            return
        with self._pending_lock:
            self._pending_memories.extend(memories)

    def _drain_pending_memories(self) -> List[MemoryObject]:
        with self._pending_lock:
            batch = self._pending_memories[:]
            self._pending_memories.clear()
            return batch

    def _process_batch(self, memories: List[MemoryObject]):
        """
        1. LLM 분석: 요약, 감정 추출, 통찰 추출
        2. 임베딩 생성: 검색을 위한 벡터화
        3. 그래프 저장: Node 생성 및 Edge 연결 (3-Tier Wiring)
        """
        if not memories: 
            return

        # 1. LLM 분석 (Analyze)
        analysis_result = self._analyze_with_llm(memories)
        if not analysis_result:
            return

        episode_summary = analysis_result.get("episode_summary", "")
        dominant_emotion = analysis_result.get("dominant_emotion", "neutral")
        claims_data = analysis_result.get("claims", [])
        notes_data = analysis_result.get("notes", [])
        legacy_insights = analysis_result.get("insights", [])

        # 2. 관련 Entity 식별 (Identify Entities)
        # 배치에 포함된 모든 유저의 ID와 닉네임을 수집
        involved_users = {} # {user_id: nickname}
        for mem in memories:
            involved_users[mem.user_id] = mem.user_name
            # (추후 확장) mem.related_users에 있는 제3자도 포함 가능

        # 3. 그래프 저장 및 연결 (Persist & Wire)
        
        # (A) Episode Node 생성
        # 검색을 위해 요약문에 대한 임베딩 생성
        ep_embedding = self.api.get_embedding(episode_summary)
        
        # 대표 화자(Primary User) 설정 - 보통 첫 번째 메시지의 유저
        primary_user_id = next((mem.user_id for mem in memories if mem.role == "user"), memories[0].user_id)
        
        episode_node = self.graph.add_episode(
            content=episode_summary,
            user_id=primary_user_id,
            emotion=dominant_emotion, # 자연어 감정 보존
            embedding=ep_embedding
        )

        # (B) Entity Node 연결 (PARTICIPATED_IN)
        for uid, nickname in involved_users.items():
            # Entity 확보 (없으면 생성, 닉네임 최신화)
            self.graph.get_or_create_user(uid, nickname)
            # Edge 연결
            self.graph.link_user_to_episode(uid, episode_node.node_id)

        # (C) Claim Node 생성 및 연결
        for claim_payload in claims_data:
            self._persist_claim(claim_payload, episode_node.node_id, involved_users, primary_user_id)

        # (D) Note Node 생성 및 연결
        for note_payload in notes_data:
            self._persist_note(note_payload, episode_node.node_id, involved_users, primary_user_id)

        # (E) Legacy Insight fallback
        for insight_text in legacy_insights:
            in_embedding = self.api.get_embedding(insight_text)
            insight_node = self.graph.add_or_update_insight(
                summary=insight_text,
                confidence=0.6,
                embedding=in_embedding
            )
            self.graph.connect_nodes(insight_node.node_id, episode_node.node_id, weight=config.EVIDENCE_EDGE_TO_EPISODE)
            self.graph.connect_nodes(episode_node.node_id, insight_node.node_id, weight=config.EVIDENCE_EDGE_TO_INSIGHT)
            for uid in involved_users:
                entity_id = self.graph.get_or_create_user(uid, involved_users[uid]).node_id
                self.graph.connect_nodes(insight_node.node_id, entity_id, weight=1.0)

        # 4. 저장 확정 (File I/O)
        self.graph.save_all()
        # print(f"💾 Reflected: {episode_summary[:30]}... (Claims: {len(claims_data)}, Notes: {len(notes_data)})")

    def _persist_claim(self, claim_payload: Dict[str, Any], episode_node_id: str,
                       involved_users: Dict[str, str], primary_user_id: str):
        subject_id = str(claim_payload.get("subject_id") or primary_user_id)
        facet = str(claim_payload.get("facet") or "")
        if not facet:
            return

        value = claim_payload.get("value") or {}
        qualifiers = claim_payload.get("qualifiers") or {}
        nl_summary = claim_payload.get("nl_summary") or claim_payload.get("summary") or ""
        if not nl_summary:
            return

        spec = get_facet_spec(facet)
        embedding = self.api.get_embedding(nl_summary)
        claim_node = self.graph.upsert_claim(
            subject_id=subject_id,
            facet=facet,
            value=value,
            qualifiers=qualifiers,
            nl_summary=nl_summary,
            source_type=claim_payload.get("source_type", "explicit"),
            confidence=float(claim_payload.get("confidence", 0.8)),
            status=claim_payload.get("status", "active"),
            valid_from=self._coerce_timestamp(claim_payload.get("valid_from")),
            valid_to=self._coerce_timestamp(claim_payload.get("valid_to")),
            last_confirmed_at=time.time(),
            evidence_episode_ids=[episode_node_id],
            sensitivity=claim_payload.get("sensitivity", spec.default_sensitivity),
            scope=claim_payload.get("scope", "user_private"),
            embedding=embedding,
        )
        if self.canonical_store:
            self.canonical_store.upsert_claim(claim_node)
            self.canonical_store.upsert_open_loop_from_claim(claim_node)

        self.graph.connect_nodes(claim_node.node_id, episode_node_id, weight=config.EVIDENCE_EDGE_TO_EPISODE)
        self.graph.connect_nodes(episode_node_id, claim_node.node_id, weight=config.EVIDENCE_EDGE_TO_INSIGHT)

        subject_entity = self.graph.get_or_create_user(subject_id, involved_users.get(subject_id, ""))
        self.graph.connect_nodes(claim_node.node_id, subject_entity.node_id, weight=1.2)

        for uid, nickname in involved_users.items():
            entity_id = self.graph.get_or_create_user(uid, nickname).node_id
            self.graph.connect_nodes(claim_node.node_id, entity_id, weight=0.6)

    def _persist_note(self, note_payload: Dict[str, Any], episode_node_id: str,
                      involved_users: Dict[str, str], primary_user_id: str):
        summary = note_payload.get("summary", "")
        if not summary:
            return

        related_entity_ids = [str(entity_id) for entity_id in note_payload.get("related_entity_ids", []) if entity_id]
        if not related_entity_ids:
            related_entity_ids = [primary_user_id]

        embedding = self.api.get_embedding(summary)
        note_node = self.graph.add_or_update_note(
            summary=summary,
            note_type=note_payload.get("note_type", "narrative"),
            tags=note_payload.get("tags", []),
            confidence=float(note_payload.get("confidence", 0.6)),
            related_entity_ids=related_entity_ids,
            evidence_episode_ids=[episode_node_id],
            embedding=embedding
        )

        self.graph.connect_nodes(note_node.node_id, episode_node_id, weight=config.EVIDENCE_EDGE_TO_EPISODE)
        self.graph.connect_nodes(episode_node_id, note_node.node_id, weight=config.EVIDENCE_EDGE_TO_INSIGHT)

        for uid in set(related_entity_ids + list(involved_users.keys())):
            entity_id = self.graph.get_or_create_user(uid, involved_users.get(uid, "")).node_id
            self.graph.connect_nodes(note_node.node_id, entity_id, weight=0.8)

    def _analyze_with_llm(self, memories: List[MemoryObject]) -> Dict[str, Any]:
        """
        [System 2] GPT-4를 이용해 파편화된 대화 로그를 구조화된 데이터로 변환
        """
        # 로그 텍스트 변환
        logs_text = ""
        for m in memories:
            logs_text += f"[{m.user_name}({m.user_id})]: {m.content} (EmotionTag: {m.emotion_tag})\n"

        supported_facets = ", ".join(spec.name for spec in FACET_SPECS.values() if spec.name != Facet.TRAIT_HYPOTHESIS.value)

        system_prompt = f"""
        You are a generic 'Memory Manager' for an AI.
        Your job is to consolidate raw chat logs into a meaningful memory structure.
        Treat ontology as a merge-policy table: only extract stable, update-worthy claims as canonical state.
        
        [Output Format]
        Return a JSON object with the following fields:
        1. "episode_summary": A concise, 1-sentence summary of the conversation event. (e.g. "Mincho-dan discussed his preference for mint chocolate.")
        2. "dominant_emotion": The overall emotional tone of the user in this interaction. Use a natural language phrase (under 10 chars). (e.g. "Passionately defensive", "Calm curiosity")
        3. "claims": A list of canonical state updates. Each item must include:
           - "subject_id": the speaker/user id the claim is about
           - "facet": one of [{supported_facets}]
           - "value": object
           - "qualifiers": object
           - "source_type": one of ["explicit", "inferred", "assistant_commitment"]
           - "confidence": 0.0-1.0
           - "status": one of ["active", "superseded", "retracted", "uncertain", "expired"]
           - "nl_summary": short natural-language paraphrase
           - optional "valid_from", "valid_to", "sensitivity", "scope"
        4. "notes": A list of non-canonical notes. Each item must include:
           - "note_type": one of ["narrative", "theme", "inside_joke", "impression", "repair"]
           - "summary": short summary
           - "tags": list of strings
           - "confidence": 0.0-1.0
           - "related_entity_ids": list of user ids
        5. "insights": keep as an empty list for backward compatibility unless you truly cannot structure the output.
        
        [Constraint]
        - Analyze purely based on the logs.
        - The language of summary, claims, and notes must match the language of the logs (Korean).
        - Store direct interaction preferences, names, commitments, schedules, and boundaries as claims when explicit.
        - Store preference updates, goals, relations, and constraints as claims only if explicit or strongly grounded in the logs.
        - Personality traits must never become canonical claims. Put them only in notes with note_type="impression" and tag "trait_hypothesis".
        - Jokes, fleeting emotions, rhetoric, and uncertain third-party facts should stay in notes, not claims.
        - If there is no canonical claim, return an empty "claims" list instead of guessing.
        - For assistant promises or follow-ups, use facet "commitment.open_loop" with source_type "assistant_commitment".
        - When time expressions are vague and cannot be normalized from the logs alone, keep them in notes instead of emitting a schedule claim.
        """
        
        user_prompt = f"""
        [Raw Logs]
        {logs_text}
        
        Analyze and Extract JSON:
        """

        try:
            # chat_slow에 json_mode=True 옵션 사용 권장
            response = self.api.chat_slow(system_prompt, user_prompt, json_mode=True)
            if isinstance(response, str):
                 response = json.loads(response)
            return self._normalize_analysis_result(response, memories)
        except Exception as e:
            logging.error(f"Reflection LLM Parsing Error: {e}")
            return {}

    def _coerce_timestamp(self, raw_value: Any) -> Any:
        if raw_value is None or raw_value == "":
            return None
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
        if isinstance(raw_value, str):
            try:
                return datetime.fromisoformat(raw_value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return raw_value
        return raw_value

    def _normalize_analysis_result(self, response: Dict[str, Any], memories: List[MemoryObject]) -> Dict[str, Any]:
        normalized = {
            "episode_summary": response.get("episode_summary", ""),
            "dominant_emotion": response.get("dominant_emotion", "neutral"),
            "claims": [],
            "notes": [],
            "insights": response.get("insights", []),
        }

        known_ids = {str(mem.user_id) for mem in memories}
        primary_user_id = next((mem.user_id for mem in memories if mem.role == "user"), memories[0].user_id)

        for claim in response.get("claims", []):
            if not isinstance(claim, dict):
                continue
            facet = str(claim.get("facet", "")).strip()
            if facet == Facet.TRAIT_HYPOTHESIS.value:
                continue
            if not facet:
                continue
            normalized["claims"].append({
                "subject_id": str(claim.get("subject_id") or primary_user_id),
                "facet": facet,
                "value": claim.get("value") or {},
                "qualifiers": claim.get("qualifiers") or {},
                "source_type": claim.get("source_type", "explicit"),
                "confidence": float(claim.get("confidence", 0.8)),
                "status": claim.get("status", "active"),
                "nl_summary": claim.get("nl_summary") or claim.get("summary") or "",
                "valid_from": claim.get("valid_from"),
                "valid_to": claim.get("valid_to"),
                "sensitivity": claim.get("sensitivity"),
                "scope": claim.get("scope", "user_private"),
            })

        for note in response.get("notes", []):
            if not isinstance(note, dict):
                continue
            related_entity_ids = [
                str(entity_id) for entity_id in note.get("related_entity_ids", [])
                if str(entity_id) in known_ids
            ]
            normalized["notes"].append({
                "note_type": note.get("note_type", "narrative"),
                "summary": note.get("summary", ""),
                "tags": note.get("tags", []),
                "confidence": float(note.get("confidence", 0.6)),
                "related_entity_ids": related_entity_ids or [primary_user_id],
            })

        return normalized
