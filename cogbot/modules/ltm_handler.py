import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from cogbot.memory.canonical_store import CanonicalMemoryStore
from cogbot.memory.query_planner import QueryPlanner
from cogbot.memory.schema import ContextBundle
from cogbot.memory_structures import ClaimNode, RetrievalQuery, EpisodeNode, InsightNode, NoteNode, EntityNode, BaseNode
from cogbot.modules.ltm_graph import MemoryGraph
from cogbot import config

class LongTermMemory:
    """
    [System 2: LTM Retrieval]
    3-Tier Graph 구조를 활용하여 '의미적(Vector)'이고 '맥락적(Graph)'인 기억을 인출합니다.
    
    Process:
    1. Resolve Entity: 쿼리 유저의 Entity Node ID(UUID)를 식별
    2. Anchoring: 벡터 유사도로 후보군 검색 + 접근 권한(Edge) 체크
    3. Spreading: 그래프 엣지를 타고 맥락 확장
    4. Reranking: 점수 재조정 (시간, 기분, 키워드)
    """
    def __init__(self, graph_db: MemoryGraph, api_client, canonical_store: CanonicalMemoryStore = None):
        self.graph = graph_db
        self.api = api_client # (Vector DB 도입 시 사용)
        self.store = canonical_store

    def retrieve(self, query: RetrievalQuery, top_k: int = 5) -> List[BaseNode]:
        bundle = self.build_context_bundle(query, top_k=top_k)
        merged: List[BaseNode] = []
        for group in [
            bundle.open_loops,
            bundle.active_claims,
            bundle.relevant_schedule,
            bundle.supporting_notes,
            bundle.supporting_events,
            bundle.legacy_insights,
        ]:
            for node in group:
                if any(existing.node_id == node.node_id for existing in merged):
                    continue
                merged.append(node)
                if len(merged) >= top_k:
                    return merged
        return merged

    def build_context_bundle(self, query: RetrievalQuery, top_k: int = 5,
                             session_referents: Optional[List[Dict[str, Any]]] = None) -> ContextBundle:
        query_text = query.query_text or " ".join(query.keywords)
        plan = QueryPlanner.plan(
            query_text,
            query.user_id,
            known_entities=self._build_known_entities(query.user_id),
            session_referents=session_referents or [],
        )
        open_loop_nodes = self._load_open_loop_nodes(query.user_id, query_text, limit=3)
        boundary_claims = self._load_claim_nodes(
            query.user_id,
            ["boundary.rule"],
            query_text,
            limit=3,
            viewer_id=query.user_id,
        )
        claim_facets = [
            facet for facet in plan.requested_facets
            if facet not in {"commitment.open_loop", "boundary.rule", "schedule.event"}
        ]
        target_claims = self._load_claim_nodes_for_entities(
            query.user_id,
            plan.target_entities,
            claim_facets,
            query_text,
            limit=top_k,
        )
        relation_claims = self._load_relation_claims_for_targets(
            query.user_id,
            plan.target_entities,
            query_text,
            limit=top_k,
        )
        active_claims = self._dedupe_nodes(boundary_claims + relation_claims + target_claims)[:top_k]
        schedule_claims = self._load_claim_nodes_for_entities(
            query.user_id,
            plan.target_entities,
            ["schedule.event"],
            query_text,
            limit=3,
        )
        interaction_policy = self.store.get_interaction_policy(query.user_id) if self.store else {}
        relation_state = self.store.get_relation_state(query.user_id) if self.store else None

        if plan.unresolved_references:
            graph_nodes = []
        else:
            graph_nodes = self._retrieve_graph_nodes(query, top_k=max(top_k * 2, 6))
            graph_nodes = self._filter_nodes_for_targets(graph_nodes, plan.target_entities)
        supporting_events = [node for node in graph_nodes if isinstance(node, EpisodeNode)][:top_k]
        supporting_notes = [node for node in graph_nodes if isinstance(node, NoteNode)][:top_k]
        legacy_insights = [node for node in graph_nodes if isinstance(node, InsightNode)][:max(1, top_k // 2)]

        uncertainties = list(plan.unresolved_references)
        if plan.requested_facets and not active_claims and not schedule_claims:
            uncertainties.append("요청과 관련된 active claim이 없어 추정 없이 답해야 함")
        if plan.target_entities and set(plan.target_entities) != {query.user_id} and not graph_nodes:
            uncertainties.append("대상 entity와 직접 연결된 evidence가 없어 관련 보조 기억을 비워둠")

        return ContextBundle(
            plan=plan,
            open_loops=open_loop_nodes,
            active_claims=active_claims,
            relevant_schedule=schedule_claims,
            interaction_policy=interaction_policy,
            relation_state=relation_state,
            supporting_events=supporting_events,
            supporting_notes=supporting_notes,
            legacy_insights=legacy_insights,
            uncertainties=uncertainties,
        )

    def _load_open_loop_nodes(self, user_id: str, query_text: str, limit: int) -> List[ClaimNode]:
        if not self.store:
            return []
        loops = self.store.get_open_loops(user_id, search_text=query_text, limit=limit)
        nodes = []
        for loop in loops:
            nodes.append(
                ClaimNode(
                    node_id=loop.loop_id,
                    subject_id=loop.owner_id,
                    facet="commitment.open_loop",
                    merge_key=f"{loop.owner_id}|commitment.open_loop|{loop.kind}|{loop.text}",
                    value={"kind": loop.kind, "text": loop.text, "priority": loop.priority},
                    qualifiers={},
                    nl_summary=loop.text,
                    source_type="explicit",
                    confidence=1.0,
                    status="active",
                    valid_to=loop.due_at,
                    evidence_episode_ids=loop.evidence_episode_ids,
                )
            )
        return nodes

    def _load_claim_nodes(self, user_id: str, facets: List[str], query_text: str, limit: int,
                          viewer_id: str = None) -> List[ClaimNode]:
        if not self.store:
            return []
        viewer_id = viewer_id or user_id
        claims = self.store.get_active_claims(
            user_id,
            facets=facets,
            search_text=query_text,
            limit=limit,
            viewer_id=viewer_id,
        )
        if claims or not query_text:
            return claims
        return self.store.get_active_claims(
            user_id,
            facets=facets,
            search_text="",
            limit=limit,
            viewer_id=viewer_id,
        )

    def _load_claim_nodes_for_entities(self, viewer_id: str, subject_ids: List[str],
                                       facets: List[str], query_text: str, limit: int) -> List[ClaimNode]:
        if not facets:
            return []
        nodes: List[ClaimNode] = []
        for subject_id in subject_ids:
            nodes.extend(
                self._load_claim_nodes(
                    subject_id,
                    facets,
                    query_text,
                    limit=limit,
                    viewer_id=viewer_id,
                )
            )
        return self._dedupe_nodes(nodes)[:limit]

    def _load_relation_claims_for_targets(self, viewer_id: str, target_entities: List[str],
                                          query_text: str, limit: int) -> List[ClaimNode]:
        if not self.store or not target_entities or target_entities == [viewer_id]:
            return []
        relation_claims = self._load_claim_nodes(
            viewer_id,
            ["relation.to_entity"],
            query_text,
            limit=limit,
            viewer_id=viewer_id,
        )
        target_set = set(target_entities)
        return [
            claim for claim in relation_claims
            if str(claim.value.get("target_entity_id")) in target_set
        ][:limit]

    def _build_known_entities(self, viewer_id: str) -> List[Dict[str, Any]]:
        known_entities: Dict[str, Dict[str, Any]] = {}
        for entity in self.graph.entities.values():
            names = []
            if entity.nickname:
                names.append(entity.nickname)
            names.extend(entity.nickname_history)
            names = list(dict.fromkeys(name for name in names if name))
            entry = known_entities.setdefault(
                entity.user_id,
                {"entity_id": entity.user_id, "names": [], "roles": [], "last_seen": 0.0},
            )
            entry["names"].extend(names)
            entry["names"] = list(dict.fromkeys(name for name in entry["names"] if name))

        if self.store:
            relation_claims = self.store.get_active_claims(
                viewer_id,
                facets=["relation.to_entity"],
                limit=100,
                viewer_id=viewer_id,
            )
            for claim in relation_claims:
                target_entity_id = str(claim.value.get("target_entity_id") or "")
                relation_kind = claim.value.get("relation_kind") or claim.qualifiers.get("relation_kind")
                if not target_entity_id:
                    continue
                entry = known_entities.setdefault(
                    target_entity_id,
                    {"entity_id": target_entity_id, "names": [], "roles": [], "last_seen": 0.0},
                )
                entry["roles"].extend(QueryPlanner.expand_role_aliases(relation_kind))
                entry["last_seen"] = max(entry.get("last_seen", 0.0), float(claim.last_confirmed_at or 0.0))
                entry["roles"] = list(dict.fromkeys(role for role in entry["roles"] if role))

        return list(known_entities.values())

    def _filter_nodes_for_targets(self, nodes: List[BaseNode], target_entities: List[str]) -> List[BaseNode]:
        if not target_entities:
            return nodes

        target_set = set(target_entities)
        target_entity_node_ids = {
            self.graph.get_or_create_user(entity_id, "").node_id
            for entity_id in target_set
        }
        filtered: List[BaseNode] = []
        for node in nodes:
            if isinstance(node, ClaimNode):
                if node.subject_id in target_set:
                    filtered.append(node)
                    continue
                if str(node.value.get("target_entity_id")) in target_set:
                    filtered.append(node)
                    continue
            elif isinstance(node, NoteNode):
                if target_set.intersection(node.related_entity_ids):
                    filtered.append(node)
                    continue
            elif isinstance(node, EpisodeNode):
                if node.user_id in target_set or any(entity_node_id in node.edges for entity_node_id in target_entity_node_ids):
                    filtered.append(node)
                    continue
            elif isinstance(node, InsightNode):
                if any(entity_node_id in node.edges for entity_node_id in target_entity_node_ids):
                    filtered.append(node)
                    continue
        return filtered

    def _dedupe_nodes(self, nodes: List[BaseNode]) -> List[BaseNode]:
        deduped: List[BaseNode] = []
        seen = set()
        for node in nodes:
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            deduped.append(node)
        return deduped

    def _retrieve_graph_nodes(self, query: RetrievalQuery, top_k: int = 5) -> List[BaseNode]:
        """
        STM의 쿼리 객체를 받아 가장 적절한 기억 노드들을 반환
        """
        # [Step 0] Query User의 내부 UUID 식별 (Graph Traversal용)
        # 닉네임은 검색엔 중요하지 않으므로 빈칸
        user_entity = self.graph.get_or_create_user(query.user_id, "") 
        user_node_uuid = user_entity.node_id

        candidates: Dict[str, Dict[str, Any]] = {} 
        # 구조: {node_id: {"node": NodeObj, "score": float, "reason": str}}

        # ---------------------------------------------------------
        # Phase 1: Anchoring (Vector Search & Access Check)
        # ---------------------------------------------------------
        # 전체 노드 스냅샷 가져오기 (임베딩 포함됨)
        all_nodes = self.graph.get_all_nodes()
        
        # 벡터 검색 (Cosine Sim)
        anchors = self._vector_search(all_nodes, query.embedding, top_k=top_k * 3)

        for node, sim_score in anchors:
            # [3-Tier Access Control]
            # 이 기억이 유저와 관련이 있는가?
            if not self._is_accessible(node, query.user_id, user_node_uuid):
                continue
            
            # Insight는 검색 점수 보너스 (지식 우선)
            if isinstance(node, ClaimNode):
                if not self._is_active_claim(node):
                    continue
                weight = getattr(config, "CLAIM_BONUS", 1.35)
            elif isinstance(node, InsightNode):
                weight = config.INSIGHT_BONUS
            elif isinstance(node, NoteNode):
                weight = getattr(config, "NOTE_BASE_WEIGHT", 0.9)
            else:
                weight = config.EPISODE_BASE_WEIGHT
            base_score = sim_score * weight
            
            candidates[node.node_id] = {
                "node": node, 
                "score": base_score,
                "reason": "vector_hit"
            }

        # ---------------------------------------------------------
        # Phase 2: Spreading (Graph Traversal)
        # ---------------------------------------------------------
        # Anchor 노드들에서 엣지를 타고 '숨겨진 맥락'을 가져옵니다.
        
        current_ids = list(candidates.keys())
        
        for anchor_id in current_ids:
            anchor_data = candidates[anchor_id]
            anchor_node = anchor_data["node"]
            anchor_score = anchor_data["score"]
            
            # 모든 이웃(Edge) 확인
            for target_id, edge_weight in anchor_node.edges.items():
                if target_id in candidates: continue # 이미 후보면 패스
                
                target_node = self.graph.get_node(target_id)
                if not target_node: continue

                # 확산 점수 계산
                spread_score = anchor_score * edge_weight * config.SPREAD_DECAY_FACTOR
                reason = "graph_connection"

                # [Logic] 노드 타입별 확산 전략
                if isinstance(anchor_node, (InsightNode, ClaimNode, NoteNode)) and isinstance(target_node, EpisodeNode):
                    # Insight(성향) -> Episode(증거): 매우 강력한 연결
                    spread_score *= config.INSIGHT_TO_EPISODE_BOOST
                    reason = "evidence_of_state"
                    
                elif isinstance(anchor_node, EpisodeNode) and isinstance(target_node, EpisodeNode):
                    # Episode -> Episode: 전후 상황
                    spread_score *= config.EPISODE_TO_EPISODE_DECAY
                    reason = "temporal_context"

                # [3-Tier Entity Check]
                # 만약 Insight가 '나(Entity)'에 대한 것이라면 가중치 부여
                if isinstance(target_node, EntityNode) and target_node.node_id == user_node_uuid:
                    # Entity 자체를 리턴하기보단, 연결된 다른 Insight를 찾는 징검다리로 쓰임
                    # 여기서는 EntityNode 자체는 검색 결과로 잘 안 쓰이므로 패스하거나 점수 낮춤
                    continue 

                candidates[target_id] = {
                    "node": target_node,
                    "score": spread_score,
                    "reason": reason
                }

        # ---------------------------------------------------------
        # Phase 3: Reranking
        # ---------------------------------------------------------
        final_results = []
        
        for nid, data in candidates.items():
            node = data["node"]
            final_score = data["score"]
            
            # EntityNode는 검색 결과에서 제외 (보통 대화 맥락엔 필요 없음)
            if isinstance(node, EntityNode):
                continue
            if isinstance(node, ClaimNode) and not self._is_active_claim(node):
                continue

            # 1. [Mood Congruence] 기분 일치성
            if isinstance(node, EpisodeNode):
                # 봇의 현재 기분과 기억의 감정 태그가 비슷하면(텍스트 매칭) 가산점
                # (더 정교하게 하려면 감정 벡터 유사도를 써야 함)
                if query.current_mood in node.emotion_tag: 
                    final_score *= config.MOOD_CONGRUENCE_BOOST
            
            # 2. [Recency] 시간 감쇠 (Episode만)
            if isinstance(node, EpisodeNode):
                hours_passed = (time.time() - node.timestamp) / 3600
                # config.RECENCY_DECAY_RATE (예: 0.05)
                decay = 1.0 / (1.0 + (hours_passed / 24.0 * config.RECENCY_DECAY_RATE))
                final_score *= decay

            # 3. [Keyword Boost]
            node_text = getattr(node, 'content', getattr(node, 'nl_summary', getattr(node, 'summary', "")))
            for kw in query.keywords:
                if kw in node_text:
                    final_score *= config.KEYWORD_MATCH_BOOST
                    break
            
            final_results.append((final_score, node))

        # 정렬 및 Top-K 반환
        final_results.sort(key=lambda x: x[0], reverse=True)
        
        return [node for score, node in final_results[:top_k]]

    def _is_active_claim(self, claim: ClaimNode) -> bool:
        if claim.status != "active":
            return False
        if isinstance(claim.valid_to, (int, float)) and claim.valid_to < time.time():
            return False
        return True

    def _is_accessible(self, node: BaseNode, user_id_str: str, user_node_uuid: str) -> bool:
        """
        [Access Control Policy]
        유저가 이 기억(Node)에 접근할 권한이 있는가?
        
        1. Ownership: 내가 주 화자(Speaker)인가? (user_id 문자열 일치)
        2. Connection: 내가 태그된(Edge로 연결된) 기억인가? (EntityNode UUID 연결)
        """
        # 1. EntityNode는 본인 것이면 OK
        if isinstance(node, EntityNode):
            return node.user_id == user_id_str

        # 2. EpisodeNode / InsightNode
        if isinstance(node, ClaimNode):
            if node.subject_id == user_id_str:
                return True
            scope = node.scope or "user_private"
            if scope != "participants":
                return False
            audience_ids = {
                str(audience_id)
                for audience_id in (node.qualifiers.get("audience_ids") or [])
                if audience_id
            }
            return user_id_str in audience_ids and user_node_uuid in node.edges

        if isinstance(node, NoteNode):
            return user_id_str in node.related_entity_ids

        if hasattr(node, 'user_id') and node.user_id == user_id_str:
            return True # 내가 만든 기억 (Fast Path)
            
        # 3. [Graph Path] 내 EntityNode와 연결되어 있는가? (Shared Memory)
        if user_node_uuid in node.edges:
            return True # 내가 태그된 기억
            
        return False

    def _vector_search(self, nodes: List[BaseNode], query_vec: List[float], top_k: int) -> List[Tuple[BaseNode, float]]:
        """
        In-Memory Vector Search (Cosine Similarity)
        """
        if not nodes or not query_vec: 
            return []
        
        scores = []
        q_vec = np.array(query_vec)
        q_norm = np.linalg.norm(q_vec)
        
        if q_norm == 0:
            return []
        
        for node in nodes:
            # 임베딩이 없으면 패스 (EntityNode 등)
            if not node.embedding: 
                continue
            
            n_vec = np.array(node.embedding)
            n_norm = np.linalg.norm(n_vec)
            
            if n_norm == 0: 
                sim = 0.0
            else: 
                sim = float(np.dot(q_vec, n_vec) / (q_norm * n_norm))
            
            scores.append((node, sim))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
