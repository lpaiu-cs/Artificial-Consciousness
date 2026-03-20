import time
import numpy as np
from typing import List, Dict, Any, Tuple
from memory_structures import ClaimNode, RetrievalQuery, EpisodeNode, InsightNode, NoteNode, EntityNode, BaseNode
from modules.ltm_graph import MemoryGraph
import config

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
    def __init__(self, graph_db: MemoryGraph, api_client):
        self.graph = graph_db
        self.api = api_client # (Vector DB 도입 시 사용)

    def retrieve(self, query: RetrievalQuery, top_k: int = 5) -> List[BaseNode]:
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
        if isinstance(node, ClaimNode) and node.subject_id == user_id_str:
            return True
        if isinstance(node, NoteNode) and user_id_str in node.related_entity_ids:
            return True
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
