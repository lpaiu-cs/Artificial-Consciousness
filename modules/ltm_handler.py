import time
from typing import List, Dict, Any
from memory_structures import RetrievalQuery
from modules.ltm_graph import MemoryGraph, EpisodeNode, InsightNode, BaseNode
import numpy as np
import config

# 실제 구현 시에는 Vector DB 클라이언트 (Pinecone, Chroma 등) 필요
# 여기서는 MockVectorDB로 대체합니다.

class LongTermMemory:
    """
    [System 2: LTM Retrieval]
    단순 유사도 검색이 아니라, '그래프 탐색'을 통해 맥락을 풍부하게 가져옵니다.
    Query -> [Vector Search] -> Anchors -> [Graph Traversal] -> Candidates -> [Reranking] -> Results
    """
    def __init__(self, graph_db: MemoryGraph, api_client):
        self.graph = graph_db
        self.api = api_client
        # 실제로는 여기서 Pinecone/Chroma 클라이언트 초기화
        # self.vector_db = PineconeClient(...)

    def retrieve(self, query: RetrievalQuery, top_k: int = 5) -> List[BaseNode]:
        """
        STM의 쿼리 객체를 받아 가장 적절한 기억 노드들을 반환
        """
        candidates: Dict[str, Dict[str, Any]] = {} 
        # 구조: {node_id: {"node": NodeObj, "score": float, "source": str}}

        # ---------------------------------------------------------
        # Phase 1: Anchoring (Vector Search)
        # ---------------------------------------------------------
        # 질문의 의도(Embedding)와 가장 유사한 '통찰(Insight)'과 '사건(Episode)'을 찾습니다.
        # Insight에 가중치를 두어 검색합니다. (지식 우선 검색)
        
        # [Thread-Safe] 그래프에서 스냅샷 가져오기
        all_nodes = self.graph.get_all_nodes()
        anchors = self._mock_vector_search(all_nodes, query.embedding, top_k=top_k * 2)

        for node, sim_score in anchors:
            # [Entity Filter] 타인의 사적인 기억은 제외 (User ID 매칭)
            if isinstance(node, EpisodeNode) and node.user_id != query.user_id:
                continue
            
            # Insight는 검색 점수 보너스 (Fact 우선) - config에서 가중치 사용
            insight_weight = config.INSIGHT_BONUS if isinstance(node, InsightNode) else config.EPISODE_BASE_WEIGHT
            base_score = sim_score * insight_weight
            
            candidates[node.node_id] = {
                "node": node, 
                "score": base_score,
                "reason": "vector_hit"
            }

        # ---------------------------------------------------------
        # Phase 2: Spreading (Graph Traversal)
        # ---------------------------------------------------------
        # 찾아낸 Anchor 노드들에서 엣지를 타고 '숨겨진 맥락'을 가져옵니다.
        
        # 딕셔너리 크기가 변하므로 리스트로 복사해서 순회
        current_candidate_ids = list(candidates.keys())
        
        for anchor_id in current_candidate_ids:
            anchor_data = candidates[anchor_id]
            anchor_node = anchor_data["node"]
            anchor_score = anchor_data["score"]
            
            # Anchor의 모든 이웃(Edge)을 확인
            for target_id, edge_weight in anchor_node.edges.items():
                if target_id in candidates:
                    continue # 이미 후보에 있으면 패스
                
                target_node = self.graph.get_node(target_id)
                if not target_node: continue

                # 확산 점수 계산: (부모 점수) * (연결 강도) * (감쇠 계수)
                spread_score = anchor_score * edge_weight * config.SPREAD_DECAY_FACTOR
                
                # [Logic] 노드 타입별 확산 전략 - config에서 가중치 사용
                if isinstance(anchor_node, InsightNode) and isinstance(target_node, EpisodeNode):
                    # Insight -> Episode: "이 생각의 근거가 되는 사건" (중요)
                    spread_score *= config.INSIGHT_TO_EPISODE_BOOST
                    reason = f"evidence_of_insight({anchor_node.summary[:10]}..)"
                    
                elif isinstance(anchor_node, EpisodeNode) and isinstance(target_node, EpisodeNode):
                    # Episode -> Episode: "그때 있었던 앞뒤 상황" (시간적/의미적 맥락)
                    spread_score *= config.EPISODE_TO_EPISODE_DECAY
                    reason = "temporal_context"
                else:
                    reason = "graph_connection"

                # 확산된 노드 후보 등록
                candidates[target_id] = {
                    "node": target_node,
                    "score": spread_score,
                    "reason": reason
                }

        # ---------------------------------------------------------
        # Phase 3: Reranking & Contextual Adjustment
        # ---------------------------------------------------------
        final_results = []
        
        for nid, data in candidates.items():
            node = data["node"]

            # 1. [Similarity] 벡터 유사도로 시작
            final_score = data["score"]
            
            # 2. [Importance/Mood] 기분 일치성 - config에서 가중치 사용
            # 현재 봇의 기분과 기억의 감정이 일치하면 가산점
            if isinstance(node, EpisodeNode):
                if node.emotion_tag == query.current_mood:
                    final_score *= config.MOOD_CONGRUENCE_BOOST
            
            # 3. [Recency] Time Decay - 오래된 기억은 감가상각 - config에서 감쇠율 사용
            if isinstance(node, EpisodeNode):
                hours_passed = (time.time() - node.timestamp) / 3600
                decay = 1.0 / (1.0 + (hours_passed / 24.0 * config.RECENCY_DECAY_RATE))
                final_score *= decay

            # 4. [Keyword Matching] Hybrid Search - config에서 부스트 사용
            # 임베딩이 놓친 고유명사(예: "민초")가 텍스트에 있으면 점수 팍 올림
            node_text = node.content if isinstance(node, EpisodeNode) else node.summary
            for kw in query.keywords:
                if kw in node_text:
                    final_score *= config.KEYWORD_MATCH_BOOST
                    break
            
            # 5. [Frequency]는 그래프 구조 자체에서 해결됨
            final_results.append((final_score, node))

        # 점수순 정렬
        final_results.sort(key=lambda x: x[0], reverse=True)
        
        # 디버깅용 로그 (나중에 삭제)
        # print(f"🔍 Retrieval Top 3:")
        # for score, node in final_results[:3]:
        #     content = node.content if isinstance(node, EpisodeNode) else node.summary
        #     print(f" - [{score:.2f}] ({type(node).__name__}) {content}")

        return [node for score, node in final_results[:top_k]]

    def _vector_search(self, nodes, query_vec, top_k):
        """
        (Mockup) 실제 Vector DB 대신 코사인 유사도 계산
        """
        if not nodes or not query_vec: return []
        
        scores = []
        q_vec = np.array(query_vec)
        q_norm = np.linalg.norm(q_vec)
        
        for node in nodes:
            if not node.embedding: continue
            
            n_vec = np.array(node.embedding)
            n_norm = np.linalg.norm(n_vec)
            
            if n_norm == 0 or q_norm == 0: sim = 0
            else: sim = np.dot(q_vec, n_vec) / (q_norm * n_norm)
            
            scores.append((node, sim))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]