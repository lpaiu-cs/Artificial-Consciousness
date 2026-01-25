import json
import os
import threading
import time
from typing import Dict, Union, Optional
from memory_structures import EpisodeNode, InsightNode
import config

class MemoryGraph:
    """
    장기 기억을 저장하는 그래프 데이터베이스 (JSON 기반)
    
    Thread Safety:
    - threading.Lock을 사용하여 동시 접근 제어
    - 읽기/쓰기 모두 잠금 사용 (RLock으로 재진입 허용)
    
    Storage Separation:
    - 그래프 구조 (노드 메타데이터 + 엣지): ltm_graph.json
    - 임베딩 벡터: ltm_embeddings.json
    """
    
    def __init__(self, graph_path: str = None, embeddings_path: str = None):
        self.graph_path = graph_path or config.LTM_GRAPH_PATH
        self.embeddings_path = embeddings_path or config.LTM_EMBEDDINGS_PATH
        
        # Thread Safety: RLock은 같은 스레드에서 여러 번 획득 가능
        self._lock = threading.RLock()
        
        # 그래프 구조 (메모리)
        self.episodes: Dict[str, EpisodeNode] = {}
        self.insights: Dict[str, InsightNode] = {}
        
        # 임베딩 벡터 캐시 (별도 저장)
        self._embeddings_cache: Dict[str, list] = {}
        
        self._load_all()

    # === Thread-Safe Public API ===
    
    def add_episode(self, content: str, user_id: str, emotion: str, 
                    embedding: Optional[list] = None) -> EpisodeNode:
        """에피소드 노드 추가 (Thread-Safe)"""
        with self._lock:
            node = EpisodeNode(
                content=content, 
                timestamp=time.time(),  # 실제 시간 사용
                emotion_tag=emotion, 
                user_id=user_id,
                embedding=None  # 임베딩은 별도 저장
            )
            
            # 최근 에피소드와 연결 (Temporal Edge)
            if self.episodes:
                last_id = list(self.episodes.keys())[-1]
                self._connect_nodes_unsafe(
                    last_id, node.node_id, 
                    weight=config.TEMPORAL_EDGE_FORWARD
                )
                self._connect_nodes_unsafe(
                    node.node_id, last_id, 
                    weight=config.TEMPORAL_EDGE_BACKWARD
                )

            self.episodes[node.node_id] = node
            
            # 임베딩 별도 저장
            if embedding:
                self._embeddings_cache[node.node_id] = embedding
                
            return node

    def add_or_update_insight(self, summary: str, subject: str, predicate: str, 
                              object_: str, embedding: Optional[list] = None) -> InsightNode:
        """통찰 노드 추가/업데이트 (Thread-Safe)"""
        with self._lock:
            # TODO: 중복 검사 로직 (Vector Sim) 추가 권장
            node = InsightNode(
                summary=summary, 
                subject=subject, 
                predicate=predicate, 
                object=object_,
                last_updated=time.time(),
                embedding=None  # 임베딩은 별도 저장
            )
            self.insights[node.node_id] = node
            
            # 임베딩 별도 저장
            if embedding:
                self._embeddings_cache[node.node_id] = embedding
                
            return node

    def connect_nodes(self, source_id: str, target_id: str, weight: float = 1.0):
        """노드 간 엣지 연결 (Thread-Safe)"""
        with self._lock:
            self._connect_nodes_unsafe(source_id, target_id, weight)

    def get_node(self, node_id: str) -> Union[EpisodeNode, InsightNode, None]:
        """노드 조회 (Thread-Safe)"""
        with self._lock:
            return self._get_node_unsafe(node_id)
    
    def get_embedding(self, node_id: str) -> Optional[list]:
        """노드의 임베딩 벡터 조회 (Thread-Safe)"""
        with self._lock:
            return self._embeddings_cache.get(node_id)
    
    def get_all_nodes(self) -> list:
        """모든 노드 조회 (Thread-Safe) - 검색용 스냅샷 반환"""
        with self._lock:
            nodes = []
            for node in list(self.episodes.values()) + list(self.insights.values()):
                # 임베딩을 노드 객체에 임시 할당 (검색 호환성)
                node_copy = node
                node_copy.embedding = self._embeddings_cache.get(node.node_id)
                nodes.append(node_copy)
            return nodes

    def save_all(self):
        """그래프와 임베딩을 파일에 저장 (Thread-Safe)"""
        with self._lock:
            self._save_graph()
            self._save_embeddings()
            print(f"✅ LTM 저장 완료: {len(self.episodes)} Episodes, {len(self.insights)} Insights")

    # === Legacy API (하위 호환) ===
    
    def save_to_json(self):
        """하위 호환용 - save_all() 호출"""
        self.save_all()
    
    def _get_node(self, node_id: str) -> Union[EpisodeNode, InsightNode, None]:
        """Legacy: Lock 없는 내부용 - 외부에서는 get_node() 사용"""
        return self._get_node_unsafe(node_id)

    # === Private Methods (Lock 없음, 내부 전용) ===
    
    def _get_node_unsafe(self, node_id: str) -> Union[EpisodeNode, InsightNode, None]:
        return self.episodes.get(node_id) or self.insights.get(node_id)
    
    def _connect_nodes_unsafe(self, source_id: str, target_id: str, weight: float):
        source = self._get_node_unsafe(source_id)
        if source:
            source.edges[target_id] = weight

    def _save_graph(self):
        """그래프 구조만 저장 (임베딩 제외)"""
        data = {
            "episodes": {},
            "insights": {}
        }
        
        for k, v in self.episodes.items():
            node_dict = v.to_dict()
            node_dict.pop("embedding", None)  # 임베딩 제거
            data["episodes"][k] = node_dict
            
        for k, v in self.insights.items():
            node_dict = v.to_dict()
            node_dict.pop("embedding", None)  # 임베딩 제거
            data["insights"][k] = node_dict
        
        try:
            with open(self.graph_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"❌ Graph Save Error: {e}")

    def _save_embeddings(self):
        """임베딩 벡터만 별도 저장"""
        try:
            with open(self.embeddings_path, "w", encoding="utf-8") as f:
                json.dump(self._embeddings_cache, f)
        except IOError as e:
            print(f"❌ Embeddings Save Error: {e}")

    def _load_all(self):
        """그래프와 임베딩 모두 로드"""
        self._load_graph()
        self._load_embeddings()

    def _load_graph(self):
        """그래프 구조 로드"""
        if not os.path.exists(self.graph_path):
            return
        try:
            with open(self.graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Episode 복원
            for node_id, node_data in data.get("episodes", {}).items():
                self.episodes[node_id] = EpisodeNode(
                    node_id=node_data.get("node_id", node_id),
                    content=node_data.get("content", ""),
                    timestamp=node_data.get("timestamp", 0.0),
                    emotion_tag=node_data.get("emotion_tag", "neutral"),
                    user_id=node_data.get("user_id", "unknown"),
                    edges=node_data.get("edges", {}),
                    embedding=None
                )
            
            # Insight 복원
            for node_id, node_data in data.get("insights", {}).items():
                self.insights[node_id] = InsightNode(
                    node_id=node_data.get("node_id", node_id),
                    summary=node_data.get("summary", ""),
                    subject=node_data.get("subject", ""),
                    predicate=node_data.get("predicate", ""),
                    object=node_data.get("object", ""),
                    confidence=node_data.get("confidence", 0.5),
                    last_updated=node_data.get("last_updated", 0.0),
                    edges=node_data.get("edges", {}),
                    embedding=None
                )
                
            print(f"📂 Graph 로드 완료: {len(self.episodes)} Episodes, {len(self.insights)} Insights")
            
        except json.JSONDecodeError:
            print("❌ Graph Load Error: Invalid JSON")
        except Exception as e:
            print(f"❌ Graph Load Error: {e}")

    def _load_embeddings(self):
        """임베딩 벡터 로드"""
        if not os.path.exists(self.embeddings_path):
            return
        try:
            with open(self.embeddings_path, "r", encoding="utf-8") as f:
                self._embeddings_cache = json.load(f)
            print(f"📂 Embeddings 로드 완료: {len(self._embeddings_cache)} vectors")
        except json.JSONDecodeError:
            print("❌ Embeddings Load Error: Invalid JSON")
        except Exception as e:
            print(f"❌ Embeddings Load Error: {e}")