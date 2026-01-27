import json
import os
import threading
import time
from typing import Dict, Union, Optional, List, Any
from memory_structures import EpisodeNode, InsightNode, EntityNode
import config

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
        self.entities: Dict[str, EntityNode] = {}
        self._embeddings_cache: Dict[str, list] = {}
        
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
            node = InsightNode(
                summary=summary, 
                confidence=confidence,
                last_updated=time.time(),
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

    def get_node(self, node_id: str) -> Union[EpisodeNode, InsightNode, EntityNode, None]:
        with self._lock:
            return self.episodes.get(node_id) or \
                   self.insights.get(node_id) or \
                   self.entities.get(node_id)
    
    def get_all_nodes(self) -> List[Any]:
        with self._lock:
            all_nodes = []
            for store in [self.episodes, self.insights, self.entities]:
                for node in store.values():
                    # 임베딩 임시 주입 (검색용)
                    node.embedding = self._embeddings_cache.get(node.node_id)
                    all_nodes.append(node)
            return all_nodes

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

    def _apply_log_entry(self, entry: Dict):
        """로그 한 줄을 메모리에 반영"""
        action = entry.get("action")
        payload = entry.get("payload", {})
        
        if action == "UPSERT_NODE":
            cat = payload.get("category")
            data = payload.get("data")
            node_id = data.get("node_id")
            
            # 객체 복원
            if cat == "episodes":
                node = EpisodeNode(**data)
                self.episodes[node_id] = node
            elif cat == "insights":
                node = InsightNode(**data)
                self.insights[node_id] = node
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