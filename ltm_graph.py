# ltm_graph.py
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union

# --- Node Definitions ---

@dataclass
class BaseNode:
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[List[float]] = field(default=None, repr=False)
    edges: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

@dataclass
class EpisodeNode(BaseNode):
    content: str = ""
    timestamp: float = 0.0
    emotion_tag: str = "neutral"
    user_id: str = "unknown"
    type: str = "episode" # 복원 시 구분용

@dataclass
class InsightNode(BaseNode):
    summary: str = ""
    subject: str = ""
    predicate: str = ""
    object: str = ""
    confidence: float = 0.5
    last_updated: float = 0.0
    type: str = "insight" # 복원 시 구분용

# --- Graph Manager with Persistence ---

class MemoryGraph:
    def __init__(self, db_path="memory_graph.json"):
        self.db_path = db_path
        self.episodes: Dict[str, EpisodeNode] = {}
        self.insights: Dict[str, InsightNode] = {}
        
        # 시작 시 파일이 있으면 로드
        self.load_from_json()

    def get_node(self, node_id: str) -> Union[EpisodeNode, InsightNode, None]:
        if node_id in self.episodes: return self.episodes[node_id]
        if node_id in self.insights: return self.insights[node_id]
        return None

    def add_episode(self, content, user_id, emotion, embedding=None) -> EpisodeNode:
        """STM -> Eviction -> Reflection을 거쳐 들어온 에피소드 저장"""
        node = EpisodeNode(
            content=content,
            timestamp=time.time(),
            emotion_tag=emotion,
            user_id=user_id,
            embedding=embedding
        )
        
        # [Temporal Edge] 시간적 연결: 바로 직전 에피소드와 연결
        # (유저별로 마지막 에피소드를 추적하는 로직이 있으면 더 좋음)
        if self.episodes:
            # 전체 중 가장 최근 것과 연결 (간단 구현)
            # 실제로는 user_id가 같은 마지막 노드를 찾아야 함
            last_id = list(self.episodes.keys())[-1]
            last_node = self.episodes[last_id]
            
            if last_node.user_id == user_id:
                # 과거 -> 현재 (강함)
                last_node.edges[node.node_id] = 1.0
                # 현재 -> 과거 (약함)
                node.edges[last_id] = 0.5

        self.episodes[node.node_id] = node
        return node

    def add_or_update_insight(self, summary, subject, predicate, object_, embedding=None) -> InsightNode:
        """통찰 저장 (중복 시 강화)"""
        # 1. 기존에 비슷한 Insight가 있는지 확인 (Vector Search 필요)
        # 여기서는 Mockup으로 '항상 새로운 것'으로 가정
        existing_id = None 
        
        if existing_id:
            node = self.insights[existing_id]
            node.confidence = min(1.0, node.confidence + 0.1)
            node.last_updated = time.time()
            return node
        else:
            node = InsightNode(
                summary=summary,
                subject=subject,
                predicate=predicate,
                object=object_,
                embedding=embedding
            )
            self.insights[node.node_id] = node
            return node

    def connect_nodes(self, source_id, target_id, weight=1.0):
        """두 노드 연결 (양방향 아님, 방향성 그래프)"""
        source = self.get_node(source_id)
        if source:
            source.edges[target_id] = weight


    def save_to_json(self):
        """현재 그래프 상태를 JSON 파일로 저장"""
        data = {
            "episodes": {k: v.to_dict() for k, v in self.episodes.items()},
            "insights": {k: v.to_dict() for k, v in self.insights.items()}
        }
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # print(f"💾 MemoryGraph saved to {self.db_path}")
        except Exception as e:
            print(f"❌ Save Error: {e}")

    def load_from_json(self):
        """JSON 파일에서 그래프 상태 복원"""
        if not os.path.exists(self.db_path):
            return

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Episodes 복원
            for k, v in data.get("episodes", {}).items():
                node = EpisodeNode(**v)
                self.episodes[k] = node

            # Insights 복원
            for k, v in data.get("insights", {}).items():
                node = InsightNode(**v)
                self.insights[k] = node
                
            print(f"📂 MemoryGraph loaded: {len(self.episodes)} episodes, {len(self.insights)} insights.")
            
        except Exception as e:
            print(f"❌ Load Error: {e}")