import json
import os
from typing import Dict, Union
from memory_structures import EpisodeNode, InsightNode
import config

class MemoryGraph:
    """장기 기억을 저장하는 그래프 데이터베이스 (JSON 기반)"""
    
    def __init__(self, db_path=config.LTM_SAVE_PATH):
        self.db_path = db_path
        self.episodes: Dict[str, EpisodeNode] = {}
        self.insights: Dict[str, InsightNode] = {}
        self.load_from_json()

    def add_episode(self, content, user_id, emotion, embedding=None) -> EpisodeNode:
        node = EpisodeNode(
            content=content, timestamp=0.0, emotion_tag=emotion, 
            user_id=user_id, embedding=embedding
        ) # timestamp 등은 실제 구현시 time.time() 사용
        
        # 최근 에피소드와 연결 (Temporal Edge)
        if self.episodes:
            last_id = list(self.episodes.keys())[-1]
            self.connect_nodes(last_id, node.node_id, weight=0.5)
            self.connect_nodes(node.node_id, last_id, weight=1.0) # 역방향 강하게

        self.episodes[node.node_id] = node
        return node

    def add_or_update_insight(self, summary, subject, predicate, object_, embedding=None) -> InsightNode:
        # TODO: 중복 검사 로직 (Vector Sim) 추가 권장
        node = InsightNode(
            summary=summary, subject=subject, predicate=predicate, object=object_, embedding=embedding
        )
        self.insights[node.node_id] = node
        return node

    def connect_nodes(self, source_id, target_id, weight=1.0):
        source = self._get_node(source_id)
        if source:
            source.edges[target_id] = weight

    def _get_node(self, node_id) -> Union[EpisodeNode, InsightNode, None]:
        return self.episodes.get(node_id) or self.insights.get(node_id)

    def save_to_json(self):
        data = {
            "episodes": {k: v.to_dict() for k, v in self.episodes.items()},
            "insights": {k: v.to_dict() for k, v in self.insights.items()}
        }
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"❌ LTM Save Error: {e}")

    def load_from_json(self):
        if not os.path.exists(self.db_path): return
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 데이터 복원 로직... (이전 코드와 동일하므로 생략)
                # 여기서 EpisodeNode, InsightNode 객체로 다시 매핑해야 함
        except json.JSONDecodeError:
            print("❌ LTM Load Error: Invalid JSON")