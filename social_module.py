import json
import os
from typing import Dict

class SocialMap:
    """
    [Social Brain]
    유저별 호감도(Affinity)와 관계(Relationship)를 관리합니다.
    """
    def __init__(self, db_path="social_map.json"):
        self.db_path = db_path
        self.affinity_map: Dict[str, float] = self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except: pass
        return {}

    def get_affinity(self, user_id: str) -> float:
        return self.affinity_map.get(user_id, 50.0) # 기본값 50 (중립)

    def update_affinity(self, user_id: str, score_delta: float):
        current = self.get_affinity(user_id)
        self.affinity_map[user_id] = max(0.0, min(100.0, current + score_delta))
        self._save_db()

    def _save_db(self):
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.affinity_map, f, indent=2)

    def get_relationship_desc(self, user_id: str) -> str:
        score = self.get_affinity(user_id)
        if score >= 80: return "절친 (매우 편하고 장난스러운 태도)"
        if score >= 60: return "친구 (호의적인 태도)"
        if score <= 20: return "적대적 (냉소적이고 사무적인 태도)"
        return "지인 (예의 바른 태도)"