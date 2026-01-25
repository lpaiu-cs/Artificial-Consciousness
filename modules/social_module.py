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

    def get_relationship_desc(self, user_id: str) -> str:
        score = self.get_affinity(user_id)
        if score >= 80: return "Best Friend (Playful)"
        if score >= 60: return "Friend (Friendly)"
        if score <= 20: return "Hostile (Cold)"
        return "Acquaintance (Polite)"
    
    def get_affinity(self, user_id: str) -> float:
        return self.affinity_map.get(user_id, 50.0) # 기본값 50 (중립)

    def update_affinity(self, user_id: str, score_delta: float):
        current = self.get_affinity(user_id)
        self.affinity_map[user_id] = max(0.0, min(100.0, current + score_delta))
        self._save_db()

    def _save_db(self):
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.affinity_map, f, indent=2)
        except Exception as e:
            print(f"❌ Social Map Save Error: {e}")

    def _load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.affinity_map = json.load(f)
            except: 
                self.affinity_map = {}