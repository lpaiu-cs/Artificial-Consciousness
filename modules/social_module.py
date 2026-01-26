import json
import os
from typing import Dict, List, Optional
import config

class SocialMap:
    """
    [Social Brain]
    유저 정체성(Identity), 호감도(Affinity), 관계(Relationship) 관리
    """
    def __init__(self):
        self.db_path = config.SOCIAL_SAVE_PATH
        # 구조: { "user_id": { "affinity": 50.0, "name": "Nick", "history": [] } }
        self.social_data: Dict[str, Dict] = {} 
        self._load_db()

    def update_identity(self, user_id: str, current_nickname: str) -> bool:
        """
        [Identity Check]
        닉네임이 변경되었는지 확인하고 업데이트합니다.
        변경되었다면 True를 반환하여 '개명 이벤트'임을 알립니다.
        """
        uid = str(user_id)
        if uid not in self.social_data:
            # 신규 유저 등록
            self.social_data[uid] = {
                "affinity": 50.0,
                "name": current_nickname,
                "history": []
            }
            self._save_db()
            return False # 신규 유저는 변경 아님

        user_info = self.social_data[uid]
        old_name = user_info.get("name", "Unknown")

        # 닉네임이 다르면 업데이트 (개명)
        if old_name != current_nickname:
            user_info["history"].append(old_name) # 역사에 기록
            user_info["name"] = current_nickname  # 현재 이름 갱신
            self._save_db()
            return True # 변경됨!

        return False
        
    def get_nickname(self, user_id: str) -> str:
        """user_id에 매핑된 현재 닉네임 반환"""
        return self.social_data.get(str(user_id), {}).get("name", "Unknown")

    def get_affinity(self, user_id: str) -> float:
        return self.social_data.get(str(user_id), {}).get("affinity", 50.0)
    
    def get_relationship_desc(self, user_id: str) -> str:
        score = self.get_affinity(user_id)
        if score >= 80: return "Best Friend (Playful)"
        if score >= 60: return "Friend (Friendly)"
        if score <= 20: return "Hostile (Cold)"
        return "Acquaintance (Polite)"

    def update_affinity(self, user_id: str, score_delta: float):
        current = self.get_affinity(user_id)
        self.social_data[user_id]["affinity"] = max(0.0, min(100.0, current + score_delta))
        self._save_db()

    def _save_db(self):
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.social_data, f, indent=2)
        except Exception as e:
            print(f"❌ Social Map Save Error: {e}")

    def _load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.social_data = json.load(f)
            except: 
                self.social_data = {}