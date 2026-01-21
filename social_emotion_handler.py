import re
import json
import os

class BotEmotionEngine:
    """
    [Feature 1: Bot's Heart]
    봇의 내부 감정 상태 벡터를 관리합니다.
    감정은 외부 자극(대화)에 의해 고양되고, 시간이 지나면 자연스럽게 소멸(Decay)됩니다.
    """
    def __init__(self):
        # 감정 벡터 (0.0 ~ 10.0)
        self.current_mood = {
            "joy": 0.0,      # 기쁨/신남
            "anger": 0.0,    # 분노/짜증
            "sadness": 0.0,  # 슬픔/우울
            "trust": 5.0     # 신뢰/평온 (기본값)
        }
        # 감정 민감도 (성격에 따라 조절 가능)
        self.sensitivity = 1.5 
        
        # 감정 감지 패턴 (Regex)
        self.patterns = {
            "praise": re.compile(r"(고마워|천재|똑똑해|잘했어|최고|사랑해|좋아)"),
            "insult": re.compile(r"(바보|멍청|꺼져|시끄러|닥쳐|노잼|별로|짜증)"),
            "sad_user": re.compile(r"(우울해|슬퍼|힘들어|망했어|죽고싶|ㅠㅠ|ㅠ_ㅠ)"),
            "funny": re.compile(r"(ㅋㅋㅋ|ㅎㅎㅎ|웃기네|재밌어)")
        }

    def update_mood(self, user_text):
        """유저의 발화에 따라 봇의 감정 벡터를 즉시 수정"""
        # 1. 칭찬을 들었을 때 (Pride/Joy)
        if self.patterns["praise"].search(user_text):
            self.current_mood["joy"] += 2.0 * self.sensitivity
            self.current_mood["trust"] += 1.0
            self.current_mood["anger"] -= 1.0 # 화 풀림

        # 2. 욕을 들었을 때 (Distress/Anger)
        if self.patterns["insult"].search(user_text):
            self.current_mood["anger"] += 2.5 * self.sensitivity
            self.current_mood["joy"] -= 2.0
            self.current_mood["trust"] -= 2.0

        # 3. 유저가 슬퍼할 때 (Empathy -> Sadness)
        if self.patterns["sad_user"].search(user_text):
            self.current_mood["sadness"] += 1.5 * self.sensitivity
            self.current_mood["joy"] -= 1.0

        # 4. 웃긴 상황 (Joy)
        if self.patterns["funny"].search(user_text):
            self.current_mood["joy"] += 1.0

        # 값 클램핑 (0~10 범위 유지)
        for k in self.current_mood:
            self.current_mood[k] = max(0.0, min(10.0, self.current_mood[k]))

    def decay_mood(self):
        """시간이 지남에 따라 감정이 식는(Decay) 과정 (매 턴 호출)"""
        for k in ["joy", "anger", "sadness"]:
            self.current_mood[k] *= 0.8  # 매 턴 20%씩 감소 (평정심을 되찾음)
        
        # 신뢰(Trust)는 5.0(평온)으로 서서히 복귀
        if self.current_mood["trust"] > 5.0: self.current_mood["trust"] -= 0.1
        elif self.current_mood["trust"] < 5.0: self.current_mood["trust"] += 0.1

    def get_mood_description(self):
        """LLM 프롬프트에 주입할 현재 기분 묘사"""
        # 가장 지배적인 감정 찾기
        dominant = max(self.current_mood, key=self.current_mood.get)
        score = self.current_mood[dominant]

        if score < 3.0: return "평온하고 차분한 상태"
        
        descriptions = {
            "joy": f"매우 신나고 기분이 좋은 상태 (강도: {score:.1f})",
            "anger": f"상당히 짜증나고 화가 난 상태 (강도: {score:.1f}, 삐딱하게 반응해라)",
            "sadness": f"약간 우울하고 감성적인 상태 (강도: {score:.1f})",
            "trust": "상대방을 깊이 신뢰하고 너그러운 상태"
        }
        return descriptions.get(dominant, "평온함")


class SocialMemoryEngine:
    """
    [Feature 2: Social Map]
    유저별 호감도(Affinity)를 관리합니다. (0~100)
    호감도에 따라 봇의 태도(말투, 허용 범위)가 달라집니다.
    """
    def __init__(self, db_path="social_affinity.json"):
        self.db_path = db_path
        self.affinity_map = self._load_db() # {user_id: score}

    def _load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except: pass
        return {}

    def save_db(self):
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.affinity_map, f, ensure_ascii=False, indent=2)

    def update_affinity(self, user_id, user_text):
        """유저의 행동에 따라 호감도 점수 가감"""
        uid = str(user_id)
        if uid not in self.affinity_map:
            self.affinity_map[uid] = 50.0 # 기본 시작 점수 (중립)

        current = self.affinity_map[uid]
        
        # 정규식 패턴 재사용 (EmotionEngine의 패턴과 유사)
        # 1. 긍정적 상호작용
        if re.search(r"(고마워|좋아|사랑|재밌|귀여워|똑똑)", user_text):
            current += 2.0
        # 2. 부정적 상호작용
        if re.search(r"(바보|멍청|싫어|짜증|노잼|닥쳐)", user_text):
            current -= 3.0 # 신뢰를 잃는 건 더 쉽다
        # 3. 단순 발화 (관심도 유지)
        current += 0.1 

        # 점수 클램핑 (0 ~ 100)
        self.affinity_map[uid] = max(0.0, min(100.0, current))
        self.save_db()

    def get_relationship_context(self, user_id):
        """특정 유저와의 관계를 자연어로 반환"""
        uid = str(user_id)
        score = self.affinity_map.get(uid, 50.0)

        if score >= 80: return f"User({uid})는 너의 '절친'이다. 아주 편하게 대하고 가끔 장난도 쳐라."
        if score >= 60: return f"User({uid})는 '친한 친구'다. 호의적으로 대해라."
        if score <= 20: return f"User({uid})와는 사이가 좋지 않다. 약간 냉소적이거나 사무적으로 대해라."
        if score <= 40: return f"User({uid})와는 약간 서먹하다. 예의를 차려라."
        return f"User({uid})와는 평범한 지인 관계다."