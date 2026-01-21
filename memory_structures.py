import time
import math
import re
import numpy as np

# [보고서 3.1 & 3.2] 망각 및 활성화 상수
DECAY_RATE = 0.5  # ACT-R 감쇠 계수 (d)
NOISE_THRESHOLD = 1.5  # 주의 필터 임계값

class MemoryObject:
    """
    [보고서 6.1] 메모리 객체 스키마 구현
    - content: 기억 내용
    - creation_time: 생성 시점 (t0)
    - access_history: 리허설(반복 인출) 기록 (t1, t2...) -> ACT-R 수식용
    - importance: 정서적 현저성 (1~10) -> 편도체 역할
    - emotion_tag: 당시의 감정 상태 (OCC 모델 간소화)
    """
    def __init__(self, content, user_id, role, importance=1.0, emotion="neutral"):
        self.content = content
        self.user_id = str(user_id)
        self.role = role
        self.creation_time = time.time()
        
        # [ACT-R 필수] 인출될 때마다 시간이 기록됨 (리허설 효과)
        self.access_history = [time.time()] 
        
        self.importance = importance
        self.emotion_tag = emotion

    def get_base_activation(self):
        """
        [보고서 3.2] ACT-R 기저 수준 활성화 (Base-Level Activation) 계산
        공식: B_i = ln( sum( t_j ^ -d ) ) + Importance_Bias
        - 자주(Frequency), 최근에(Recency) 인출될수록 값이 높음.
        - 중요도(Importance)가 높으면 기본 활성화 수준이 높음 (편도체 보정).
        """
        now = time.time()
        summation = 0
        
        for access_time in self.access_history:
            # 시간 차이 (초 단위 -> 시간 단위 변환 후 계산)
            t_diff = (now - access_time) / 3600.0 
            # 0으로 나누기 방지 (최소 1초)
            t_diff = max(t_diff, 0.0002) 
            
            summation += math.pow(t_diff, -DECAY_RATE)
            
        # 기본 활성화 점수 (로그 스케일)
        base_activation = math.log(summation)
        
        # 중요도에 따른 가중치 추가 (보고서 4.1: 정서적 강화)
        # 중요도(1~10)를 로그 스케일에 맞춰 보정
        emotional_boost = math.log(self.importance + 1) 
        
        return base_activation + emotional_boost

    def __lt__(self, other):
        # Min-Heap(우선순위 큐)용 비교 연산자
        # 활성화 수준이 낮은 기억이 먼저 방출(Eviction)됨
        return self.get_base_activation() < other.get_base_activation()

class CognitiveScorer:
    """
    [보고서 2.1.1] 감각 등록기 및 주의(Attention) 필터
    단순 키워드 매칭을 넘어, '정서적 자극'과 '정보 가치'를 평가함.
    """
    def __init__(self):
        self.emotional_pattern = re.compile(r"(사랑|슬퍼|화나|행복|우울|짜증|기뻐|충격|대박|ㅠㅠ|ㅋㅋ)")
        self.info_pattern = re.compile(r"(사실|계획|예정|약속|주소|번호|추천)")

    def calculate_attention(self, text, role):
        """
        입력된 정보가 '주의(Attention)'를 끌 만한지 점수화 (1.0 ~ 10.0)
        """
        score = 1.0 # 기본 점수
        
        # 1. 정서적 자극 (Arousal)
        if self.emotional_pattern.search(text):
            score += 3.0 # 감정은 기억에 강하게 남음
            
        # 2. 정보적 가치 (Information)
        if self.info_pattern.search(text):
            score += 2.0
            
        # 3. 자기 참조 (Cocktail Party Effect) - 봇이 언급됨
        if "잼봇" in text:
            score += 4.0
            
        # 4. 길이 및 노이즈 패널티
        if len(text) < 3: score -= 0.5
        
        # 5. 봇 자신의 발화는 맥락 유지를 위해 기본 점수 보장
        if role == "assistant":
            score = max(score, 3.0)

        return max(1.0, min(10.0, score))