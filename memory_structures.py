import time
import re

class MemoryObject:
    """
    기억의 최소 단위. 우선순위 큐에서 비교 가능하도록 __lt__ 구현.
    """
    def __init__(self, content, user_id, role, importance=1.0):
        self.content = content
        self.user_id = str(user_id)
        self.role = role  # 'user' or 'assistant'
        self.timestamp = time.time()
        self.importance = importance  # 1.0 ~ 10.0
        self.access_count = 1

    def calculate_retention_score(self):
        """
        큐에서 살아남을 점수 계산.
        공식: (중요도 * 0.6) + (최신성 * 0.4) + (반복노출보너스)
        """
        time_diff = time.time() - self.timestamp
        # 최신성: 시간이 지날수록 0에 수렴 (반감기 적용)
        recency = 1.0 / (1.0 + (time_diff / 3600)) 
        
        # 리허설 효과: 많이 조회된 기억은 가산점
        repetition_bonus = min(self.access_count * 0.1, 1.0)
        
        return (self.importance * 0.6) + (recency * 0.4) + repetition_bonus

    def __lt__(self, other):
        # Min-Heap이므로 점수가 *낮은* 것이 작다고 판단되어 먼저 추출(방출)됨
        return self.calculate_retention_score() < other.calculate_retention_score()

class ImportanceScorer:
    """
    [System 1] NLP 기반 경량화 중요도 채점기.
    LLM 호출 없이 키워드와 패턴으로 점수(1.0~10.0)를 매김.
    """
    def __init__(self, bot_name="잼봇"):
        self.bot_name = bot_name
        # 감정 키워드 (확장 가능)
        self.emotional_words = re.compile(r"(사랑|슬퍼|화나|행복|우울|짜증|기뻐|고마워|미안|축하)")
        # 정보성 키워드
        self.info_words = re.compile(r"(사실|알고보니|뉴스|정보|팁|방법|계획|예정)")
        # 노이즈 패턴 (단순 웃음)
        self.noise_pattern = re.compile(r"^([ㅋㅎ]{2,}|[?.!]{2,})$") # ㅋㅋ, ㅎㅎ만 있는 경우

    def score(self, text, user_role="user"):
        score = 1.0 # 기본 점수
        
        # [cite_start]1. 봇 이름 언급 (칵테일 파티 효과) -> 주의 집중 [cite: 2]
        if self.bot_name in text:
            score += 3.0
            
        # [cite_start]2. 감정적 어휘 사용 -> 편도체 자극 (기억 강화) [cite: 18]
        if self.emotional_words.search(text):
            score += 2.0
            
        # 3. 정보성/계획 언급
        if self.info_words.search(text):
            score += 1.5

        # 4. 의문문 (상호작용 시도)
        if "?" in text:
            score += 0.5
            
        # 5. 텍스트 길이 보정 (너무 짧으면 감점, 적당히 길면 가산)
        length = len(text)
        if length > 20: score += 1.0
        elif length < 3: score -= 0.5

        # 6. 노이즈 필터링 (단순 ㅋㅋ, ㅎㅎ는 점수 대폭 깎아서 빨리 망각되게 함)
        if self.noise_pattern.match(text.strip()):
            score = 0.5 

        # 7. 봇 자신의 말은 적당한 중요도 유지 (맥락 유지용)
        if user_role == "assistant":
            score = max(score, 2.0)

        # 1.0 ~ 10.0 클램핑
        return max(1.0, min(10.0, score))