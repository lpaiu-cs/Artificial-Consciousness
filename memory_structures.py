import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class MemoryObject:
    """
    [Active Memory]
    모든 기억은 '활성화 점수(Activation)'를 가집니다.
    이 점수는 참조(Retrieval)될 때 오르고, 시간이 지나거나 참조되지 않으면 떨어집니다.
    """
    content: str
    role: str          # 'user' or 'assistant'
    user_id: str
    user_name: str
    
    # 고유 ID (참조 및 갱신용)
    mem_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    timestamp: float = field(default_factory=time.time)
    
    # [Dynamic Scorer]
    # 0.0 ~ 100.0 (임계값 미만 시 Eviction)
    activation: float = 50.0 
    
    # 감정 태그 (추후 사용)
    emotion_val: int = 0
    emotion_tag: str = "neutral"
    
    # 검색용 임베딩
    embedding: Optional[List[float]] = None

    def __lt__(self, other):
        # Priority Queue용: 활성화 점수가 낮은 순으로 정렬 (Eviction 후보)
        return self.activation < other.activation
    

@dataclass
class RetrievalQuery:
    """
    STM -> LTM 검색 요청 객체
    단순 텍스트가 아닌 '맥락'을 담아서 검색함
    """
    embedding: List[float]       # 의미 검색의 핵심
    user_id: str                 # 누구와의 기억인가?
    intent: Optional[str] = None # (옵션) 질문, 위로, 잡담 등
    current_mood: str = "neutral" # 봇의 현재 기분 (기분 일치성 효과용)
    keywords: List[str] = field(default_factory=list) # 고유명사 매칭용