import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

@dataclass
class MemoryObject:
    """STM과 Reflection, Sensory System에서 데이터 이동을 위한 DTO"""
    content: str
    role: str           # "user", "assistant", "system"
    user_id: str        # 불변 ID ("12345")
    user_name: str      # 당시의 닉네임 ("민초단장")
    mem_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    activation: float = 50.0
    emotion_val: int = 0
    emotion_tag: str = "neutral"
    # [New] 관련 유저 리스트 (다인 대화 대비용, 현재는 user_id와 동일하게 시작)
    related_users: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = field(default=None, repr=False)
    
    def to_dict(self):
        return asdict(self)
    
    def __lt__(self, other):
        # 활성도가 낮은 순서대로 정렬 (Min-Heap)
        return self.activation < other.activation

@dataclass
class RetrievalQuery:
    """LTM 검색 요청 객체"""
    embedding: List[float]
    user_id: str
    keywords: List[str]
    intent: str = "chat"
    current_mood: str = "neutral"

# --- Graph Nodes (LTM Storage Units) ---

@dataclass
class BaseNode:
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # 임베딩은 JSON 저장 시 제외하고 별도 파일로 관리됨
    embedding: Optional[List[float]] = field(default=None, repr=False)
    edges: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)

@dataclass
class EpisodeNode(BaseNode):
    """Tier 1: 구체적인 사건 (Immutable)"""
    content: str = ""
    timestamp: float = 0.0
    emotion_tag: str = "neutral"
    user_id: str = "unknown" # 주 화자 식별용
    type: str = "episode"

@dataclass
class InsightNode(BaseNode):
    """Tier 2: 추론된 지식 (Mutable)"""
    summary: str = ""
    confidence: float = 0.5
    last_updated: float = field(default_factory=time.time)
    type: str = "insight"

@dataclass
class EntityNode(BaseNode):
    """Tier 3: 인물 및 정체성 정보 (Social Map 대체)"""
    user_id: str = ""          # 불변 ID ("12345") - 검색 Key
    nickname: str = ""         # 현재 호칭 ("민초단장")
    nickname_history: List[str] = field(default_factory=list) # 닉네임 변경 이력
    affinity: float = 50.0     # 호감도 (0.0 ~ 100.0)
    tags: List[str] = field(default_factory=list) # ["친구", "장난꾸러기"]
    type: str = "entity"