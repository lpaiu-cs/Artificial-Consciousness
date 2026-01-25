import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

@dataclass
class MemoryObject:
    """STM과 Reflection에서 사용되는 기본 기억 단위"""
    content: str
    role: str
    user_id: str
    user_name: str
    mem_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    activation: float = 50.0
    emotion_val: int = 0
    emotion_tag: str = "neutral"
    
    def to_dict(self):
        return asdict(self)

@dataclass
class RetrievalQuery:
    """LTM 검색 요청 객체"""
    embedding: List[float]
    user_id: str
    keywords: List[str]
    intent: str = "chat"
    current_mood: str = "neutral"

# --- Graph Nodes ---
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
    type: str = "episode"

@dataclass
class InsightNode(BaseNode):
    summary: str = ""
    subject: str = ""
    predicate: str = ""
    object: str = ""
    confidence: float = 0.5
    last_updated: float = 0.0
    type: str = "insight"