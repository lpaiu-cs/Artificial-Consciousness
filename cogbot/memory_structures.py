import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, ClassVar

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
        if self.activation != other.activation:
            # 활성도가 낮은 순서대로 정렬 (Min-Heap)
            return self.activation < other.activation
        return self.timestamp < other.timestamp # 동점 시 오래된 것이 우선

@dataclass
class RetrievalQuery:
    """LTM 검색 요청 객체"""
    embedding: List[float]
    user_id: str
    keywords: List[str]
    query_text: str = ""
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

    def to_graph_dict(self):
        return self.to_dict()

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
class NoteNode(BaseNode):
    """Narrative or impression memory for retrieval, not canonical state."""
    note_type: str = "narrative"
    summary: str = ""
    tags: List[str] = field(default_factory=list)
    confidence: float = 0.5
    related_entity_ids: List[str] = field(default_factory=list)
    evidence_episode_ids: List[str] = field(default_factory=list)
    type: str = "note"


@dataclass
class ClaimNode(BaseNode):
    """Current or reviewable state with facet-specific merge rules."""
    BOUNDARY_PUBLIC_VALUE_FIELDS: ClassVar[frozenset[str]] = frozenset({"kind", "policy_kind"})
    BOUNDARY_PUBLIC_QUALIFIER_FIELDS: ClassVar[frozenset[str]] = frozenset({"topic_label", "audience_ids", "participants"})

    subject_id: str = ""
    facet: str = ""
    merge_key: str = ""
    value: Dict[str, Any] = field(default_factory=dict)
    qualifiers: Dict[str, Any] = field(default_factory=dict)
    nl_summary: str = ""
    source_type: str = "explicit"
    confidence: float = 0.5
    status: str = "active"
    valid_from: Optional[float] = None
    valid_to: Optional[float] = None
    last_confirmed_at: Optional[float] = None
    evidence_episode_ids: List[str] = field(default_factory=list)
    sensitivity: str = "personal"
    scope: str = "user_private"
    type: str = "claim"

    def to_graph_dict(self):
        data = self.to_dict()
        if self.facet != "boundary.rule":
            return data

        public_value = {
            key: value
            for key, value in (data.get("value") or {}).items()
            if key in self.BOUNDARY_PUBLIC_VALUE_FIELDS and value not in (None, "", [], {})
        }
        public_qualifiers = {
            key: value
            for key, value in (data.get("qualifiers") or {}).items()
            if key in self.BOUNDARY_PUBLIC_QUALIFIER_FIELDS and value not in (None, "", [], {})
        }

        data["value"] = public_value
        data["qualifiers"] = public_qualifiers
        return data

@dataclass
class EntityNode(BaseNode):
    """Tier 3: 인물 및 정체성 정보 (Social Map 대체)"""
    user_id: str = ""          # 불변 ID ("12345") - 검색 Key
    nickname: str = ""         # 현재 호칭 ("민초단장")
    nickname_history: List[str] = field(default_factory=list) # 닉네임 변경 이력
    affinity: float = 50.0     # 호감도 (0.0 ~ 100.0)
    tags: List[str] = field(default_factory=list) # ["친구", "장난꾸러기"]
    type: str = "entity"


@dataclass
class RelationState:
    entity_id: str
    trust: float = 0.5
    warmth: float = 0.5
    familiarity: float = 0.1
    respect: float = 0.5
    tension: float = 0.0
    reliability: float = 0.5
    last_interaction_at: float = 0.0


@dataclass
class OpenLoop:
    loop_id: str
    owner_id: str
    kind: str
    text: str
    due_at: Optional[float] = None
    status: str = "open"
    priority: int = 0
    evidence_episode_ids: List[str] = field(default_factory=list)
