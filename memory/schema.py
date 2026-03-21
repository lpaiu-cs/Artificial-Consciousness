from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from memory_structures import ClaimNode, EpisodeNode, InsightNode, NoteNode, RelationState


@dataclass
class QueryPlan:
    target_entities: List[str]
    requested_facets: List[str]
    entity_hints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    unresolved_references: List[str] = field(default_factory=list)
    time_scope: Dict[str, Any] = field(default_factory=dict)
    need_relation_context: bool = True
    need_evidence: bool = True


@dataclass
class ContextBundle:
    plan: QueryPlan
    open_loops: List[ClaimNode] = field(default_factory=list)
    active_claims: List[ClaimNode] = field(default_factory=list)
    relevant_schedule: List[ClaimNode] = field(default_factory=list)
    interaction_policy: Dict[str, Any] = field(default_factory=dict)
    relation_state: Optional[RelationState] = None
    supporting_events: List[EpisodeNode] = field(default_factory=list)
    supporting_notes: List[NoteNode] = field(default_factory=list)
    legacy_insights: List[InsightNode] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
