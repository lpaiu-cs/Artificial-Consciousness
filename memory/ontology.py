from dataclasses import dataclass
from enum import Enum
from typing import Dict, Literal, Tuple, Union


class Facet(str, Enum):
    IDENTITY_PREFERRED_NAME = "identity.preferred_name"
    IDENTITY_ALIAS = "identity.alias"
    INTERACTION_PREFERENCE = "interaction.preference"
    PREFERENCE_ITEM = "preference.item"
    CONSTRAINT_CONTEXT = "constraint.context"
    GOAL_CURRENT = "goal.current"
    COMMITMENT_OPEN_LOOP = "commitment.open_loop"
    SCHEDULE_EVENT = "schedule.event"
    BOUNDARY_RULE = "boundary.rule"
    RELATION_TO_ENTITY = "relation.to_entity"
    SHARED_MILESTONE = "shared_history.milestone"
    SESSION_TEMPORARY = "session.temporary_state"
    TRAIT_HYPOTHESIS = "trait.hypothesis"


@dataclass(frozen=True)
class FacetSpec:
    name: str
    key_fields: Tuple[str, ...]
    merge_policy: Literal[
        "replace",
        "set_union",
        "statusful",
        "multi_active",
        "interval",
        "sticky",
        "hypothesis",
    ]
    ttl_policy: Literal["session", "timebound", "persistent", "review"]
    promotion_policy: Literal[
        "explicit_only",
        "explicit_or_repeated",
        "inferred_with_confirmation",
        "never_canonical",
    ]
    default_sensitivity: Literal["low", "personal", "high"]
    retrieval_priority: int


FACET_SPECS: Dict[Facet, FacetSpec] = {
    Facet.IDENTITY_PREFERRED_NAME: FacetSpec(
        name=Facet.IDENTITY_PREFERRED_NAME.value,
        key_fields=(),
        merge_policy="replace",
        ttl_policy="persistent",
        promotion_policy="explicit_only",
        default_sensitivity="personal",
        retrieval_priority=1,
    ),
    Facet.IDENTITY_ALIAS: FacetSpec(
        name=Facet.IDENTITY_ALIAS.value,
        key_fields=("alias",),
        merge_policy="set_union",
        ttl_policy="persistent",
        promotion_policy="explicit_only",
        default_sensitivity="personal",
        retrieval_priority=3,
    ),
    Facet.INTERACTION_PREFERENCE: FacetSpec(
        name=Facet.INTERACTION_PREFERENCE.value,
        key_fields=("dimension",),
        merge_policy="replace",
        ttl_policy="persistent",
        promotion_policy="explicit_only",
        default_sensitivity="personal",
        retrieval_priority=1,
    ),
    Facet.PREFERENCE_ITEM: FacetSpec(
        name=Facet.PREFERENCE_ITEM.value,
        key_fields=("domain", "target"),
        merge_policy="statusful",
        ttl_policy="review",
        promotion_policy="explicit_or_repeated",
        default_sensitivity="personal",
        retrieval_priority=4,
    ),
    Facet.CONSTRAINT_CONTEXT: FacetSpec(
        name=Facet.CONSTRAINT_CONTEXT.value,
        key_fields=("kind", "value"),
        merge_policy="statusful",
        ttl_policy="review",
        promotion_policy="explicit_or_repeated",
        default_sensitivity="personal",
        retrieval_priority=2,
    ),
    Facet.GOAL_CURRENT: FacetSpec(
        name=Facet.GOAL_CURRENT.value,
        key_fields=("label", "project_id"),
        merge_policy="multi_active",
        ttl_policy="review",
        promotion_policy="explicit_or_repeated",
        default_sensitivity="personal",
        retrieval_priority=2,
    ),
    Facet.COMMITMENT_OPEN_LOOP: FacetSpec(
        name=Facet.COMMITMENT_OPEN_LOOP.value,
        key_fields=("kind", "text"),
        merge_policy="multi_active",
        ttl_policy="timebound",
        promotion_policy="explicit_only",
        default_sensitivity="personal",
        retrieval_priority=0,
    ),
    Facet.SCHEDULE_EVENT: FacetSpec(
        name=Facet.SCHEDULE_EVENT.value,
        key_fields=("title", "start_at"),
        merge_policy="interval",
        ttl_policy="timebound",
        promotion_policy="explicit_only",
        default_sensitivity="personal",
        retrieval_priority=1,
    ),
    Facet.BOUNDARY_RULE: FacetSpec(
        name=Facet.BOUNDARY_RULE.value,
        key_fields=("kind", "target"),
        merge_policy="sticky",
        ttl_policy="persistent",
        promotion_policy="explicit_only",
        default_sensitivity="high",
        retrieval_priority=0,
    ),
    Facet.RELATION_TO_ENTITY: FacetSpec(
        name=Facet.RELATION_TO_ENTITY.value,
        key_fields=("target_entity_id", "relation_kind"),
        merge_policy="statusful",
        ttl_policy="review",
        promotion_policy="explicit_or_repeated",
        default_sensitivity="personal",
        retrieval_priority=3,
    ),
    Facet.SHARED_MILESTONE: FacetSpec(
        name=Facet.SHARED_MILESTONE.value,
        key_fields=("label",),
        merge_policy="multi_active",
        ttl_policy="persistent",
        promotion_policy="explicit_or_repeated",
        default_sensitivity="personal",
        retrieval_priority=5,
    ),
    Facet.SESSION_TEMPORARY: FacetSpec(
        name=Facet.SESSION_TEMPORARY.value,
        key_fields=("kind",),
        merge_policy="replace",
        ttl_policy="session",
        promotion_policy="explicit_or_repeated",
        default_sensitivity="personal",
        retrieval_priority=6,
    ),
    Facet.TRAIT_HYPOTHESIS: FacetSpec(
        name=Facet.TRAIT_HYPOTHESIS.value,
        key_fields=("label",),
        merge_policy="hypothesis",
        ttl_policy="review",
        promotion_policy="never_canonical",
        default_sensitivity="personal",
        retrieval_priority=8,
    ),
}

DEFAULT_FACET_SPEC = FacetSpec(
    name="unknown",
    key_fields=(),
    merge_policy="replace",
    ttl_policy="review",
    promotion_policy="explicit_or_repeated",
    default_sensitivity="personal",
    retrieval_priority=9,
)


def get_facet_spec(facet: Union[str, Facet]) -> FacetSpec:
    try:
        facet_enum = facet if isinstance(facet, Facet) else Facet(facet)
    except ValueError:
        return DEFAULT_FACET_SPEC
    return FACET_SPECS.get(facet_enum, DEFAULT_FACET_SPEC)
