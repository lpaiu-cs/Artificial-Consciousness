from typing import List

from memory.ontology import Facet
from memory.schema import QueryPlan


class QueryPlanner:
    """Lightweight heuristic planner for state-first retrieval."""

    @staticmethod
    def plan(user_input: str, user_id: str) -> QueryPlan:
        text = (user_input or "").strip().lower()
        requested_facets: List[str] = [
            Facet.COMMITMENT_OPEN_LOOP.value,
            Facet.BOUNDARY_RULE.value,
        ]

        def include(facet: Facet):
            value = facet.value
            if value not in requested_facets:
                requested_facets.append(value)

        if any(token in text for token in ["이름", "불러", "호칭", "닉네임"]):
            include(Facet.IDENTITY_PREFERRED_NAME)
            include(Facet.IDENTITY_ALIAS)

        if any(token in text for token in ["한국어", "영어", "길게", "짧게", "자세히", "간단히", "지적", "말투"]):
            include(Facet.INTERACTION_PREFERENCE)

        if any(token in text for token in ["좋아", "싫어", "선호", "취향", "prefer", "like", "hate"]):
            include(Facet.PREFERENCE_ITEM)

        if any(token in text for token in ["예산", "건강", "환경", "기기", "장비", "timezone", "시간대"]):
            include(Facet.CONSTRAINT_CONTEXT)

        if any(token in text for token in ["목표", "진행", "프로젝트", "blocker", "막혔", "해야 해"]):
            include(Facet.GOAL_CURRENT)

        if any(token in text for token in ["내일", "다음 주", "화요일", "일정", "예약", "마감", "언제", "due"]):
            include(Facet.SCHEDULE_EVENT)

        if any(token in text for token in ["친구", "교수", "형제", "팀원", "관계"]):
            include(Facet.RELATION_TO_ENTITY)

        if any(token in text for token in ["기억", "예전에", "지난번", "우리", "농담"]):
            include(Facet.SHARED_MILESTONE)

        time_scope = {}
        if any(token in text for token in ["오늘", "지금", "이번 주"]):
            time_scope["focus"] = "recent"
        elif any(token in text for token in ["내일", "다음 주", "다음달", "마감", "예약"]):
            time_scope["focus"] = "future"

        return QueryPlan(
            target_entities=[user_id],
            requested_facets=requested_facets,
            time_scope=time_scope,
            need_relation_context=True,
            need_evidence=True,
        )
