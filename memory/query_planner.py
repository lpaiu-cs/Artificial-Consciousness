from typing import Any, Dict, List, Optional, Sequence

from memory.ontology import Facet
from memory.schema import QueryPlan


class QueryPlanner:
    """Lightweight heuristic planner for state-first retrieval."""

    GENERIC_REFERENT_MARKERS = (
        "걔",
        "걔는",
        "걔가",
        "그 친구",
        "그 사람",
        "그분",
        "그 애",
        "아까 그 사람",
        "아까 그 친구",
        "그 교수님",
        "그 교수",
        "그 팀원",
        "그 동료",
    )

    ROLE_ALIASES = {
        "friend": ["friend", "친구"],
        "professor": ["professor", "교수", "교수님"],
        "teammate": ["teammate", "teammate", "팀원", "팀메이트"],
        "coworker": ["coworker", "colleague", "동료", "직장동료"],
        "sibling": ["sibling", "형제", "형", "누나", "언니", "오빠", "동생"],
        "family": ["family", "가족", "부모", "엄마", "아빠", "어머니", "아버지"],
        "partner": ["partner", "연인", "애인", "남자친구", "여자친구"],
        "person": ["person", "사람", "그분", "분"],
    }

    @staticmethod
    def plan(user_input: str, user_id: str,
             known_entities: Optional[Sequence[Dict[str, object]]] = None,
             session_referents: Optional[Sequence[Dict[str, object]]] = None) -> QueryPlan:
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

        resolution = QueryPlanner._resolve_target_entities(
            user_id,
            text,
            known_entities or [],
            session_referents or [],
        )
        target_entities = resolution["target_entities"]
        if target_entities and target_entities != [user_id]:
            include(Facet.RELATION_TO_ENTITY)

        return QueryPlan(
            target_entities=target_entities,
            requested_facets=requested_facets,
            entity_hints=resolution["entity_hints"],
            unresolved_references=resolution["unresolved_references"],
            time_scope=time_scope,
            need_relation_context=True,
            need_evidence=True,
        )

    @staticmethod
    def expand_role_aliases(role_value: Any) -> List[str]:
        normalized = str(role_value or "").strip().lower()
        if not normalized:
            return []

        aliases = {normalized}
        for canonical, raw_aliases in QueryPlanner.ROLE_ALIASES.items():
            normalized_aliases = {alias.strip().lower() for alias in raw_aliases if alias}
            if normalized == canonical or normalized in normalized_aliases:
                aliases.add(canonical)
                aliases.update(normalized_aliases)
        return list(dict.fromkeys(alias for alias in aliases if alias))

    @staticmethod
    def _resolve_target_entities(user_id: str, text: str,
                                 known_entities: Sequence[Dict[str, object]],
                                 session_referents: Sequence[Dict[str, object]]) -> Dict[str, Any]:
        inventory = QueryPlanner._build_entity_inventory(known_entities, session_referents)
        matched = QueryPlanner._match_entities_by_name(text, inventory)
        if matched:
            return {
                "target_entities": matched,
                "entity_hints": QueryPlanner._entity_hints_for(matched, inventory),
                "unresolved_references": [],
            }

        requested_roles = QueryPlanner._detect_requested_roles(text)
        cached_match = QueryPlanner._resolve_from_referent_cache(
            user_id,
            requested_roles,
            text,
            session_referents,
        )
        if cached_match:
            return {
                "target_entities": [cached_match["entity_id"]],
                "entity_hints": {
                    cached_match["entity_id"]: QueryPlanner._entity_hint(cached_match),
                },
                "unresolved_references": [],
            }

        role_match = QueryPlanner._resolve_by_role_aliases(
            user_id,
            requested_roles,
            inventory,
        )
        if role_match:
            return {
                "target_entities": [role_match["entity_id"]],
                "entity_hints": {
                    role_match["entity_id"]: QueryPlanner._entity_hint(role_match),
                },
                "unresolved_references": [],
            }

        if QueryPlanner._looks_like_referent_request(text, requested_roles):
            requested_role_text = ", ".join(sorted(requested_roles))
            if requested_role_text:
                detail = f"역할 단서({requested_role_text})만으로 대상 entity를 확정하지 못함"
            else:
                detail = "최근 referent cache가 없어 '걔/그 사람'의 대상을 확정하지 못함"
            return {
                "target_entities": [],
                "entity_hints": {},
                "unresolved_references": [detail],
            }

        return {
            "target_entities": [user_id],
            "entity_hints": {},
            "unresolved_references": [],
        }

    @staticmethod
    def _build_entity_inventory(known_entities: Sequence[Dict[str, object]],
                                session_referents: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, Any]]:
        inventory: Dict[str, Dict[str, Any]] = {}
        for source in list(known_entities) + list(session_referents):
            entity_id = str(source.get("entity_id", ""))
            if not entity_id:
                continue

            entry = inventory.setdefault(
                entity_id,
                {"entity_id": entity_id, "names": [], "roles": [], "last_seen": 0.0},
            )
            for name in source.get("names", []) or []:
                normalized_name = str(name or "").strip()
                if normalized_name:
                    entry["names"].append(normalized_name)
            for role in source.get("roles", []) or []:
                entry["roles"].extend(QueryPlanner.expand_role_aliases(role))
            if source.get("last_seen"):
                entry["last_seen"] = max(float(source.get("last_seen") or 0.0), entry["last_seen"])

            entry["names"] = list(dict.fromkeys(entry["names"]))
            entry["roles"] = list(dict.fromkeys(role for role in entry["roles"] if role))
        return inventory

    @staticmethod
    def _match_entities_by_name(text: str,
                                inventory: Dict[str, Dict[str, Any]]) -> List[str]:
        matched: List[str] = []
        for entity_id, entity in inventory.items():
            for name in entity.get("names", []):
                normalized_name = str(name or "").strip().lower()
                if len(normalized_name) < 2:
                    continue
                if normalized_name in text and entity_id not in matched:
                    matched.append(entity_id)
                    break
        return matched

    @staticmethod
    def _detect_requested_roles(text: str) -> List[str]:
        requested_roles: List[str] = []
        for canonical, aliases in QueryPlanner.ROLE_ALIASES.items():
            normalized_aliases = [alias.strip().lower() for alias in aliases if alias]
            if any(alias in text for alias in normalized_aliases):
                requested_roles.append(canonical)
        return list(dict.fromkeys(requested_roles))

    @staticmethod
    def _resolve_from_referent_cache(user_id: str, requested_roles: Sequence[str], text: str,
                                     session_referents: Sequence[Dict[str, object]]) -> Optional[Dict[str, Any]]:
        referents = [
            QueryPlanner._entity_hint(entry)
            for entry in session_referents
            if str(entry.get("entity_id", "")) and str(entry.get("entity_id")) != str(user_id)
        ]
        if not referents:
            return None

        requested_role_set = set(requested_roles)
        if requested_role_set:
            role_matches = [
                referent for referent in referents
                if requested_role_set.intersection(set(referent.get("roles", [])))
            ]
            if role_matches:
                return role_matches[0]

        if any(marker in text for marker in QueryPlanner.GENERIC_REFERENT_MARKERS):
            if len(referents) == 1:
                return referents[0]
            return None

        return None

    @staticmethod
    def _resolve_by_role_aliases(user_id: str, requested_roles: Sequence[str],
                                 inventory: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not requested_roles:
            return None

        requested_role_set = set(requested_roles)
        candidates = [
            entity for entity in inventory.values()
            if entity["entity_id"] != str(user_id)
            and requested_role_set.intersection(set(entity.get("roles", [])))
        ]
        if not candidates:
            return None

        candidates.sort(key=lambda entity: entity.get("last_seen", 0.0), reverse=True)
        if len(candidates) == 1:
            return candidates[0]
        freshest = candidates[0]
        if freshest.get("last_seen", 0.0) > 0.0:
            return freshest
        return None

    @staticmethod
    def _looks_like_referent_request(text: str, requested_roles: Sequence[str]) -> bool:
        if any(marker in text for marker in QueryPlanner.GENERIC_REFERENT_MARKERS):
            return True
        return bool(requested_roles)

    @staticmethod
    def _entity_hints_for(entity_ids: Sequence[str],
                          inventory: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return {
            entity_id: QueryPlanner._entity_hint(inventory.get(entity_id, {"entity_id": entity_id}))
            for entity_id in entity_ids
        }

    @staticmethod
    def _entity_hint(entity: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "entity_id": str(entity.get("entity_id", "")),
            "names": list(dict.fromkeys(str(name) for name in (entity.get("names") or []) if name)),
            "roles": list(dict.fromkeys(str(role) for role in (entity.get("roles") or []) if role)),
            "last_seen": float(entity.get("last_seen") or 0.0),
        }
