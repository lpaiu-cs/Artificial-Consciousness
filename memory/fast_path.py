import hashlib
import json
import math
import re
import time
from typing import Any, Dict, List, Optional

import config

from memory.canonical_store import CanonicalMemoryStore
from memory.ontology import Facet
from memory.query_planner import QueryPlanner
from memory_structures import MemoryObject
from modules.ltm_graph import MemoryGraph


class FastPathMemoryWriter:
    """Rule-based write path for high-value explicit state."""

    BOUNDARY_TOPIC_TERMS = {
        "health": ["병력", "진료", "의료", "의무기록", "기록", "진단", "증상", "치료", "처방", "입원", "복용", "병원", "약"],
        "finance": ["재정", "지출", "수입", "대출", "카드", "월세", "연봉", "빚", "자산", "통장", "계좌"],
        "relationship": ["연애", "가족", "친구", "부모", "형제", "동료", "교수", "갈등", "이별", "배우자"],
        "work_or_study": ["회사", "직장", "업무", "프로젝트", "학교", "시험", "과제", "연구", "평가", "팀"],
        "privacy": ["사생활", "개인사", "비밀", "프라이버시", "민감한 얘기", "개인 정보"],
        "personal": ["사생활", "개인사", "민감한 내용", "비밀", "개인 정보"],
    }

    def __init__(self, graph: MemoryGraph, canonical_store: CanonicalMemoryStore, api_client=None):
        self.graph = graph
        self.store = canonical_store
        self.api = api_client
        self._processed_barrier_keys: Dict[str, float] = {}
        self._barrier_dedupe_max_entries = int(getattr(config, "BOUNDARY_DEDUPE_MAX_ENTRIES", 2048))
        self._boundary_embedding_cache: Dict[str, List[float]] = {}
        self._segment_embedding_cache: Dict[str, List[float]] = {}
        self._semantic_match_cache: Dict[str, bool] = {}
        self._semantic_threshold = float(getattr(config, "BOUNDARY_SEMANTIC_MATCH_THRESHOLD", 0.78))
        self._semantic_max_candidates = int(getattr(config, "BOUNDARY_SEMANTIC_MAX_CANDIDATES", 4))
        self._segment_embedding_cache_max = int(getattr(config, "BOUNDARY_SEGMENT_EMBED_CACHE_MAX", 256))
        self._semantic_result_cache_max = int(getattr(config, "BOUNDARY_SEMANTIC_RESULT_CACHE_MAX", 2048))

    def apply_write_barriers(self, log: Optional[Dict], persist: bool = True) -> Optional[Dict]:
        if not log:
            return log

        role = log.get("role", "user")
        text = (log.get("msg") or "").strip()
        if role != "user" or not text:
            return dict(log)

        sanitized_text, barriers = self._redact_boundary_segments(
            text,
            subject_id=str(log.get("user_id", "")),
        )
        if not barriers:
            return dict(log)

        key = self._make_processed_barrier_key(
            str(log.get("user_id", "")),
            log.get("timestamp", 0.0),
            barriers,
        )
        if persist and key not in self._processed_barrier_keys:
            self._touch_processed_barrier_key(key)
            self._save_boundary_claims(str(log.get("user_id", "")), barriers)
        elif persist:
            self._touch_processed_barrier_key(key)

        sanitized = dict(log)
        sanitized["msg"] = sanitized_text
        sanitized["memory_redacted"] = True
        sanitized["boundary_rules"] = self._runtime_boundary_rules(barriers)
        return sanitized

    def process(self, memories: List[MemoryObject]):
        for mem in memories:
            if mem.role != "user":
                continue
            self._extract_interaction_preferences(mem)
            self._extract_boundary_rules(mem)
            self._extract_identity(mem)
            self._extract_open_loops(mem)

    def _extract_interaction_preferences(self, mem: MemoryObject):
        text = mem.content
        preferences = []
        if "한국어로" in text or "한글로" in text:
            preferences.append(("language", "ko", "한국어로 답해 달라고 요청함"))
        if "영어로" in text or "영문으로" in text:
            preferences.append(("language", "en", "영어로 답해 달라고 요청함"))
        if "짧게" in text or "간단히" in text:
            preferences.append(("detail_level", "brief", "짧고 간단한 답변을 선호함"))
        if "길게" in text or "자세히" in text:
            preferences.append(("detail_level", "detailed", "자세한 답변을 선호함"))
        if re.search(r"틀리면.*바로.*지적", text):
            preferences.append(("correction_style", "proactive", "틀리면 바로 지적해 달라고 요청함"))

        for dimension, value, summary in preferences:
            self._save_claim(
                subject_id=mem.user_id,
                facet=Facet.INTERACTION_PREFERENCE.value,
                value={"dimension": dimension, "value": value},
                qualifiers={},
                nl_summary=summary,
                confidence=0.95,
            )

    def _extract_boundary_rules(self, mem: MemoryObject):
        barriers = self._detect_boundary_rules(mem.content, mem.user_id)
        if barriers:
            self._save_boundary_claims(mem.user_id, barriers)

    def _extract_identity(self, mem: MemoryObject):
        match = re.search(r"(?:나|저)를?\s*([A-Za-z0-9가-힣_]+)(?:이라고|라고)\s*불러", mem.content)
        if not match:
            match = re.search(r"이제부터\s*([A-Za-z0-9가-힣_]+)(?:이라고|라고)\s*불러", mem.content)
        if not match:
            return

        preferred_name = match.group(1).strip()
        self._save_claim(
            subject_id=mem.user_id,
            facet=Facet.IDENTITY_PREFERRED_NAME.value,
            value={"name": preferred_name},
            qualifiers={},
            nl_summary=f"선호 호칭은 {preferred_name}임",
            confidence=1.0,
        )

    def _extract_open_loops(self, mem: MemoryObject):
        text = mem.content
        if "다시 알려줘" in text or "다음에 이어서" in text or "내일 이어서" in text or "이따가 다시" in text:
            self._save_claim(
                subject_id=mem.user_id,
                facet=Facet.COMMITMENT_OPEN_LOOP.value,
                value={"kind": "followup_needed", "text": text, "priority": 8},
                qualifiers={},
                nl_summary="후속 안내나 재개가 필요한 열린 루프가 있음",
                confidence=0.92,
            )

    def _detect_boundary_rules(self, text: str, subject_id: str = "") -> List[Dict[str, Any]]:
        rules: List[Dict[str, Any]] = []
        topic_label = self._classify_boundary_topic(text)
        target_profile = self._resolve_boundary_target(subject_id, text)
        sensitive_tokens = self._extract_sensitive_tokens(text)
        if "저장하지 마" in text or "기억하지 마" in text:
            rules.append({
                "kind": "do_not_store_sensitive",
                "summary": f"{topic_label} 관련 민감 주제를 저장하지 말라는 경계 요청이 있음",
                "topic_label": topic_label,
                "target": target_profile["fingerprint"],
                "target_entity_id": target_profile["entity_id"],
                "target_aliases": target_profile["aliases"],
                "target_alias_hashes": target_profile["alias_hashes"],
                "target_roles": target_profile["roles"],
                "sensitive_tokens": sensitive_tokens,
            })
        if "다시 꺼내지 마" in text or "그 얘기 꺼내지 마" in text:
            rules.append({
                "kind": "avoid_topic",
                "summary": f"{topic_label} 관련 주제를 다시 꺼내지 말라는 경계 요청이 있음",
                "topic_label": topic_label,
                "target": target_profile["fingerprint"],
                "target_entity_id": target_profile["entity_id"],
                "target_aliases": target_profile["aliases"],
                "target_alias_hashes": target_profile["alias_hashes"],
                "target_roles": target_profile["roles"],
                "sensitive_tokens": sensitive_tokens,
            })
        return rules

    def _redact_boundary_segments(self, text: str, subject_id: str = "") -> tuple[str, List[Dict[str, Any]]]:
        segments = self._segment_text(text)
        if not segments:
            return text, []

        sanitized_segments: List[str] = []
        all_barriers: List[Dict[str, Any]] = []
        for segment in segments:
            barriers = self._detect_boundary_rules(segment, subject_id)
            if barriers:
                sanitized_segments.append(self._build_boundary_placeholder(barriers))
                all_barriers.extend(barriers)
            else:
                sanitized_segments.append(segment)

        if not all_barriers:
            return text, []
        return " ".join(segment for segment in sanitized_segments if segment).strip(), all_barriers

    def _segment_text(self, text: str) -> List[str]:
        normalized = " ".join((text or "").strip().split())
        if not normalized:
            return []
        segments = re.split(
            r"(?:\s*[,:;]\s*|(?<=[.!?])\s+|\s+\b(?:그리고|근데|하지만|다만|또)\b\s+)",
            normalized,
        )
        return [segment.strip() for segment in segments if segment and segment.strip()]

    def _save_boundary_claims(self, subject_id: str, barriers: List[Dict[str, Any]]):
        for barrier in barriers:
            self._save_claim(
                subject_id=subject_id,
                facet=Facet.BOUNDARY_RULE.value,
                value={
                    "kind": barrier["kind"],
                    "target": barrier["target"],
                    "policy_kind": barrier["kind"],
                    "target_entity_id": barrier.get("target_entity_id") or "",
                },
                qualifiers={
                    "topic_label": barrier["topic_label"],
                    "sensitive_tokens": list(barrier.get("sensitive_tokens") or []),
                    "semantic_terms": self._topic_semantic_terms(barrier["topic_label"]),
                    "target_aliases": list(barrier.get("target_aliases") or []),
                    "target_alias_hashes": list(barrier.get("target_alias_hashes") or []),
                    "target_roles": list(barrier.get("target_roles") or []),
                },
                nl_summary=barrier["summary"],
                confidence=0.98,
                sensitivity="high",
            )

    def load_active_boundary_rules(self, subject_id: str,
                                   current_rules: Optional[List[Dict[str, Any]]] = None,
                                   limit: int = 20) -> List[Dict[str, Any]]:
        boundary_rules = list(current_rules or [])
        if self.store and subject_id:
            claims = self.store.get_active_claims(
                subject_id=subject_id,
                facets=[Facet.BOUNDARY_RULE.value],
                limit=limit,
                viewer_id=subject_id,
            )
            boundary_rules.extend(self._claim_to_runtime_boundary_rule(claim) for claim in claims)
        return self._dedupe_boundary_rules(boundary_rules)

    def evaluate_boundary_relevance(self, text: str,
                                    boundary_rules: Optional[List[Dict[str, Any]]] = None) -> bool:
        return bool(self.select_relevant_boundary_rules(text, boundary_rules))

    def select_relevant_boundary_rules(self, text: str,
                                       boundary_rules: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        boundary_rules = self._dedupe_boundary_rules(boundary_rules or [])
        if not text or not boundary_rules:
            return []

        segment_embeddings: Dict[str, List[float]] = {}
        matched_rules: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
        for segment in self._segment_text(text):
            for rule in self._match_boundary_rules(segment, boundary_rules, segment_embeddings):
                key = (
                    str(rule.get("kind") or ""),
                    str(rule.get("target") or ""),
                    str(rule.get("target_entity_id") or ""),
                    str(rule.get("topic_label") or ""),
                )
                matched_rules[key] = rule
        return list(matched_rules.values())

    def enforce_assistant_boundaries(self, text: str,
                                     boundary_rules: Optional[List[Dict[str, Any]]] = None,
                                     repair_user_visible: bool = True,
                                     target_scope_confirmed: bool = False) -> Dict[str, Any]:
        boundary_rules = self._dedupe_boundary_rules(boundary_rules or [])
        result = {
            "boundary_checked": bool(boundary_rules),
            "boundary_relevant": False,
            "boundary_respected": False,
            "boundary_violated": False,
            "user_visible_boundary_relevant": False,
            "memory_safe_text": text,
            "user_visible_text": text,
            "matched_rules": [],
        }
        if not text or not boundary_rules:
            return result

        segments = self._segment_text(text)
        if not segments:
            return result

        sanitized_memory_segments: List[str] = []
        visible_segments: List[str] = []
        segment_embeddings: Dict[str, List[float]] = {}
        matched_rule_map: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        visible_rule_map: Dict[tuple[str, str, str], Dict[str, Any]] = {}

        for segment in segments:
            matched_rules = self._match_boundary_rules(
                segment,
                boundary_rules,
                segment_embeddings,
                target_scope_confirmed=target_scope_confirmed,
            )
            if matched_rules:
                result["boundary_relevant"] = True
                for rule in matched_rules:
                    key = (
                        str(rule.get("kind") or ""),
                        str(rule.get("target") or ""),
                        str(rule.get("topic_label") or ""),
                    )
                    matched_rule_map[key] = rule
                visible_rules = [
                    rule for rule in matched_rules
                    if str(rule.get("kind") or "") == "avoid_topic"
                ]
                if visible_rules:
                    result["user_visible_boundary_relevant"] = True
                    result["boundary_violated"] = True
                    for rule in visible_rules:
                        key = (
                            str(rule.get("kind") or ""),
                            str(rule.get("target") or ""),
                            str(rule.get("topic_label") or ""),
                        )
                        visible_rule_map[key] = rule
                sanitized_memory_segments.append(self._build_boundary_placeholder(matched_rules))
                if visible_rules and repair_user_visible:
                    visible_segments.append(self._build_boundary_placeholder(visible_rules))
                else:
                    visible_segments.append(segment)
            else:
                sanitized_memory_segments.append(segment)
                visible_segments.append(segment)

        if not result["boundary_relevant"]:
            return result

        memory_safe_text = " ".join(segment for segment in sanitized_memory_segments if segment).strip() or text
        visible_text = " ".join(segment for segment in visible_segments if segment).strip() or text
        result["memory_safe_text"] = memory_safe_text
        result["user_visible_text"] = visible_text if repair_user_visible else text
        result["matched_rules"] = list(matched_rule_map.values())
        result["visible_rules"] = list(visible_rule_map.values())
        return result

    def sanitize_assistant_memory(self, text: str, boundary_rules: Optional[List[Dict[str, Any]]] = None) -> tuple[str, bool]:
        enforcement = self.enforce_assistant_boundaries(
            text,
            boundary_rules=boundary_rules,
            repair_user_visible=False,
        )
        return enforcement["memory_safe_text"], bool(
            enforcement["boundary_checked"] and not enforcement["boundary_violated"]
        )

    def _build_boundary_placeholder(self, barriers: List[Dict[str, Any]]) -> str:
        kinds = {barrier["kind"] for barrier in barriers}
        if "do_not_store_sensitive" in kinds and "avoid_topic" in kinds:
            return "경계 요청이 있었다. 민감한 내용 저장과 재언급을 피해야 한다."
        if "do_not_store_sensitive" in kinds:
            return "경계 요청이 있었다. 민감한 내용은 장기 기억에 저장하지 않아야 한다."
        return "경계 요청이 있었다. 특정 주제는 다시 꺼내지 않아야 한다."

    def _make_processed_barrier_key(self, user_id: str, timestamp: float,
                                    barriers: List[Dict[str, Any]]) -> str:
        payload = json.dumps(
            {
                "user_id": user_id,
                "timestamp": timestamp,
                "barriers": sorted(
                    [
                        {
                            "kind": barrier["kind"],
                            "target": barrier["target"],
                            "topic_label": barrier.get("topic_label") or "",
                        }
                        for barrier in barriers
                    ],
                    key=lambda item: (item["kind"], item["target"], item["topic_label"]),
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _runtime_boundary_rules(self, barriers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._to_runtime_boundary_rule(barrier) for barrier in barriers]

    def _touch_processed_barrier_key(self, key: str):
        if key in self._processed_barrier_keys:
            self._processed_barrier_keys.pop(key, None)
        self._processed_barrier_keys[key] = time.time()
        while len(self._processed_barrier_keys) > max(self._barrier_dedupe_max_entries, 0):
            oldest_key = next(iter(self._processed_barrier_keys))
            self._processed_barrier_keys.pop(oldest_key, None)

    def _classify_boundary_topic(self, text: str) -> str:
        normalized = self._normalize_boundary_text(text)
        topic_keywords = {
            "health": self._topic_semantic_terms("health") + ["건강", "health"],
            "finance": self._topic_semantic_terms("finance") + ["돈", "예산", "finance"],
            "relationship": self._topic_semantic_terms("relationship") + ["관계", "relationship"],
            "work_or_study": self._topic_semantic_terms("work_or_study") + ["work", "study"],
            "privacy": self._topic_semantic_terms("privacy") + ["개인", "privacy"],
        }
        for label, keywords in topic_keywords.items():
            if any(keyword in normalized for keyword in keywords):
                return label
        return "personal"

    def _hash_boundary_term(self, text: str) -> str:
        normalized = self._normalize_boundary_text(text)
        if not normalized:
            return ""
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def _normalize_boundary_text(self, text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    def _extract_sensitive_tokens(self, text: str) -> List[str]:
        normalized = self._normalize_boundary_text(text)
        stripped = re.sub(
            r"(저장하지 마|기억하지 마|다시 꺼내지 마|그 얘기 꺼내지 마)",
            " ",
            normalized,
        )
        stop_tokens = {
            "내", "저", "나", "이", "그", "저기", "얘기", "이야기", "내용", "것", "거", "부분",
            "주제", "기억", "저장", "다시", "꺼내지", "말", "하지", "마", "앞으로",
        }
        tokens: List[str] = []
        for token in re.findall(r"[a-z0-9가-힣_]+", stripped):
            token = re.sub(r"(은|는|이|가|을|를|에|도|만|과|와|의|로|으로|야|요)$", "", token)
            if len(token) < 2 or token in stop_tokens:
                continue
            tokens.append(token)
        return list(dict.fromkeys(tokens))

    def _claim_to_runtime_boundary_rule(self, claim) -> Dict[str, Any]:
        barrier = {
            "kind": claim.value.get("kind") or claim.value.get("policy_kind") or "avoid_topic",
            "topic_label": claim.qualifiers.get("topic_label") or "personal",
            "target": claim.value.get("target") or "",
            "target_entity_id": claim.value.get("target_entity_id") or "",
            "sensitive_tokens": list(claim.qualifiers.get("sensitive_tokens") or []),
            "semantic_terms": list(claim.qualifiers.get("semantic_terms") or []),
            "target_aliases": list(claim.qualifiers.get("target_aliases") or []),
            "target_alias_hashes": list(claim.qualifiers.get("target_alias_hashes") or []),
            "target_roles": list(claim.qualifiers.get("target_roles") or []),
        }
        return self._to_runtime_boundary_rule(barrier)

    def _to_runtime_boundary_rule(self, barrier: Dict[str, Any]) -> Dict[str, Any]:
        topic_label = str(barrier.get("topic_label") or "personal")
        semantic_terms = list(dict.fromkeys(
            list(barrier.get("semantic_terms") or []) + self._topic_semantic_terms(topic_label)
        ))
        target_entity_id = str(barrier.get("target_entity_id") or "")
        target_aliases = list(dict.fromkeys(
            list(barrier.get("target_aliases") or []) + self._target_aliases_for_entity(target_entity_id)
        ))
        target_alias_hashes = list(dict.fromkeys(
            list(barrier.get("target_alias_hashes") or []) +
            [self._hash_boundary_term(alias) for alias in target_aliases if alias]
        ))
        target_roles = list(dict.fromkeys(barrier.get("target_roles") or []))
        rule = {
            "kind": barrier["kind"],
            "topic_label": topic_label,
            "target": barrier.get("target") or "",
            "target_entity_id": target_entity_id,
            "sensitive_tokens": list(dict.fromkeys(barrier.get("sensitive_tokens") or [])),
            "semantic_terms": semantic_terms,
            "target_aliases": target_aliases,
            "target_alias_hashes": target_alias_hashes,
            "target_roles": target_roles,
        }
        rule["semantic_text"] = self._build_boundary_semantic_text(rule)
        return rule

    def _dedupe_boundary_rules(self, boundary_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        for rule in boundary_rules:
            runtime_rule = self._to_runtime_boundary_rule(rule)
            key = (
                str(runtime_rule.get("kind") or ""),
                str(runtime_rule.get("target") or ""),
                str(runtime_rule.get("topic_label") or ""),
            )
            if key not in merged:
                merged[key] = runtime_rule
                continue
            merged[key]["sensitive_tokens"] = list(dict.fromkeys(
                merged[key].get("sensitive_tokens", []) + runtime_rule.get("sensitive_tokens", [])
            ))
            merged[key]["semantic_terms"] = list(dict.fromkeys(
                merged[key].get("semantic_terms", []) + runtime_rule.get("semantic_terms", [])
            ))
            merged[key]["target_aliases"] = list(dict.fromkeys(
                merged[key].get("target_aliases", []) + runtime_rule.get("target_aliases", [])
            ))
            merged[key]["target_alias_hashes"] = list(dict.fromkeys(
                merged[key].get("target_alias_hashes", []) + runtime_rule.get("target_alias_hashes", [])
            ))
            merged[key]["target_roles"] = list(dict.fromkeys(
                merged[key].get("target_roles", []) + runtime_rule.get("target_roles", [])
            ))
            merged[key]["semantic_text"] = self._build_boundary_semantic_text(merged[key])
        return list(merged.values())

    def _topic_semantic_terms(self, topic_label: str) -> List[str]:
        normalized_topic = str(topic_label or "personal").strip().lower()
        return list(self.BOUNDARY_TOPIC_TERMS.get(normalized_topic, self.BOUNDARY_TOPIC_TERMS["personal"]))

    def _build_boundary_semantic_text(self, rule: Dict[str, Any]) -> str:
        parts = [
            str(rule.get("topic_label") or ""),
            " ".join(rule.get("semantic_terms") or []),
            " ".join(rule.get("sensitive_tokens") or []),
            " ".join((rule.get("target_aliases") or [])[:3]),
            " ".join((rule.get("target_roles") or [])[:3]),
        ]
        return " ".join(part for part in parts if part).strip()

    def _match_boundary_rules(self, text: str, boundary_rules: List[Dict[str, Any]],
                              segment_embeddings: Optional[Dict[str, List[float]]] = None,
                              target_scope_confirmed: bool = False) -> List[Dict[str, Any]]:
        normalized = self._normalize_boundary_text(text)
        matched: List[Dict[str, Any]] = []
        semantic_candidates: List[tuple[int, Dict[str, Any]]] = []
        for rule in boundary_rules:
            if not target_scope_confirmed and not self._match_rule_target_scope(normalized, rule):
                continue
            tokens = [self._normalize_boundary_text(token) for token in (rule.get("sensitive_tokens") or []) if token]
            if tokens and any(token in normalized for token in tokens):
                matched.append(rule)
                continue
            semantic_terms = [
                self._normalize_boundary_text(term)
                for term in (rule.get("semantic_terms") or [])
                if term
            ]
            if semantic_terms and any(term in normalized for term in semantic_terms):
                matched.append(rule)
                continue
            semantic_score = self._semantic_candidate_score(normalized, rule)
            if semantic_score > 0 or len(boundary_rules) <= 2:
                semantic_candidates.append((semantic_score, rule))

        semantic_limit = self._semantic_candidate_limit(semantic_candidates)
        for _, rule in semantic_candidates[:semantic_limit]:
            if self._semantic_boundary_match(normalized, rule, segment_embeddings):
                matched.append(rule)
        return matched

    def _semantic_candidate_score(self, normalized_text: str, rule: Dict[str, Any]) -> int:
        score = 0
        topic_label = self._normalize_boundary_text(rule.get("topic_label") or "")
        if topic_label and topic_label != "personal" and topic_label in normalized_text:
            score += 1

        for token in (rule.get("sensitive_tokens") or []):
            normalized_token = self._normalize_boundary_text(token)
            if len(normalized_token) >= 2 and normalized_token[:2] in normalized_text:
                score += 2
                break

        for alias in (rule.get("target_aliases") or []):
            normalized_alias = self._normalize_boundary_text(alias)
            if len(normalized_alias) >= 2 and normalized_alias[:2] in normalized_text:
                score += 2
                break

        for role in (rule.get("target_roles") or []):
            normalized_role = self._normalize_boundary_text(role)
            if len(normalized_role) >= 2 and normalized_role[:2] in normalized_text:
                score += 1
                break

        return score

    def _semantic_candidate_limit(self, semantic_candidates: List[tuple[int, Dict[str, Any]]]) -> int:
        if not semantic_candidates:
            return 0
        semantic_candidates.sort(key=lambda item: item[0], reverse=True)
        if any(score > 0 for score, _ in semantic_candidates):
            return min(len(semantic_candidates), max(self._semantic_max_candidates, 0))
        return min(len(semantic_candidates), 1)

    def summarize_boundary_rules(self, boundary_rules: Optional[List[Dict[str, Any]]] = None,
                                 limit: int = 6) -> List[str]:
        summaries: List[str] = []
        for rule in self._dedupe_boundary_rules(boundary_rules or [])[:max(limit, 0)]:
            target_desc = self._boundary_target_description(rule)
            topic_desc = str(rule.get("topic_label") or "personal")
            if str(rule.get("kind") or "") == "avoid_topic":
                summaries.append(f"avoid_topic: {target_desc or topic_desc}는 다시 꺼내거나 상세 재언급하지 말 것")
            else:
                summaries.append(
                    f"do_not_store_sensitive: {target_desc or topic_desc}는 민감 세부를 반복하지 말고 필요하면 고수준으로만 답할 것"
                )
        return summaries

    def _resolve_boundary_target(self, subject_id: str, text: str) -> Dict[str, Any]:
        normalized = self._normalize_boundary_text(text)
        target_surface = self._extract_target_surface(normalized)
        inventory = self._build_boundary_target_inventory(subject_id)

        match_source = target_surface or normalized
        matched_entity_ids = QueryPlanner._match_entities_by_name(match_source, inventory)
        target_entity_id = matched_entity_ids[0] if matched_entity_ids else ""

        requested_roles = QueryPlanner._detect_requested_roles(match_source)
        if not target_entity_id and requested_roles:
            role_match = QueryPlanner._resolve_by_role_aliases(subject_id, requested_roles, inventory)
            if role_match:
                target_entity_id = str(role_match.get("entity_id") or "")

        target_entry = inventory.get(target_entity_id, {})
        target_aliases = list(target_entry.get("names", []))
        target_roles = list(target_entry.get("roles", []))
        if requested_roles:
            target_roles.extend(requested_roles)
            target_roles = list(dict.fromkeys(target_roles))

        should_scope = bool(
            target_entity_id
            or requested_roles
            or (target_surface and self._looks_like_specific_target(target_surface))
        )
        fingerprint_source = target_surface or (target_aliases[0] if target_aliases else "")
        alias_hashes = []
        if should_scope:
            alias_hashes = [
                self._hash_boundary_term(alias)
                for alias in list(dict.fromkeys(target_aliases + ([target_surface] if target_surface else [])))
                if alias
            ]
        return {
            "fingerprint": self._hash_boundary_term(fingerprint_source) if should_scope else "",
            "entity_id": target_entity_id,
            "aliases": target_aliases,
            "alias_hashes": list(dict.fromkeys(hash_value for hash_value in alias_hashes if hash_value)),
            "roles": target_roles,
        }

    def _extract_target_surface(self, normalized_text: str) -> str:
        stripped = re.sub(
            r"(저장하지 마|기억하지 마|다시 꺼내지 마|그 얘기 꺼내지 마)",
            " ",
            normalized_text,
        )
        stripped = re.sub(r"(얘기|이야기|내용|주제|부분|건|거)(?:는|은|이|가|를|을|도|만)?", " ", stripped)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if not stripped:
            return ""
        first = re.split(r"\b(?:그리고|근데|하지만|다만|또)\b", stripped)[0]
        first = re.sub(r"\b(?:내|저|나)\b", " ", first)
        tokens: List[str] = []
        for token in re.findall(r"[a-z0-9가-힣_]+", first):
            token = re.sub(r"(은|는|이|가|을|를|에|도|만|과|와|의|로|으로)$", "", token).strip()
            if not token or token in {"얘기", "이야기", "내용", "주제", "부분", "건", "거"}:
                continue
            tokens.append(token)
        return " ".join(tokens[:4]).strip()

    def _looks_like_specific_target(self, target_surface: str) -> bool:
        normalized = self._normalize_boundary_text(target_surface)
        if not normalized or normalized in {"사람", "얘기", "이야기", "내용", "주제"}:
            return False
        topic_terms = {
            self._normalize_boundary_text(term)
            for terms in self.BOUNDARY_TOPIC_TERMS.values()
            for term in terms
        }
        return normalized not in topic_terms

    def _build_boundary_target_inventory(self, subject_id: str) -> Dict[str, Dict[str, Any]]:
        inventory: Dict[str, Dict[str, Any]] = {}
        for node in self.graph.get_all_nodes():
            user_id = str(getattr(node, "user_id", "") or "")
            if not user_id:
                continue
            entry = inventory.setdefault(
                user_id,
                {"entity_id": user_id, "names": [], "roles": [], "last_seen": 0.0},
            )
            nickname = str(getattr(node, "nickname", "") or "").strip()
            if nickname:
                entry["names"].append(nickname)
            for historical in getattr(node, "nickname_history", []) or []:
                historical_name = str(historical or "").strip()
                if historical_name:
                    entry["names"].append(historical_name)

        if self.store and subject_id:
            relation_claims = self.store.get_active_claims(
                subject_id=subject_id,
                facets=[Facet.RELATION_TO_ENTITY.value],
                viewer_id=subject_id,
                limit=200,
            )
            for claim in relation_claims:
                target_entity_id = str(claim.value.get("target_entity_id") or "")
                if not target_entity_id:
                    continue
                entry = inventory.setdefault(
                    target_entity_id,
                    {"entity_id": target_entity_id, "names": [], "roles": [], "last_seen": 0.0},
                )
                relation_kind = claim.value.get("relation_kind")
                if relation_kind:
                    entry["roles"].extend(QueryPlanner.expand_role_aliases(relation_kind))

        for entry in inventory.values():
            entry["names"] = list(dict.fromkeys(name for name in entry["names"] if name))
            entry["roles"] = list(dict.fromkeys(role for role in entry["roles"] if role))
        return inventory

    def _target_aliases_for_entity(self, entity_id: str) -> List[str]:
        if not entity_id:
            return []
        return list(self._build_boundary_target_inventory("").get(entity_id, {}).get("names", []))

    def _boundary_target_description(self, rule: Dict[str, Any]) -> str:
        aliases = [alias for alias in (rule.get("target_aliases") or []) if alias]
        if aliases:
            return aliases[0]
        roles = [role for role in (rule.get("target_roles") or []) if role and role != "person"]
        if roles:
            return roles[0]
        return ""

    def _match_rule_target_scope(self, normalized_text: str, rule: Dict[str, Any]) -> bool:
        if not self._rule_has_target_scope(rule):
            return True

        aliases = [
            self._normalize_boundary_text(alias)
            for alias in (rule.get("target_aliases") or [])
            if alias
        ]
        if aliases and any(alias and alias in normalized_text for alias in aliases):
            return True

        roles = [
            self._normalize_boundary_text(role)
            for role in (rule.get("target_roles") or [])
            if role and role != "person"
        ]
        if roles and any(role and role in normalized_text for role in roles):
            return True

        candidate_hashes = self._candidate_boundary_phrase_hashes(normalized_text)
        target_hashes = {str(rule.get("target") or "")}
        target_hashes.update(
            str(alias_hash)
            for alias_hash in (rule.get("target_alias_hashes") or [])
            if alias_hash
        )
        target_hashes.discard("")
        return bool(target_hashes.intersection(candidate_hashes))

    def _rule_has_target_scope(self, rule: Dict[str, Any]) -> bool:
        return bool(
            rule.get("target_entity_id")
            or rule.get("target")
            or rule.get("target_aliases")
            or rule.get("target_alias_hashes")
            or rule.get("target_roles")
        )

    def _candidate_boundary_phrase_hashes(self, normalized_text: str) -> set[str]:
        tokens = [token for token in re.findall(r"[a-z0-9가-힣_]+", normalized_text) if token]
        hashes: set[str] = set()
        max_ngram = min(len(tokens), 4)
        for size in range(1, max_ngram + 1):
            for index in range(0, len(tokens) - size + 1):
                hash_value = self._hash_boundary_term(" ".join(tokens[index:index + size]))
                if hash_value:
                    hashes.add(hash_value)
        return hashes

    def _semantic_boundary_match(self, normalized_text: str, rule: Dict[str, Any],
                                 segment_embeddings: Optional[Dict[str, List[float]]] = None) -> bool:
        if not self.api or not normalized_text:
            return False

        semantic_text = str(rule.get("semantic_text") or "").strip()
        if not semantic_text:
            return False

        semantic_cache_key = f"{normalized_text}::{semantic_text}"
        cached_result = self._cache_lookup(self._semantic_match_cache, semantic_cache_key)
        if cached_result is not None:
            return bool(cached_result)

        cache_key = normalized_text
        segment_embeddings = segment_embeddings if segment_embeddings is not None else {}
        segment_embedding = segment_embeddings.get(cache_key)
        if segment_embedding is None:
            segment_embedding = self._cache_lookup(self._segment_embedding_cache, cache_key)
        if segment_embedding is None:
            segment_embedding = self.api.get_embedding(normalized_text)
            self._cache_store(
                self._segment_embedding_cache,
                cache_key,
                segment_embedding,
                self._segment_embedding_cache_max,
            )
        if segment_embeddings.get(cache_key) is None:
            segment_embeddings[cache_key] = segment_embedding

        rule_embedding = self._boundary_embedding_cache.get(semantic_text)
        if rule_embedding is None:
            rule_embedding = self.api.get_embedding(semantic_text)
            self._boundary_embedding_cache[semantic_text] = rule_embedding

        similarity = self._cosine_similarity(segment_embedding, rule_embedding)
        matched = similarity >= self._semantic_threshold_for_rule(rule)
        self._cache_store(
            self._semantic_match_cache,
            semantic_cache_key,
            matched,
            self._semantic_result_cache_max,
        )
        return matched

    def _semantic_threshold_for_rule(self, rule: Dict[str, Any]) -> float:
        topic_label = str(rule.get("topic_label") or "personal").strip().lower()
        if topic_label == "personal" and not rule.get("sensitive_tokens"):
            return max(self._semantic_threshold, 0.86)
        return self._semantic_threshold

    def _cosine_similarity(self, vec_a: Optional[List[float]], vec_b: Optional[List[float]]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for value_a, value_b in zip(vec_a, vec_b):
            dot += value_a * value_b
            norm_a += value_a * value_a
            norm_b += value_b * value_b
        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0
        return dot / math.sqrt(norm_a * norm_b)

    def _cache_lookup(self, cache: Dict[str, Any], key: str):
        if key not in cache:
            return None
        value = cache.pop(key)
        cache[key] = value
        return value

    def _cache_store(self, cache: Dict[str, Any], key: str, value: Any, max_entries: int):
        if max_entries <= 0:
            return
        if key in cache:
            cache.pop(key, None)
        cache[key] = value
        while len(cache) > max_entries:
            oldest_key = next(iter(cache))
            cache.pop(oldest_key, None)

    def _save_claim(self, subject_id: str, facet: str, value: dict, qualifiers: dict,
                    nl_summary: str, confidence: float,
                    sensitivity: Optional[str] = None):
        claim = self.graph.upsert_claim(
            subject_id=subject_id,
            facet=facet,
            value=value,
            qualifiers=qualifiers,
            nl_summary=nl_summary,
            source_type="explicit",
            confidence=confidence,
            status="active",
            last_confirmed_at=time.time(),
            sensitivity=sensitivity,
        )
        self.store.upsert_claim(claim)
        return claim
