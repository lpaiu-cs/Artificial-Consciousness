import hashlib
import json
import math
import re
import time
from typing import Any, Dict, List, Optional

import config

from memory.canonical_store import CanonicalMemoryStore
from memory.ontology import Facet
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
        self._semantic_threshold = float(getattr(config, "BOUNDARY_SEMANTIC_MATCH_THRESHOLD", 0.78))

    def apply_write_barriers(self, log: Optional[Dict], persist: bool = True) -> Optional[Dict]:
        if not log:
            return log

        role = log.get("role", "user")
        text = (log.get("msg") or "").strip()
        if role != "user" or not text:
            return dict(log)

        sanitized_text, barriers = self._redact_boundary_segments(text)
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
        barriers = self._detect_boundary_rules(mem.content)
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

    def _detect_boundary_rules(self, text: str) -> List[Dict[str, str]]:
        rules: List[Dict[str, str]] = []
        topic_label = self._classify_boundary_topic(text)
        target = self._boundary_target_hash(text)
        sensitive_tokens = self._extract_sensitive_tokens(text)
        if "저장하지 마" in text or "기억하지 마" in text:
            rules.append({
                "kind": "do_not_store_sensitive",
                "summary": f"{topic_label} 관련 민감 주제를 저장하지 말라는 경계 요청이 있음",
                "topic_label": topic_label,
                "target": target,
                "sensitive_tokens": sensitive_tokens,
            })
        if "다시 꺼내지 마" in text or "그 얘기 꺼내지 마" in text:
            rules.append({
                "kind": "avoid_topic",
                "summary": f"{topic_label} 관련 주제를 다시 꺼내지 말라는 경계 요청이 있음",
                "topic_label": topic_label,
                "target": target,
                "sensitive_tokens": sensitive_tokens,
            })
        return rules

    def _redact_boundary_segments(self, text: str) -> tuple[str, List[Dict[str, str]]]:
        segments = self._segment_text(text)
        if not segments:
            return text, []

        sanitized_segments: List[str] = []
        all_barriers: List[Dict[str, str]] = []
        for segment in segments:
            barriers = self._detect_boundary_rules(segment)
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

    def _save_boundary_claims(self, subject_id: str, barriers: List[Dict[str, str]]):
        for barrier in barriers:
            self._save_claim(
                subject_id=subject_id,
                facet=Facet.BOUNDARY_RULE.value,
                value={
                    "kind": barrier["kind"],
                    "target": barrier["target"],
                    "policy_kind": barrier["kind"],
                },
                qualifiers={
                    "topic_label": barrier["topic_label"],
                    "semantic_terms": self._topic_semantic_terms(barrier["topic_label"]),
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

    def sanitize_assistant_memory(self, text: str, boundary_rules: Optional[List[Dict[str, Any]]] = None) -> tuple[str, bool]:
        boundary_rules = boundary_rules or []
        if not text or not boundary_rules:
            return text, False

        segments = self._segment_text(text)
        if not segments:
            return text, False

        sanitized_segments: List[str] = []
        redacted = False
        segment_embeddings: Dict[str, List[float]] = {}
        for segment in segments:
            matched_rules = self._match_boundary_rules(segment, boundary_rules, segment_embeddings)
            if matched_rules:
                sanitized_segments.append(self._build_boundary_placeholder(matched_rules))
                redacted = True
            else:
                sanitized_segments.append(segment)

        sanitized_text = " ".join(segment for segment in sanitized_segments if segment).strip()
        return sanitized_text or text, not redacted

    def _build_boundary_placeholder(self, barriers: List[Dict[str, Any]]) -> str:
        kinds = {barrier["kind"] for barrier in barriers}
        if "do_not_store_sensitive" in kinds and "avoid_topic" in kinds:
            return "경계 요청이 있었다. 민감한 내용 저장과 재언급을 피해야 한다."
        if "do_not_store_sensitive" in kinds:
            return "경계 요청이 있었다. 민감한 내용은 장기 기억에 저장하지 않아야 한다."
        return "경계 요청이 있었다. 특정 주제는 다시 꺼내지 않아야 한다."

    def _make_processed_barrier_key(self, user_id: str, timestamp: float,
                                    barriers: List[Dict[str, str]]) -> str:
        payload = json.dumps(
            {
                "user_id": user_id,
                "timestamp": timestamp,
                "barriers": sorted(
                    [
                        {
                            "kind": barrier["kind"],
                            "target": barrier["target"],
                        }
                        for barrier in barriers
                    ],
                    key=lambda item: (item["kind"], item["target"]),
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

    def _boundary_target_hash(self, text: str) -> str:
        normalized = self._normalize_boundary_text(text)
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
            "sensitive_tokens": list(claim.qualifiers.get("sensitive_tokens") or []),
            "semantic_terms": list(claim.qualifiers.get("semantic_terms") or []),
        }
        return self._to_runtime_boundary_rule(barrier)

    def _to_runtime_boundary_rule(self, barrier: Dict[str, Any]) -> Dict[str, Any]:
        topic_label = str(barrier.get("topic_label") or "personal")
        semantic_terms = list(dict.fromkeys(
            list(barrier.get("semantic_terms") or []) + self._topic_semantic_terms(topic_label)
        ))
        rule = {
            "kind": barrier["kind"],
            "topic_label": topic_label,
            "target": barrier.get("target") or "",
            "sensitive_tokens": list(dict.fromkeys(barrier.get("sensitive_tokens") or [])),
            "semantic_terms": semantic_terms,
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
        ]
        return " ".join(part for part in parts if part).strip()

    def _match_boundary_rules(self, text: str, boundary_rules: List[Dict[str, Any]],
                              segment_embeddings: Optional[Dict[str, List[float]]] = None) -> List[Dict[str, Any]]:
        normalized = self._normalize_boundary_text(text)
        matched: List[Dict[str, Any]] = []
        for rule in boundary_rules:
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
            if self._semantic_boundary_match(normalized, rule, segment_embeddings):
                matched.append(rule)
        return matched

    def _semantic_boundary_match(self, normalized_text: str, rule: Dict[str, Any],
                                 segment_embeddings: Optional[Dict[str, List[float]]] = None) -> bool:
        if not self.api or not normalized_text:
            return False

        semantic_text = str(rule.get("semantic_text") or "").strip()
        if not semantic_text:
            return False

        cache_key = normalized_text
        segment_embeddings = segment_embeddings if segment_embeddings is not None else {}
        segment_embedding = segment_embeddings.get(cache_key)
        if segment_embedding is None:
            segment_embedding = self.api.get_embedding(normalized_text)
            segment_embeddings[cache_key] = segment_embedding

        rule_embedding = self._boundary_embedding_cache.get(semantic_text)
        if rule_embedding is None:
            rule_embedding = self.api.get_embedding(semantic_text)
            self._boundary_embedding_cache[semantic_text] = rule_embedding

        similarity = self._cosine_similarity(segment_embedding, rule_embedding)
        return similarity >= self._semantic_threshold_for_rule(rule)

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
