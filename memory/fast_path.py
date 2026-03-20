import hashlib
import re
import time
from typing import Dict, List, Optional

from memory.canonical_store import CanonicalMemoryStore
from memory.ontology import Facet
from memory_structures import MemoryObject
from modules.ltm_graph import MemoryGraph


class FastPathMemoryWriter:
    """Rule-based write path for high-value explicit state."""

    def __init__(self, graph: MemoryGraph, canonical_store: CanonicalMemoryStore):
        self.graph = graph
        self.store = canonical_store
        self._processed_barrier_keys = set()

    def apply_write_barriers(self, log: Optional[Dict]) -> Optional[Dict]:
        if not log:
            return log

        role = log.get("role", "user")
        text = (log.get("msg") or "").strip()
        if role != "user" or not text:
            return dict(log)

        barriers = self._detect_boundary_rules(text)
        if not barriers:
            return dict(log)

        key = (
            str(log.get("user_id", "")),
            log.get("timestamp", 0.0),
            text,
        )
        if key not in self._processed_barrier_keys:
            self._processed_barrier_keys.add(key)
            self._save_boundary_claims(str(log.get("user_id", "")), barriers)

        sanitized = dict(log)
        sanitized["msg"] = self._build_boundary_placeholder(barriers)
        sanitized["memory_redacted"] = True
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
        if "저장하지 마" in text or "기억하지 마" in text:
            rules.append({
                "kind": "do_not_store_sensitive",
                "summary": f"{topic_label} 관련 민감 주제를 저장하지 말라는 경계 요청이 있음",
                "topic_label": topic_label,
                "target": target,
            })
        if "다시 꺼내지 마" in text or "그 얘기 꺼내지 마" in text:
            rules.append({
                "kind": "avoid_topic",
                "summary": f"{topic_label} 관련 주제를 다시 꺼내지 말라는 경계 요청이 있음",
                "topic_label": topic_label,
                "target": target,
            })
        return rules

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
                },
                nl_summary=barrier["summary"],
                confidence=0.98,
                sensitivity="high",
            )

    def _build_boundary_placeholder(self, barriers: List[Dict[str, str]]) -> str:
        kinds = {barrier["kind"] for barrier in barriers}
        if "do_not_store_sensitive" in kinds and "avoid_topic" in kinds:
            return "경계 요청이 있었다. 민감한 내용 저장과 재언급을 피해야 한다."
        if "do_not_store_sensitive" in kinds:
            return "경계 요청이 있었다. 민감한 내용은 장기 기억에 저장하지 않아야 한다."
        return "경계 요청이 있었다. 특정 주제는 다시 꺼내지 않아야 한다."

    def _classify_boundary_topic(self, text: str) -> str:
        normalized = self._normalize_boundary_text(text)
        topic_keywords = {
            "health": ["건강", "병원", "아프", "치료", "우울", "불안", "진단", "약", "health"],
            "finance": ["돈", "예산", "카드", "대출", "월세", "연봉", "주식", "빚", "finance"],
            "relationship": ["연애", "가족", "친구", "부모", "형제", "교수", "동료", "관계", "relationship"],
            "work_or_study": ["회사", "직장", "업무", "프로젝트", "학교", "시험", "과제", "연구", "work"],
            "privacy": ["비밀", "개인", "사생활", "프라이버시", "privacy"],
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
