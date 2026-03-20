import re
import time
from typing import List, Optional

from memory.canonical_store import CanonicalMemoryStore
from memory.ontology import Facet
from memory_structures import MemoryObject
from modules.ltm_graph import MemoryGraph


class FastPathMemoryWriter:
    """Rule-based write path for high-value explicit state."""

    def __init__(self, graph: MemoryGraph, canonical_store: CanonicalMemoryStore):
        self.graph = graph
        self.store = canonical_store

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
        text = mem.content
        if "저장하지 마" in text or "기억하지 마" in text:
            self._save_claim(
                subject_id=mem.user_id,
                facet=Facet.BOUNDARY_RULE.value,
                value={"kind": "do_not_store_sensitive", "rule": text},
                qualifiers={},
                nl_summary="민감한 내용을 저장하지 말라는 경계 요청이 있음",
                confidence=0.98,
                sensitivity="high",
            )

        if "다시 꺼내지 마" in text or "그 얘기 꺼내지 마" in text:
            self._save_claim(
                subject_id=mem.user_id,
                facet=Facet.BOUNDARY_RULE.value,
                value={"kind": "avoid_topic", "rule": text},
                qualifiers={},
                nl_summary="특정 주제를 다시 꺼내지 말라는 경계 요청이 있음",
                confidence=0.98,
                sensitivity="high",
            )

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
            claim = self._save_claim(
                subject_id=mem.user_id,
                facet=Facet.COMMITMENT_OPEN_LOOP.value,
                value={"kind": "followup_needed", "text": text, "priority": 8},
                qualifiers={},
                nl_summary="후속 안내나 재개가 필요한 열린 루프가 있음",
                confidence=0.92,
            )
            self.store.upsert_open_loop_from_claim(claim)

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
