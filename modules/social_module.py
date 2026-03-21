import re
import numpy as np
import time
from typing import Dict, Any, Optional
import config
from memory.canonical_store import CanonicalMemoryStore
from memory.ontology import Facet
from memory_structures import RelationState
from modules.ltm_graph import MemoryGraph
from api_client import UnifiedAPIClient

class SocialManager:
    """
    [Social Brain]
    유저의 정체성(Identity)을 추적하고, 
    감정 벡터와 상호작용 사건 신호를 통해 관계 상태를 동적으로 업데이트합니다.
    닉네임 사칭(Impersonation)을 방지하는 보안 로직이 포함됩니다.
    """
    def __init__(self, ltm_graph: MemoryGraph, api_client: UnifiedAPIClient,
                 canonical_store: CanonicalMemoryStore = None):
        self.graph = ltm_graph
        self.api = api_client
        self.store = canonical_store
        
        # [Social Logic] 긍정 기준점 벡터 미리 계산 (캐싱)
        # 매번 API를 호출하지 않고, 봇이 켜질 때 한 번만 계산합니다.
        print(f"❤️ [Social] Loading Positive Anchor: '{config.POSITIVE_EMOTION_ANCHOR}'...")
        self.positive_anchor_vec = self.api.get_embedding(config.POSITIVE_EMOTION_ANCHOR)

        # [Security] 예약어 설정 (어뷰징 방지)
        # 이 단어들이 포함된 닉네임은 시스템이 강제로 태깅합니다.
        self.reserved_nicknames = {f"{config.BOT_NAME}", "Bot", "Admin", "System", "관리자", "개발자", "운영자",
                                   "Developer", "Administrator", "Moderator", "me", "나"}

    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """
        LLM 프롬프트용 유저 정보 반환
        Return: { "nickname": "민초단장", "desc": "Friend (85.0)" }
        """
        node = self.graph.get_or_create_user(user_id, "") # 닉네임 없으면 기존거 유지
        relation_state = self.store.get_relation_state(user_id) if self.store else None
        return {
            "nickname": node.nickname,
            "affinity": node.affinity,
            "desc": self._relation_to_desc(node.affinity, relation_state),
            "relation_state": relation_state,
            # "history": node.nickname_history # 옵션
        }

    def process_identity(self, user_id: str, current_nickname: str) -> bool:
        """
        [Identity Check Implementation]
        유저의 닉네임 변경 여부를 확인하고, 변경 시 이력을 남깁니다.
        + 예약어 필터링(Sanitization)
        
        Returns:
            True: 닉네임이 변경됨 (Orchestrator가 시스템 메시지를 띄워야 함)
            False: 변경 없음 또는 신규 유저
        """
        if not current_nickname: 
            return False
        
        # 1. 닉네임 어뷰징 방지 (Sanitization)
        safe_nickname = current_nickname
        for reserved in self.reserved_nicknames:
            if reserved.lower() in current_nickname.lower():
                # 예약어가 포함된 경우 태그 부착하여 혼동 방지
                safe_nickname = f"{current_nickname}(User)" 
                break

        # 2. 노드 조회 (업데이트 없이)
        node = self.graph.get_or_create_user(user_id, "")
        
        # 신규 유저 처리
        if not node.nickname:
            self.graph.get_or_create_user(user_id, safe_nickname)
            self._write_identity_claim(user_id, safe_nickname)
            return False # 첫 만남은 변경 이벤트가 아님

        # 3. 변경 감지 및 이력 기록
        if node.nickname != safe_nickname:
            # (A) 과거 닉네임을 역사(History)에 기록
            # dataclass의 field(default_factory=list) 덕분에 바로 append 가능
            node.nickname_history.append(node.nickname)
            
            # (B) 현재 닉네임 갱신
            old_name = node.nickname
            self.graph.get_or_create_user(user_id, safe_nickname)
            self._write_identity_claim(user_id, safe_nickname)
            
            # (C) 로그 출력 (선택 사항)
            print(f"🔄 [Identity] Nickname Changed: {old_name} -> {safe_nickname}")
            
            return True # 변경됨! (시스템 메시지 트리거)

        return False

    def calculate_and_update_affinity(self, user_id: str, current_emotion_vec: list):
        """
        [Legacy Wrapper]
        벡터 기반 기본 관계 업데이트. P2 이후에는 tone-only fallback으로 유지한다.
        """
        if not current_emotion_vec or not self.positive_anchor_vec:
            return

        similarity = self._cosine_similarity(current_emotion_vec, self.positive_anchor_vec)
        deltas = self._compose_relation_deltas(similarity, {})
        self._apply_relation_update(user_id, deltas)

    def update_relationship(self, user_id: str, user_text: str, assistant_text: str = "",
                            user_embedding: Optional[list] = None,
                            boundary_requested: bool = False,
                            boundary_respected: bool = False) -> Dict[str, float]:
        """
        [Event-centric Relation Update]
        톤 유사도는 약한 베이스 신호로 쓰고, 감사/불만/수정/repair/boundary respect 같은
        상호작용 사건이 각 축을 다르게 갱신한다.
        """
        similarity = 0.0
        if user_embedding and self.positive_anchor_vec:
            similarity = self._cosine_similarity(user_embedding, self.positive_anchor_vec)

        signals = self._extract_interaction_signals(
            user_text=user_text,
            assistant_text=assistant_text,
            boundary_requested=boundary_requested,
            boundary_respected=boundary_respected,
        )
        deltas = self._compose_relation_deltas(similarity, signals)
        self._apply_relation_update(user_id, deltas)
        return deltas

    def handle_open_loop_event(self, event: Dict[str, Any]) -> Dict[str, float]:
        owner_id = str(event.get("owner_id", ""))
        if not owner_id or not self._should_score_open_loop_event(event):
            return {}

        deltas = self._compose_fulfillment_deltas(event)
        if not any(abs(value) > 0.0 for value in deltas.values()):
            return {}

        self._apply_relation_update(owner_id, deltas)
        return deltas

    def _score_to_desc(self, score: float) -> str:
        """점수를 자연어 관계 설명으로 변환"""
        if score >= 90: return f"Soulmate ({score:.1f})"
        if score >= 70: return f"Close Friend ({score:.1f})"
        if score >= 40: return f"Acquaintance ({score:.1f})"
        if score >= 20: return f"Awkward ({score:.1f})"
        return f"Hostile ({score:.1f})"

    def _relation_to_desc(self, affinity: float, relation_state: RelationState = None) -> str:
        if not relation_state:
            return self._score_to_desc(affinity)
        return (
            f"trust={relation_state.trust:.2f}, warmth={relation_state.warmth:.2f}, "
            f"familiarity={relation_state.familiarity:.2f}, respect={relation_state.respect:.2f}, "
            f"tension={relation_state.tension:.2f}, reliability={relation_state.reliability:.2f}"
        )

    def _write_identity_claim(self, user_id: str, preferred_name: str):
        if not self.store or not preferred_name:
            return
        claim = self.graph.upsert_claim(
            subject_id=user_id,
            facet=Facet.IDENTITY_PREFERRED_NAME.value,
            value={"name": preferred_name},
            qualifiers={},
            nl_summary=f"선호 호칭은 {preferred_name}임",
            source_type="explicit",
            confidence=1.0,
            status="active",
            last_confirmed_at=time.time(),
        )
        self.store.upsert_claim(claim)

    def _cosine_similarity(self, vec_a, vec_b) -> float:
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0: return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _extract_interaction_signals(self, user_text: str, assistant_text: str,
                                     boundary_requested: bool,
                                     boundary_respected: bool) -> Dict[str, float]:
        user = (user_text or "").lower()
        assistant = (assistant_text or "").lower()

        complaint = self._contains_any(
            user,
            ["별로", "실망", "불편", "짜증", "화나", "문제", "심하다", "이상해", "wrong", "bad", "not helpful"],
        )
        correction = self._contains_any(
            user,
            ["틀렸", "아니야", "수정", "정정", "다시 해", "고쳐", "아닌데", "정확하지", "잘못"],
        )
        gratitude = self._contains_any(
            user,
            ["고마워", "고맙", "감사", "덕분", "thanks", "thx", "thank you"],
        )
        praise = self._contains_any(
            user,
            ["좋았", "좋네", "잘했", "최고", "도움됐", "helpful", "great", "nice"],
        )
        trust = self._contains_any(
            user,
            ["믿", "맡길게", "의지", "trust you", "count on"],
        )
        assistant_repair = self._contains_any(
            assistant,
            ["미안", "죄송", "사과", "정정", "수정", "바로잡", "다시 설명", "다시 정리"],
        )

        return {
            "complaint": float(complaint),
            "correction": float(correction),
            "gratitude": float(gratitude),
            "praise": float(praise),
            "trust": float(trust),
            "boundary_requested": float(boundary_requested),
            "boundary_respected": float(boundary_respected),
            "repair": float(assistant_repair and (complaint or correction)),
        }

    def _compose_relation_deltas(self, similarity: float,
                                 signals: Dict[str, float]) -> Dict[str, float]:
        positive_tone = max(similarity, 0.0)
        negative_tone = max(-similarity, 0.0)
        gratitude = signals.get("gratitude", 0.0)
        praise = signals.get("praise", 0.0)
        complaint = signals.get("complaint", 0.0)
        correction = signals.get("correction", 0.0)
        trust_signal = signals.get("trust", 0.0)
        repair = signals.get("repair", 0.0)
        boundary_respected = signals.get("boundary_respected", 0.0)

        return {
            "affinity": (
                similarity * config.SOCIAL_SENSITIVITY
                + gratitude * 1.4
                + praise * 1.0
                + trust_signal * 0.8
                + repair * 0.7
                + boundary_respected * 0.5
                - complaint * 1.3
                - correction * 0.8
            ),
            "warmth": (
                positive_tone * 0.03
                - negative_tone * 0.015
                + gratitude * 0.08
                + praise * 0.05
                - complaint * 0.04
            ),
            "trust": (
                gratitude * 0.03
                + trust_signal * 0.05
                + boundary_respected * 0.08
                + repair * 0.05
                - complaint * 0.03
                - correction * 0.02
            ),
            "familiarity": (
                0.025
                + gratitude * 0.01
                + praise * 0.01
            ),
            "respect": (
                praise * 0.03
                + boundary_respected * 0.06
                + repair * 0.05
                - complaint * 0.015
            ),
            "tension": (
                negative_tone * 0.02
                + complaint * 0.08
                + correction * 0.05
                - repair * 0.08
                - gratitude * 0.02
            ),
            "reliability": (
                boundary_respected * 0.05
                + repair * 0.05
                - complaint * 0.03
                - correction * 0.04
            ),
        }

    def _should_score_open_loop_event(self, event: Dict[str, Any]) -> bool:
        kind = str(event.get("kind") or "").strip().lower()
        source_type = str(event.get("source_type") or "").strip().lower()
        return source_type == "assistant_commitment" or kind in {"assistant_promise", "followup_needed"}

    def _compose_fulfillment_deltas(self, event: Dict[str, Any]) -> Dict[str, float]:
        event_type = str(event.get("event_type") or "").strip().lower()
        terminal_status = str(event.get("terminal_status") or "").strip().lower()
        due_at = event.get("due_at")
        occurred_at = float(event.get("occurred_at") or time.time())
        was_overdue = bool(event.get("was_overdue"))
        if due_at is not None:
            try:
                was_overdue = was_overdue or float(due_at) < occurred_at
            except (TypeError, ValueError):
                pass

        if event_type == "overdue":
            return {
                "affinity": -1.0,
                "trust": -0.03,
                "respect": -0.02,
                "tension": 0.03,
                "reliability": -0.06,
            }

        if event_type != "closed":
            return {}

        if terminal_status == "done":
            if was_overdue:
                return {
                    "affinity": 0.3,
                    "trust": 0.01,
                    "respect": 0.0,
                    "tension": -0.01,
                    "reliability": 0.02,
                }
            return {
                "affinity": 1.0,
                "trust": 0.04,
                "respect": 0.02,
                "tension": -0.02,
                "reliability": 0.08,
            }

        if terminal_status == "abandoned":
            return {
                "affinity": -1.2,
                "trust": -0.05,
                "respect": -0.03,
                "tension": 0.04,
                "reliability": -0.08,
            }

        return {}

    def _apply_relation_update(self, user_id: str, deltas: Dict[str, float]):
        affinity_delta = deltas.get("affinity", 0.0)
        self.graph.update_affinity(user_id, affinity_delta)
        if self.store:
            relation_deltas = {key: value for key, value in deltas.items() if key != "affinity"}
            self.store.update_relation_state(user_id, relation_deltas)

    def _contains_any(self, text: str, tokens: list[str]) -> bool:
        normalized = re.sub(r"\s+", " ", text or "")
        return any(token in normalized for token in tokens)
