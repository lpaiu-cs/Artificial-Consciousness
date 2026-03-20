import numpy as np
import time
from typing import Dict, Any
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
    감정 벡터 연산을 통해 관계(Affinity)를 동적으로 업데이트합니다.
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
        [Vector Math]
        (현재 기분 벡터) vs (긍정 앵커 벡터) 유사도 계산 -> 호감도 업데이트
        """
        if not current_emotion_vec or not self.positive_anchor_vec:
            return

        # 1. 코사인 유사도 계산
        similarity = self._cosine_similarity(current_emotion_vec, self.positive_anchor_vec)
        
        # 2. 변화량 계산 (Config 스케일링 적용)
        # Sim: 1.0 -> +5점, 0.0 -> 0점, -1.0 -> -5점
        delta = similarity * config.SOCIAL_SENSITIVITY
        
        # 3. 그래프 업데이트
        self.graph.update_affinity(user_id, delta)
        if self.store:
            self.store.update_relation_state(
                user_id,
                {
                    "trust": similarity * 0.04,
                    "warmth": similarity * 0.06,
                    "familiarity": 0.03,
                    "respect": similarity * 0.02,
                    "tension": similarity * -0.05,
                    "reliability": similarity * 0.03,
                },
            )
        
        # Debug
        # print(f"❤️ [Social Update] Sim: {similarity:.2f} -> Delta: {delta:+.2f}")

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
