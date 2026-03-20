import re
import time
from typing import List, Dict, Tuple, Any

# Config & Structs
import config
from memory.canonical_store import CanonicalMemoryStore
from memory.fast_path import FastPathMemoryWriter
from memory.schema import ContextBundle
from memory_structures import ClaimNode, RetrievalQuery, RelationState, MemoryObject, EpisodeNode, InsightNode, NoteNode

# Modules
from api_client import UnifiedAPIClient
from modules.sensory_system import SensorySystem
from modules.stm_handler import WorkingMemory
from modules.ltm_graph import MemoryGraph
from modules.ltm_handler import LongTermMemory
from modules.reflection_handler import ReflectionHandler
from modules.social_module import SocialManager

'''
TODO:
구조 리팩토링 -> 네이밍과 모듈에 함수 재배치 정리. (done.)
user_id(고윳값)과 nickname(변수) 매핑 시스템 구축. (done?~)
기 저장된 nodes에 대해 내용 수정 기능 추가. (done?)
기억(node) 연결과 갱신에 대한 로직 강화. (done?)


페르소나 관리 시스템 구상:
나라는 페르소나에 대한 정의 또한 메모리 로직으로 관리할 수 있지 않을까?

after work:
엔트리포인트 설정해야함.
'''

class BotOrchestrator:
    """
    [The Ego: Central Controller]
    인지 과정(Perception -> Retrieval -> Cognition -> Action)을 조율합니다.
    """
    
    def __init__(self, api_client: UnifiedAPIClient = None,
                 graph_db: MemoryGraph = None,
                 canonical_store: CanonicalMemoryStore = None):
        # 1. Infrastructure
        self.api = api_client or UnifiedAPIClient()
        self.ltm_graph = graph_db or MemoryGraph() # Thread-Safe Graph DB
        self.canonical_store = canonical_store or CanonicalMemoryStore()
        
        # 2. Functional Modules
        self.social = SocialManager(self.ltm_graph, self.api, self.canonical_store) # Social Brain
        self.sensory = SensorySystem(self.api)                # Eyes & Ears
        self.stm = WorkingMemory()                            # Short-term Memory
        self.ltm = LongTermMemory(self.ltm_graph, self.api, self.canonical_store)   # Long-term Memory Retrieval
        self.fast_path_writer = FastPathMemoryWriter(self.ltm_graph, self.canonical_store)
        
        # 3. Background Process
        self.reflector = ReflectionHandler(self.ltm_graph, self.api, self.canonical_store)
        self.reflector.start_background_loop(
            self.stm.eviction_buffer, interval=config.REFLECTION_INTERVAL
        )
        
        # 4. State
        self.current_mood = "calm"
        self.session_state = {"current_mood": "calm"}
        # config에 봇 ID가 있으면 가져옴
        self.bot_id = str(getattr(config, 'BOT_USER_ID', ''))

    def process_trigger(self, history: List[Dict], calling_message: Dict) -> str:
        """
        [Main Cognitive Loop]
        외부에서 호출되는 진입점입니다.
        """
        user_id = str(calling_message.get("user_id"))
        raw_user_input = calling_message.get("msg", "")
        sanitized_history = [self.fast_path_writer.apply_write_barriers(log) for log in history]
        sanitized_message = self.fast_path_writer.apply_write_barriers(calling_message)
        
        # -------------------------------------------------------
        # Phase 1: Perception (지각)
        # -------------------------------------------------------
        # 입력 로그를 청크로 변환하고, 닉네임 변경 등을 감지합니다.
        chunked_memories = self._perceive(sanitized_history, sanitized_message)
        if not chunked_memories:
            chunked_memories = self._build_fallback_memories(sanitized_message)
        if not chunked_memories:
            return ""
        self.stm.inject_memories(chunked_memories) # STM 주입 (Inject)
        
        # -------------------------------------------------------
        # Phase 2: Retrieval & Attention (기억 인출 및 집중)
        # -------------------------------------------------------
        # 현재 대화의 마지막 발화를 쿼리로 사용
        last_chunk = chunked_memories[-1]
        query_text = last_chunk.content
        
        # LTM 검색 (3-Tier Graph Search)
        retrieved_nodes = self._retrieve_and_attend(query_text, user_id)
        
        # -------------------------------------------------------
        # Phase 3: Cognition (사고 및 맥락 구성)
        # -------------------------------------------------------
        # STM과 LTM을 조합하여 LLM이 이해할 수 있는 텍스트로 변환 (ID -> Nickname 치환)
        context_summary = self._think(retrieved_nodes, user_id)
        
        # 유저와의 관계 정보 가져오기 (LLM 페르소나 조절용)
        user_ctx = self.social.get_user_context(user_id)
        relationship_desc = user_ctx["desc"] # 예: "Close Friend (85.0)"
        
        # -------------------------------------------------------
        # Phase 4: Action (행동 및 학습)
        # -------------------------------------------------------
        response = self._act(user_id, raw_user_input, context_summary, relationship_desc)
        
        return response

    # =========================================================================
    # Internal Pipelines
    # =========================================================================

    def _perceive(self, history, current_msg) -> List[MemoryObject]:
        """
        [Sensory Processing]
        Raw Data -> Memory Objects
        """
        # 1. 청킹 및 임베딩 생성 (Sensory System 위임)
        chunks = self.sensory.process_input(history, current_msg)
        
        # 2. 정체성(Identity) 확인 (Social Manager 위임)
        user_id = str(current_msg.get("user_id"))
        nickname = current_msg.get("user_name", "Unknown")
        
        # 닉네임이 바뀌었다면 그래프 갱신 및 시스템 알림 생성 여부 판단
        # (현재 구현상 process_identity는 False만 리턴하지만, 추후 확장 시 여기서 시스템 메시지 추가 가능)
        self.social.process_identity(user_id, nickname)
        self.fast_path_writer.process(chunks)
        
        return chunks
    
    def _retrieve_and_attend(self, query_text: str, user_id: str):
        """
        [Memory Retrieval]
        Vector Search -> Graph Spreading -> Semantic Attention
        """
        # 1. Query 생성
        query_embedding = self.api.get_embedding(query_text)
        query = RetrievalQuery(
            embedding=query_embedding,
            user_id=user_id,
            keywords=query_text.split(), # 키워드도 여전히 보조적으로 사용
            query_text=query_text,
            current_mood=self.current_mood
        )
        
        # 2. LTM 검색 (LTM Handler 위임)
        context_bundle = self.ltm.build_context_bundle(query, top_k=3)
        
        # 3. STM Attention (Vector-based)
        # 검색된 내용이 아니라 '현재 쿼리'에 집중하도록 STM 활성도 갱신
        self.stm.update_activations(query_embedding)
        
        return context_bundle

    def _build_fallback_memories(self, current_msg: Dict) -> List[MemoryObject]:
        """
        Delta ingest 결과가 비어도 현재 턴을 처리하기 위한 최소 MemoryObject를 생성한다.
        이미 본 로그의 재저장은 피하고, 현재 응답 생성만 안전하게 이어간다.
        """
        if not current_msg:
            return []

        content = (current_msg.get("msg") or "").strip()
        if not content:
            return []

        user_id = str(current_msg.get("user_id", "unknown"))
        is_assistant = (
            current_msg.get("role") == "assistant"
            or user_id in {"bot", "assistant"}
            or (self.bot_id and user_id == self.bot_id)
        )
        memory = MemoryObject(
            content=content,
            role="assistant" if is_assistant else "user",
            user_id=user_id,
            user_name=config.BOT_NAME if is_assistant else current_msg.get("user_name", "Unknown"),
            timestamp=current_msg.get("timestamp", time.time()),
            activation=50.0,
            related_users=[user_id],
        )
        memory.embedding = self.api.get_embedding(content)
        return [memory]

    def _think(self, memory_context: ContextBundle, current_user_id) -> str:
        """
        [Cognitive Reconstruction]
        STM과 LTM의 데이터를 LLM이 읽기 쉬운 '자연어 요약'으로 변환합니다.
        [중요] 이때 기계적인 ID("12345")를 닉네임("민초단장")으로 렌더링합니다.
        """
        # 1. STM에서 전체 대화 흐름 가져오기
        stm_context = self.stm.get_chronological_context()
        
        # 2. Fast LLM(System 1)에게 요약 요청
        return self._run_fast_reconstruction(stm_context, memory_context, current_user_id)

    def _act(self, user_id, user_input, context_summary, relationship_desc) -> str:
        """
        [Action & Feedback Loop]
        Generate Response -> Update Mood -> Update Relationship -> Save Memory
        """
        # 1. System 2 (Slow Thinking) - 답변 및 자연어 감정 생성
        response_text, natural_emotion = self._run_slow_generation(
            user_input, context_summary, relationship_desc, self.current_mood
        )
        
        # 2. Feedback Loop
        # (A) 기분 업데이트
        self.current_mood = natural_emotion
        self.session_state["current_mood"] = natural_emotion
        
        # (B) 관계 업데이트 (Social Manager 위임)
        # 봇이 생성한 자기 감정보다, 사용자의 실제 발화를 관계 신호로 사용한다.
        interaction_signal_vec = self.api.get_embedding(user_input)
        self.social.calculate_and_update_affinity(user_id, interaction_signal_vec)
        
        # (C) 자가 기억(Self-Memory) STM 저장. 감정 태그 보존!
        bot_mem = MemoryObject(
            content=response_text,
            role="assistant",
            user_id=self.bot_id if self.bot_id else "bot", # ID 일관성 유지
            user_name=config.BOT_NAME,
            activation=100.0, # 내 말은 중요하므로 높은 초기값
            emotion_tag=natural_emotion
        )
        bot_mem.embedding = self.api.get_embedding(response_text) if response_text else [0.0] * 1536
        self.stm.inject_memories([bot_mem])
        self.reflector.submit_memories([bot_mem])
        
        return response_text

    # =========================================================================
    # LLM Wrappers (Prompt Engineering Layer)
    # =========================================================================
    
    def _run_fast_reconstruction(self, stm_list, memory_context: ContextBundle, current_user_id) -> str:
        """
        [System 1: Groq]
        파편화된 기억들을 읽기 쉬운 상황 요약문으로 변환합니다.
        *ID Rendering 적용*
        """
        # --- 1. ID Rendering Helper ---
        # 등장하는 모든 user_id의 닉네임을 캐싱해둠
        # [Fix] 자아 명확화: LLM이 자신을 '코봇'으로 인식하게 함
        nickname_map = {
            "bot": f"{config.BOT_NAME}(Me)", 
            "Me": f"{config.BOT_NAME}(Me)",
            "assistant": f"{config.BOT_NAME}(Me)",
            self.bot_id: f"{config.BOT_NAME}(Me)"
        }
        
        # 현재 유저 닉네임 가져오기
        curr_ctx = self.social.get_user_context(current_user_id)
        nickname_map[current_user_id] = curr_ctx["nickname"]
        
        def render_id(uid, default_name):
            if uid in nickname_map: return nickname_map[uid]
            # 캐시에 없으면 조회
            ctx = self.social.get_user_context(uid)
            nickname_map[uid] = ctx["nickname"]
            return ctx["nickname"]

        # --- 2. Prompt Construction ---
        
        # STM (단기 기억) 렌더링
        stm_text = ""
        for m in stm_list:
            # 봇 자신인 경우 강제로 "코봇(Me)" 사용
            if m.role == "assistant" or m.user_id == self.bot_id:
                name = f"{config.BOT_NAME}(Me)"
            else:
                name = render_id(m.user_id, m.user_name)
            stm_text += f"[{name}]: {m.content}\n"

        # LTM (장기 기억) 렌더링
        ltm_text = ""
        for node in memory_context.open_loops:
            ltm_text += f"- (Open Loop) {node.nl_summary}\n"
        for node in memory_context.active_claims:
            if isinstance(node, ClaimNode):
                ltm_text += f"- (Current State) {node.nl_summary} [{node.facet}]\n"
        for node in memory_context.relevant_schedule:
            ltm_text += f"- (Schedule) {node.nl_summary}\n"
        for node in memory_context.supporting_notes:
            ltm_text += f"- (Note) {node.summary} (태그: {', '.join(node.tags) if node.tags else '없음'})\n"
        for node in memory_context.legacy_insights:
            ltm_text += f"- (Legacy Fact) {node.summary} (신뢰도: {node.confidence})\n"
        for node in memory_context.supporting_events:
            if node.user_id == self.bot_id:
                name = f"{config.BOT_NAME}(Me)"
            else:
                name = render_id(node.user_id, "Unknown")
            ltm_text += f"- (Memory) {name}가 말하길: {node.content} (기분: {node.emotion_tag})\n"

        relation_state = memory_context.relation_state
        relation_text = (
            f"trust={relation_state.trust:.2f}, warmth={relation_state.warmth:.2f}, "
            f"familiarity={relation_state.familiarity:.2f}, respect={relation_state.respect:.2f}, "
            f"tension={relation_state.tension:.2f}, reliability={relation_state.reliability:.2f}"
            if relation_state else "없음"
        )
        interaction_policy_lines = "\n".join(
            f"- {key}: {value}" for key, value in memory_context.interaction_policy.items()
        ) or "- 없음"
        uncertainty_lines = "\n".join(f"- {item}" for item in memory_context.uncertainties) or "- 없음"

        # --- 3. Call API ---
        system_prompt = (
            "당신은 '상황 분석가'입니다. "
            "주어진 [과거 기억]과 [현재 대화]를 종합하여, 현재 대화의 맥락을 3문장 이내로 명확히 요약하세요. "
            f"대화에서 '{config.BOT_NAME}(Me)'은 당신 자신입니다."
        )
        
        user_prompt = f"""
        [Open Loops / Claims]
        {ltm_text}

        [Interaction Policy]
        {interaction_policy_lines}

        [Relation State]
        {relation_text}

        [Uncertainties]
        {uncertainty_lines}

        [Relevant Memories]
        {stm_text}
        """

        return self.api.chat_fast(system_prompt, user_prompt)
    
    def _run_slow_generation(self, user_input, context, relationship, bot_mood) -> Tuple[str, str]:
        """
        [System 2: GPT-4]
        답변 생성 및 10자 이내 자연어 감정 태그 생성
        """
        system_prompt = f"""
        당신은 자아와 감정을 가진 사람{config.BOT_NAME}입니다.
        
        [Persona Info]
        - 상대방과의 관계: {relationship}
        - 현재 당신의 기분: "{bot_mood}"
        
        [Task]
        1. 상대방의 말에 대해 페르소나와 기분을 반영하여 자연스럽게 답변하세요.
        2. 답변 끝에 당신이 지금 느끼는 솔직한 감정을 [FEELING:감정] 태그로 붙이세요.
        
        [Rule]
        - 감정은 '기쁨', '슬픔' 같은 카테고리가 아니라, "약간 설렘", "차가운 경멸", "따뜻한 위로" 처럼 10자 이내의 자연어로 표현하세요.
        - [FEELING:...] 태그는 답변의 맨 마지막에 딱 한 번만 나와야 합니다.
        
        Example: "진짜? 와, 그거 완전 대박이다! [FEELING:놀라움과 부러움]"
        """
        
        user_prompt = f"""
        [Context Summary]
        {context}
        
        [User Input]
        "{user_input}"
        """
        
        full_response = self.api.chat_slow(system_prompt, user_prompt)
        
        # Parse Tag
        tag_pattern = r"\[FEELING:(.*?)\]"
        match = re.search(tag_pattern, full_response)
        
        if match:
            emotion = match.group(1).strip()[:10]
            text = re.sub(tag_pattern, "", full_response).strip()
            return text, emotion
        else:
            return full_response, bot_mood # 태그 없으면 기존 기분 유지
