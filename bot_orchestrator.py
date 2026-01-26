import re
import numpy as np
from typing import List, Dict, Tuple

# Config & Structs
import config
from memory_structures import RetrievalQuery, MemoryObject

# Modules
from api_clients import UnifiedAPIClient
from modules.sensory_system import SensorySystem
from modules.stm_handler import WorkingMemory
from modules.ltm_graph import MemoryGraph
from modules.ltm_handler import LongTermMemory
from modules.reflection_handler import ReflectionHandler
from modules.social_module import SocialMap

'''
TODO:
감정 상태 저장에 대한 로직 변경. (현재 의도대로 안되어있음)


after work:
엔트리포인트 설정해야함.
'''

class BotOrchestrator:
    def __init__(self):
        self.api = UnifiedAPIClient()
        self.sensory = SensorySystem(self.api)
        self.stm = WorkingMemory()
        self.ltm_graph = MemoryGraph()
        self.ltm = LongTermMemory(self.ltm_graph, self.api)
        self.social = SocialMap()
        
        self.current_mood = "calm"
        self.positive_anchor_vec = self.api.get_embedding(config.POSITIVE_EMOTION_ANCHOR)
        
        # Background Process
        self.reflector = ReflectionHandler(self.ltm_graph, self.api)
        self.reflector.start_background_loop(
            self.stm.eviction_buffer, interval=config.REFLECTION_INTERVAL
        )

    def process_trigger(self, history: List[Dict], calling_message: Dict) -> str:
        """
        Main cognitive loop

        # I. Trigger (1)
        트리거로 최근 대화기록(history)과 트리거 이벤트(calling_message)를 받음으로써 호출됨.
        """
        user_id = str(calling_message.get("user_id"))

        # II. Perception (2, 3)
        chunked_memories = self._perceive(history, calling_message)
        
        # III. Retrieval & Attention (4, 5, 6, 7) - 수정점. 5 이름이 Search and Traverse인데, Attention이 빠졌고, 5와 6의 경계가 불분명.
        query_text = chunked_memories[-1].content
        retrieved_nodes = self._retrieve_and_attend(query_text, user_id)
        
        # IV. Cognition & Context Reconstruction (8)
        context_summary = self._think(chunked_memories, retrieved_nodes)
        rel_desc = self.social.get_relationship_desc(user_id)
        
        # V. Action (9)
        response = self._act(query_text, context_summary, rel_desc)
        
        return response

    # --- Private Pipelines ---

    def _perceive(self, history, calling_message) -> List[MemoryObject]:
        # Sensory System (Chunking)
        chunks = self.sensory.process_input(history, calling_message)
        # Inject into Working Memory (STM)
        self.stm.inject_memories(chunks)
        return chunks

    def _retrieve_and_attend(self, query_text: str, user_id: str):
        # Create Query
        query_embedding = self.api.get_embedding(query_text)
        query = RetrievalQuery(
            embedding=query_embedding,
            user_id=user_id,
            keywords=query_text.split(),
            current_mood=self.current_mood
        )
        
        # LTM Handler (Retrieval)
        nodes = self.ltm.retrieve(query, top_k=3)
        
        # STM Attention (Vector Semantic Matching)
        self.stm.update_activations(query_embedding)
        
        return nodes

    def _think(self, stm_chunks, ltm_nodes) -> str:
        stm_context = self.stm.get_chronological_context()
        # Fast LLM을 이용한 요약
        return self._run_fast_reconstruction(stm_context, ltm_nodes)

    def _act(self, user_id, user_input, context, relationship) -> str:
        """
        Action: 답변 생성 -> 감정 변화 -> 관계(호감도) 계산 -> 기억 저장
        """
        # 1. LLM 생성 (자연어 감정 태그 포함)
        response_text, natural_emotion = self._run_slow_generation(
            user_input, context, relationship, self.current_mood
        )
        
        # 2. Feedback Loop
        # (1) 기분 업데이트 (자연어 그대로)
        self.current_mood = natural_emotion
        
        # (2) [New] 벡터 기반 관계 업데이트
        self._update_social_relationship(user_id, natural_emotion)
        
        # (3) 자가 기억 STM 저장 (감정 태그 그대로 보존)
        bot_mem = MemoryObject(
            content=response_text,
            role="assistant",
            user_id="bot",
            user_name="Me",
            activation=100.0,
            emotion_tag=natural_emotion # "싸늘한 분노" 등이 그대로 들어감
        )
        self.stm.inject_memories([bot_mem])
        
        return response_text

    def _update_social_relationship(self, user_id: str, current_emotion_text: str):
        """
        [Vector-based Social Logic]
        현재 봇의 기분(current_emotion)이 '긍정(Positive Anchor)'과 얼마나 가까운가?
        가까우면 호감도 상승, 멀면 하락.
        """
        # 1. 현재 감정 임베딩
        emotion_vec = self.api.get_embedding(current_emotion_text)
        
        # 2. 코사인 유사도 계산 (-1.0 ~ 1.0)
        similarity = self._cosine_similarity(emotion_vec, self.positive_anchor_vec)
        
        # 3. 점수 매핑 (Mapping Strategy)
        # 유사도 1.0 (완전 긍정) -> +5.0점
        # 유사도 0.0 (무관/중립) -> 0.0점
        # 유사도 -1.0 (완전 반대) -> -5.0점
        # 단, 노이즈를 줄이기 위해 유사도 0.2 미만은 무시할 수도 있음 (여기선 그대로 적용)
        
        delta = similarity * config.SOCIAL_SENSITIVITY
        
        # 4. 업데이트 적용
        self.social.update_affinity(user_id, delta)
        
        # Debug Log
        # print(f"❤️ [Social] '{current_emotion_text}' (Sim: {similarity:.2f}) -> Delta: {delta:+.2f}")

    def _cosine_similarity(self, vec_a, vec_b):
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0: return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    # --- LLM Wrappers ---
    
    def _run_fast_reconstruction(self, stm_list, ltm_nodes) -> str:
        """
        [System 1] Groq: 기억 파편들을 읽기 쉬운 텍스트로 요약
        """
        # 1. 프롬프트 데이터 구성
        stm_text = "\n".join([f"[{m.user_name}]: {m.content}" for m in stm_list])
        ltm_text = ""
        for node in ltm_nodes:
            if hasattr(node, 'summary'): 
                ltm_text += f"- (Fact) {node.summary}\n"
            else: 
                ltm_text += f"- (Memory) {node.content} (Emotion: {node.emotion_tag})\n"
        
        # 2. 프롬프트 분리 (System / User)
        system_prompt = (
            "당신은 상황 분석가입니다. "
            "주어진 [장기 기억]과 [현재 대화]를 바탕으로, 현재 대화의 맥락을 3문장 이내로 요약하세요. "
            "특히 과거 기억이 현재 대화와 어떻게 연결되는지 명시하세요."
        )
        
        user_prompt = f"""
        [Long-term Memory]
        {ltm_text}
        
        [Current Conversation]
        {stm_text}
        """

        # 3. API 호출 (Groq)
        return self.api.chat_fast(system_prompt, user_prompt)
    
    def _run_slow_generation(self, user_input, context, relationship, bot_mood) -> Tuple[str, str]:
        """
        [System 2] GPT-4: 답변 및 10자 이내 자연어 감정 생성
        """
        system_prompt = f"""
        당신은 자아를 가진 AI입니다. 
        유저와의 관계({relationship})와 현재 당신의 기분("{bot_mood}")을 반영하여 답변하세요.
        
        [출력 규칙]
        답변 끝에 당신이 느끼는 **구체적인 감정**을 [FEELING:감정단어] 형태로 붙이세요.
        - 감정 단어는 10자 이내의 자연어로 자유롭게 표현하세요. (예: "묘한 설렘", "차가운 분노", "귀찮음", "안도감")
        - 카테고리로 분류하지 말고, **뉘앙스**를 살리세요.
        
        예시 Output: "아 진짜? 그건 좀 너무했다. [FEELING:어이없음]"
        """
        
        user_prompt = f"""
        [상황]
        {context}
        
        [유저]
        "{user_input}"
        """
        
        full_response = self.api.chat_slow(system_prompt, user_prompt)
        
        # 파싱 로직
        tag_pattern = r"\[FEELING:(.*?)\]"
        match = re.search(tag_pattern, full_response)
        
        if match:
            emotion = match.group(1).strip()[:10] # 10자 제한 안전장치
            text = re.sub(tag_pattern, "", full_response).strip()
            return text, emotion
        else:
            return full_response, "calm" # 기본값