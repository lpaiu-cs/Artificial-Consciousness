import re
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

class BotOrchestrator:
    def __init__(self):
        self.api = UnifiedAPIClient()
        self.sensory = SensorySystem()
        self.stm = WorkingMemory()
        self.ltm_graph = MemoryGraph()
        self.ltm = LongTermMemory(self.ltm_graph, self.api)
        self.social = SocialMap()
        
        self.current_mood = "neutral"
        
        # Background Process
        self.reflector = ReflectionHandler(self.ltm_graph, self.api)
        self.reflector.start_background_loop(
            self.stm.eviction_buffer, interval=config.REFLECTION_INTERVAL
        )

    def process_trigger(self, history: List[Dict], current_msg_data: Dict) -> str:
        """Main Cognitive Loop"""
        user_id = str(current_msg_data.get("user_id"))

        # 1. Perception (감각)
        chunked_memories = self._perceive(history, current_msg_data)
        
        # 2. Retrieval & Attention (기억 인출 및 주의 집중)
        query_text = chunked_memories[-1].content
        retrieved_nodes = self._retrieve_and_attend(query_text, user_id)
        
        # 3. Cognition (사고 및 맥락 재구성)
        context_summary = self._think(chunked_memories, retrieved_nodes)
        rel_desc = self.social.get_relationship_desc(user_id)
        
        # 4. Action (답변 생성 및 상태 업데이트)
        response = self._act(query_text, context_summary, rel_desc)
        
        return response

    # --- Private Pipelines ---

    def _perceive(self, history, current_msg) -> List[MemoryObject]:
        chunks = self.sensory.process_input(history, current_msg)
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
        
        # LTM Search
        nodes = self.ltm.retrieve(query, top_k=3)
        
        # STM Attention (Scoring)
        referenced_ids = set()
        # (간단화) 현재 키워드가 포함된 STM 기억 강화
        for mem in self.stm.get_chronological_context():
            if any(k in mem.content for k in query.keywords):
                referenced_ids.add(mem.mem_id)
        self.stm.update_activations(referenced_ids)
        
        return nodes

    def _think(self, stm_chunks, ltm_nodes) -> str:
        stm_context = self.stm.get_chronological_context()
        # Fast LLM을 이용한 요약
        return self._run_fast_reconstruction(stm_context, ltm_nodes)

    def _act(self, user_input, context, relationship) -> str:
        # Slow LLM을 이용한 답변 생성
        response_text, emotion = self._run_slow_generation(
            user_input, context, relationship, self.current_mood
        )
        
        # Feedback Loop
        self.current_mood = emotion
        bot_mem = MemoryObject(
            content=response_text, role="assistant", 
            user_id="bot", user_name="Me", activation=100.0
        )
        self.stm.inject_memories([bot_mem])
        
        return response_text

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
        [System 2] GPT-4: 페르소나와 맥락을 반영한 답변 생성
        Return: (답변 텍스트, 감정 태그)
        """
        # 1. System Prompt (페르소나 및 출력 형식 정의)
        system_prompt = f"""
        당신은 감정을 가진 AI '잼봇'입니다.
        유저와의 관계({relationship})와 당신의 현재 기분({bot_mood})을 반영하여 답변하세요.
        
        반드시 답변 끝에 감정 태그를 붙이세요. 예: "안녕? [EMOTION:joy]"
        감정 종류: joy, sadness, anger, neutral, trust
        """
        
        # 2. User Prompt (상황 정보)
        user_prompt = f"""
        [상황 요약]
        {context}
        
        [유저의 말]
        "{user_input}"
        """
        
        # 3. API 호출 (GPT-4)
        full_response = self.api.chat_slow(system_prompt, user_prompt)
        
        # 4. 응답 파싱 (텍스트와 감정 분리)
        # 예: "반가워! [EMOTION:joy]" -> "반가워!", "joy"
        emotion_pattern = r"\[EMOTION:([a-zA-Z]+)\]"
        match = re.search(emotion_pattern, full_response)
        
        if match:
            emotion = match.group(1).lower()
            text = re.sub(emotion_pattern, "", full_response).strip()
            return text, emotion
        else:
            return full_response, "neutral" # 태그가 없으면 중립