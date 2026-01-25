import re
from typing import List, Dict, Tuple

# --- Modules ---
from sensory_system import SensorySystem
from stm_handler import WorkingMemory
from ltm_graph import MemoryGraph
from ltm_handler import LongTermMemory
from reflection_handler import ReflectionHandler
from social_module import SocialMap
from memory_structures import RetrievalQuery, MemoryObject
from api_clients import UnifiedAPIClient

class BotOrchestrator:
    """
    [The Ego]
    감각, 기억, 감정, 사고를 총괄하는 중앙 제어 장치.
    """
    def __init__(self):
        # 0. API Clients
        self.api = UnifiedAPIClient()

        # 1. Memory Systems
        self.sensory = SensorySystem()
        self.stm = WorkingMemory(capacity=15)
        self.ltm_graph = MemoryGraph()
        self.ltm = LongTermMemory(self.ltm_graph, self.api)
        
        # 2. Social & Emotion
        self.social = SocialMap()
        self.current_mood = "neutral" # 봇의 현재 기분 (state)

        # 3. Background Processes
        # STM에서 밀려난 기억을 LTM으로 넘기는 성찰 프로세스 가동
        self.reflector = ReflectionHandler(self.ltm_graph, self.api)
        self.reflector.start_background_loop(self.stm.eviction_buffer, interval=30)

    def process_trigger(self, history: List[Dict], current_msg_data: Dict) -> str:
        """
        [Main Cognitive Loop]
        외부 자극(Trigger)이 들어왔을 때 작동하는 사고 과정
        """
        user_id = str(current_msg_data.get("user_id"))
        user_name = current_msg_data.get("user_name", "Unknown")
        
        # -------------------------------------------------------
        # Phase 1: Perception (감각 및 전처리)
        # -------------------------------------------------------
        # 파편화된 채팅 로그를 의미 단위(Chunk)로 변환
        chunked_memories = self.sensory.process_input(history, current_msg_data)
        
        # STM에 주입 (이 과정에서 중요도 낮은 기억은 Eviction Buffer로 밀려남)
        self.stm.inject_memories(chunked_memories)

        # -------------------------------------------------------
        # Phase 2: Retrieval (기억 인출)
        # -------------------------------------------------------
        # 검색 쿼리 구성 (현재 대화의 마지막 발화를 기준)
        last_chunk = chunked_memories[-1]
        query_text = last_chunk.content
        
        # 임베딩 생성
        query_embedding = self.api.get_embedding(query_text)
        
        # Context Query 객체 생성
        retrieval_query = RetrievalQuery(
            embedding=query_embedding,
            user_id=user_id,
            intent="chat", # 추후 Intent Classifier 연동 가능
            current_mood=self.current_mood,
            keywords=query_text.split() # 간단한 공백 토크나이징
        )
        
        # LTM에서 [Anchor & Spread] 전략으로 기억 인출
        retrieved_nodes = self.ltm.retrieve(retrieval_query, top_k=3)
        
        # -------------------------------------------------------
        # Phase 3: Attention & Scoring (주의 집중)
        # -------------------------------------------------------
        # 이번 사고 과정에서 참조된 STM 기억들에 가산점 부여 (Scoring)
        # (LTM 검색 결과와 연관된 STM 기억들을 찾아서 강화)
        referenced_ids = set()
        for node in retrieved_nodes:
             # 만약 LTM 노드가 STM에도 존재한다면 ID 추가 (여기선 생략)
             pass
             
        # 현재 대화 주제와 관련된 STM 기억 강화
        stm_snapshot = self.stm.get_chronological_context()
        for mem in stm_snapshot:
            # 간단한 키워드 매칭으로 관련성 판단
            if any(k in mem.content for k in retrieval_query.keywords):
                referenced_ids.add(mem.mem_id)
        
        self.stm.update_activations(referenced_ids)

        # -------------------------------------------------------
        # Phase 4: Context Reconstruction (맥락 재구성)
        # -------------------------------------------------------
        # Fast LLM (Groq)을 사용해 현재 상황 요약
        context_summary = self._run_fast_reconstruction(stm_snapshot, retrieved_nodes)
        
        # 사회적 맥락 가져오기
        rel_desc = self.social.get_relationship_desc(user_id)

        # -------------------------------------------------------
        # Phase 5: Meta-Cognition & Generation (답변 생성)
        # -------------------------------------------------------
        # Smart LLM (GPT-4)을 사용해 최종 답변 생성
        final_response, detected_emotion = self._run_slow_generation(
            user_input=query_text,
            context=context_summary,
            relationship=rel_desc,
            bot_mood=self.current_mood
        )

        # -------------------------------------------------------
        # Phase 6: Feedback Loop (상태 업데이트)
        # -------------------------------------------------------
        # 봇의 답변도 STM에 저장 (자신의 말을 기억)
        bot_mem = MemoryObject(
            content=final_response,
            role="assistant",
            user_id="bot",
            user_name="Me",
            activation=100.0 # 방금 한 말은 가장 생생함
        )
        self.stm.inject_memories([bot_mem])
        
        # 봇의 기분 업데이트
        self.current_mood = detected_emotion
        
        # (옵션) 유저 호감도 조정은 여기서 detected_emotion이나 유저 발화 분석을 통해 수행

        return final_response

    # --- Private LLM Helper Methods ---

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