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
        # (이전 코드와 동일, api.chat_fast 사용)
        return "(Summary) ..." 

    def _run_slow_generation(self, user_input, context, relationship, bot_mood) -> Tuple[str, str]:
        # (이전 코드와 동일, api.chat_slow 및 Regex 파싱 사용)
        return "Response", "neutral"