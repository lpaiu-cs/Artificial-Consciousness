import time
import threading
import json
import logging
from typing import List, Dict, Any

import config
from memory_structures import MemoryObject, EpisodeNode, InsightNode, EntityNode
from modules.ltm_graph import MemoryGraph
from api_client import UnifiedAPIClient

class ReflectionHandler:
    """
    [Background Process]
    STM에서 방출(Eviction)된 기억들을 주기적으로 가져와 LTM에 저장합니다.
    단순 저장이 아니라, LLM을 통해 '성찰(Reflection)'하여 구조화된 데이터로 변환합니다.
    """

    def __init__(self, graph_db: MemoryGraph, api_client: UnifiedAPIClient):
        self.graph = graph_db
        self.api = api_client
        self.stop_event = threading.Event()
        self.thread = None

    def start_background_loop(self, eviction_buffer: List[MemoryObject], interval: int = 30):
        """STM의 eviction_buffer를 주기적으로 감시하는 데몬 스레드 시작"""
        if self.thread and self.thread.is_alive():
            return

        def loop():
            while not self.stop_event.is_set():
                time.sleep(interval)
                if eviction_buffer:
                    # 버퍼에 있는 모든 기억을 가져옴 (Batch Processing)
                    batch = eviction_buffer[:]
                    eviction_buffer.clear() # 버퍼 비우기
                    
                    try:
                        self._process_batch(batch)
                    except Exception as e:
                        logging.error(f"❌ Reflection Error: {e}")

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        print("🌙 Reflection Handler Started (Background)")

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()

    def _process_batch(self, memories: List[MemoryObject]):
        """
        1. LLM 분석: 요약, 감정 추출, 통찰 추출
        2. 임베딩 생성: 검색을 위한 벡터화
        3. 그래프 저장: Node 생성 및 Edge 연결 (3-Tier Wiring)
        """
        if not memories: 
            return

        # 1. LLM 분석 (Analyze)
        analysis_result = self._analyze_with_llm(memories)
        if not analysis_result:
            return

        episode_summary = analysis_result.get("episode_summary", "")
        dominant_emotion = analysis_result.get("dominant_emotion", "neutral")
        insights_data = analysis_result.get("insights", [])

        # 2. 관련 Entity 식별 (Identify Entities)
        # 배치에 포함된 모든 유저의 ID와 닉네임을 수집
        involved_users = {} # {user_id: nickname}
        for mem in memories:
            involved_users[mem.user_id] = mem.user_name
            # (추후 확장) mem.related_users에 있는 제3자도 포함 가능

        # 3. 그래프 저장 및 연결 (Persist & Wire)
        
        # (A) Episode Node 생성
        # 검색을 위해 요약문에 대한 임베딩 생성
        ep_embedding = self.api.get_embedding(episode_summary)
        
        # 대표 화자(Primary User) 설정 - 보통 첫 번째 메시지의 유저
        primary_user_id = memories[0].user_id
        
        episode_node = self.graph.add_episode(
            content=episode_summary,
            user_id=primary_user_id,
            emotion=dominant_emotion, # 자연어 감정 보존
            embedding=ep_embedding
        )

        # (B) Entity Node 연결 (PARTICIPATED_IN)
        for uid, nickname in involved_users.items():
            # Entity 확보 (없으면 생성, 닉네임 최신화)
            self.graph.get_or_create_user(uid, nickname)
            # Edge 연결
            self.graph.link_user_to_episode(uid, episode_node.node_id)

        # (C) Insight Node 생성 및 연결 (EVIDENCE_OF, ABOUT)
        for insight_text in insights_data:
            # 임베딩 생성
            in_embedding = self.api.get_embedding(insight_text)
            
            # Insight 생성
            insight_node = self.graph.add_or_update_insight(
                summary=insight_text,
                confidence=0.8, # 기본 신뢰도
                embedding=in_embedding
            )
            
            # Edge 1: Insight <-> Episode (증거)
            self.graph.connect_nodes(
                insight_node.node_id, 
                episode_node.node_id, 
                weight=config.EVIDENCE_EDGE_TO_EPISODE
            )
            
            # Edge 2: Insight <-> Entity (대상)
            # 통찰은 보통 주 화자에 대한 것임
            # (추후 LLM이 '누구에 대한 통찰인지' 지정하게 할 수도 있음)
            self.graph.connect_nodes(
                insight_node.node_id, # Source: 지식
                self.graph.get_or_create_user(primary_user_id, "").node_id, # Target: 유저
                weight=1.2 # 매우 강한 연결
            )

        # 4. 저장 확정 (File I/O)
        self.graph.save_all()
        # print(f"💾 Reflected: {episode_summary[:30]}... (Ins: {len(insights_data)})")

    def _analyze_with_llm(self, memories: List[MemoryObject]) -> Dict[str, Any]:
        """
        [System 2] GPT-4를 이용해 파편화된 대화 로그를 구조화된 데이터로 변환
        """
        # 로그 텍스트 변환
        logs_text = ""
        for m in memories:
            logs_text += f"[{m.user_name}({m.user_id})]: {m.content} (EmotionTag: {m.emotion_tag})\n"

        system_prompt = """
        You are a generic 'Memory Manager' for an AI.
        Your job is to consolidate raw chat logs into a meaningful memory structure.
        
        [Output Format]
        Return a JSON object with the following fields:
        1. "episode_summary": A concise, 1-sentence summary of the conversation event. (e.g. "Mincho-dan discussed his preference for mint chocolate.")
        2. "dominant_emotion": The overall emotional tone of the user in this interaction. Use a natural language phrase (under 10 chars). (e.g. "Passionately defensive", "Calm curiosity")
        3. "insights": A list of strings. Extract factual knowledge or personality traits revealed in the conversation. If none, return empty list. (e.g. ["User loves mint chocolate.", "User dislikes spicy food."])
        
        [Constraint]
        - Analyze purely based on the logs.
        - The language of summary and insights must match the language of the logs (Korean).
        """
        
        user_prompt = f"""
        [Raw Logs]
        {logs_text}
        
        Analyze and Extract JSON:
        """

        try:
            # chat_slow에 json_mode=True 옵션 사용 권장
            response = self.api.chat_slow(system_prompt, user_prompt, json_mode=True)
            if isinstance(response, str):
                 return json.loads(response)
            return response
        except Exception as e:
            logging.error(f"Reflection LLM Parsing Error: {e}")
            return {}