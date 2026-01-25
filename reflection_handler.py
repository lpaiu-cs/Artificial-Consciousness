import json
import time
import threading
from typing import List, Dict, Any, Tuple # [수정] Tuple 추가
from memory_structures import MemoryObject
from ltm_graph import MemoryGraph
from api_clients import UnifiedAPIClient

class ReflectionHandler:
    """
    [System 2: Consolidation]
    STM에서 방출(Eviction)된 기억 파편들을 모아
    1. 하나의 에피소드(Episode)로 요약하고
    2. 그 안에서 통찰(Insight)을 추출하여
    3. Memory Graph에 저장하고 연결(Wiring)합니다.
    """
    def __init__(self, ltm_graph: MemoryGraph, api_client: UnifiedAPIClient):
        self.graph = ltm_graph
        self.api = api_client
        self.is_running = False
        
    def start_background_loop(self, stm_buffer: List[MemoryObject], interval=60):
        """백그라운드에서 주기적으로 Reflection 수행"""
        self.is_running = True
        thread = threading.Thread(target=self._loop, args=(stm_buffer, interval))
        thread.daemon = True
        thread.start()

    def _loop(self, stm_buffer, interval):
        while self.is_running:
            time.sleep(interval)
            # 버퍼에 데이터가 쌓였는지 확인 (Lock 처리가 이상적이나, 리스트 atomic 연산 의존)
            if len(stm_buffer) >= 3: 
                self.process_buffer(stm_buffer)

    def process_buffer(self, buffer: List[MemoryObject]):
        """버퍼의 내용을 가져와 LTM에 저장"""
        # 1. 데이터 스냅샷 & 버퍼 비우기 (Thread-safe)
        if not buffer: return
        
        chunk = buffer[:]
        buffer.clear() # 원본 리스트 비우기
        
        print(f"💤 [Reflection] 기억 정리 및 저장 시작... ({len(chunk)} items)")
        
        # 2. LLM에게 분석 요청 (GPT-4)
        analysis_result = self._analyze_with_llm(chunk)
        
        if not analysis_result: 
            print("⚠️ [Reflection] 분석 실패 또는 결과 없음")
            return

        # 3. 그래프에 저장 (Graph Update) + 임베딩 생성
        self._apply_to_graph(analysis_result)

        # 4. [중요] 영속성 저장 (파일 쓰기)
        self.graph.save_to_json()

    def _analyze_with_llm(self, memories: List[MemoryObject]) -> Dict[str, Any]:
        """LLM: Raw Logs -> Structured Memory (Episode & Insights)"""
        
        logs_text = "\n".join([f"[{m.role}] {m.content}" for m in memories])
        # 대표 유저 ID 추출 (가장 빈번하게 등장한 ID를 쓰는 것이 좋으나 여기선 첫번째 사용)
        user_id = memories[0].user_id 

        system_prompt = (
            "당신은 AI의 기억 관리자(Memory Manager)입니다. "
            "주어진 대화 로그를 분석하여 장기 기억으로 변환할 수 있도록 구조화된 JSON을 생성하세요."
        )

        user_prompt = f"""
        [대화 로그]
        {logs_text}

        위 대화를 분석하여 다음 JSON 형식으로 출력하세요:
        1. episode_summary: 대화 전체를 "누가, 언제, 무엇을 했다" 형태로 1문장 요약.
        2. dominant_emotion: 이 대화의 지배적인 감정 (joy, sadness, anger, neutral, trust 중 택1).
        3. insights: 대화에서 발견된 유저에 대한 불변의 사실(Fact)이나 성향 리스트.
           (형식: subject(주어), predicate(술어), object(목적어), summary(설명))

        [Output Example]
        {{
            "episode_summary": "유저가 비 오는 날씨를 보고 과거 파전 먹던 추억을 이야기함.",
            "dominant_emotion": "joy",
            "insights": [
                {{"subject": "User", "predicate": "likes", "object": "Rainy days", "summary": "유저는 비 오는 날의 분위기를 좋아함"}}
            ],
            "user_id": "{user_id}" 
        }}
        """

        # API Client의 chat_slow (GPT-4) 호출, json_mode=True 사용
        result = self.api.chat_slow(system_prompt, user_prompt, json_mode=True)
        
        # 혹시 모를 user_id 누락 방지
        if isinstance(result, dict) and "user_id" not in result:
            result["user_id"] = user_id
            
        return result

    def _apply_to_graph(self, data: Dict[str, Any]):
        """분석 결과를 노드와 엣지로 변환하여 저장"""
        
        # 1. 에피소드 임베딩 생성 및 노드 추가
        ep_summary = data.get("episode_summary", "")
        ep_embedding = self.api.get_embedding(ep_summary) # [API 호출]
        
        ep_node = self.graph.add_episode(
            content=ep_summary,
            user_id=data.get("user_id", "unknown"),
            emotion=data.get("dominant_emotion", "neutral"),
            embedding=ep_embedding
        )
        
        # 2. 통찰(Insight) 처리
        insights = data.get("insights", [])
        for info in insights:
            ins_summary = info.get("summary", "")
            
            # Insight 임베딩 생성
            ins_embedding = self.api.get_embedding(ins_summary) # [API 호출]
            
            ins_node = self.graph.add_or_update_insight(
                summary=ins_summary,
                subject=info.get("subject", "User"),
                predicate=info.get("predicate", "is"),
                object_=info.get("object", "Unknown"),
                embedding=ins_embedding
            )
            
            # [Evidence Edge] 연결
            # 통찰 <-> 에피소드 (가중치는 로직에 따라 조절 가능)
            self.graph.connect_nodes(ins_node.node_id, ep_node.node_id, weight=0.8)
            self.graph.connect_nodes(ep_node.node_id, ins_node.node_id, weight=1.0)
            
        print(f"✅ [Reflection] 저장 완료: Episode({ep_node.node_id[:8]}) + {len(insights)} Insights saved.")