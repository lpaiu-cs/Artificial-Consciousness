import json
import time
import threading
from typing import List, Dict, Any, Tuple # [수정] Tuple 추가
from memory_structures import MemoryObject
from modules.ltm_graph import MemoryGraph
from api_client import UnifiedAPIClient
import config

class ReflectionHandler:
    """
    [System 2: Consolidation]
    STM에서 방출(Eviction)된 기억 파편들을 모아
    1. 하나의 에피소드(Episode)로 요약하고
    2. 그 안에서 통찰(Insight)을 추출하여
    3. Memory Graph에 저장하고 연결(Wiring)합니다.
    
    Thread Safety:
    - MemoryGraph는 내부적으로 Lock을 사용하므로 
      이 핸들러에서는 별도의 Lock 없이 그래프 메서드 호출
    - STM 버퍼 접근 시에는 스냅샷 복사 후 처리
    """
    def __init__(self, ltm_graph: MemoryGraph, api_client: UnifiedAPIClient):
        self.graph = ltm_graph
        self.api = api_client
        self.is_running = False
        self._buffer_lock = threading.Lock()  # STM 버퍼 접근용 Lock
        
    def start_background_loop(self, stm_buffer: List[MemoryObject], interval=60):
        """백그라운드에서 주기적으로 Reflection 수행"""
        self.is_running = True
        thread = threading.Thread(target=self._loop, args=(stm_buffer, interval))
        thread.daemon = True
        thread.start()

    def stop(self):
        """백그라운드 루프 중지"""
        self.is_running = False

    def _loop(self, stm_buffer, interval):
        while self.is_running:
            time.sleep(interval)
            # Thread-Safe: Lock으로 버퍼 길이 확인
            with self._buffer_lock:
                buffer_size = len(stm_buffer)
            
            if buffer_size >= 3: 
                self.process_buffer(stm_buffer)

    def process_buffer(self, buffer: List[MemoryObject]):
        """버퍼의 내용을 가져와 LTM에 저장 (Thread-Safe)"""
        # 1. 데이터 스냅샷 & 버퍼 비우기 (Thread-Safe)
        with self._buffer_lock:
            if not buffer: 
                return
            chunk = buffer[:]
            buffer.clear()  # 원본 리스트 비우기
        
        print(f"💤 [Reflection] 기억 정리 및 저장 시작... ({len(chunk)} items)")
        
        # 2. LLM에게 분석 요청 (GPT-4)
        analysis_result = self._analyze_with_llm(chunk)
        
        if not analysis_result: 
            print("⚠️ [Reflection] 분석 실패 또는 결과 없음")
            return

        # 3. 그래프에 저장 (Graph Update) + 임베딩 생성
        # MemoryGraph는 내부적으로 Lock 사용하므로 안전
        self._apply_to_graph(analysis_result)

        # 4. [중요] 영속성 저장 (파일 쓰기) - Thread-Safe
        self.graph.save_all()

    def _analyze_with_llm(self, memories: List[MemoryObject]) -> Dict[str, Any]:
        """LLM: Raw Logs -> Structured Memory (Episode & Insights)"""
        logs_text = "\n".join([f"[{m.role}] {m.content} (Feel: {m.emotion_tag})" for m in memories])
        # 대표 유저 ID 추출 (가장 빈번하게 등장한 ID를 쓰는 것이 좋으나 여기선 첫번째 사용)
        user_id = memories[0].user_id

        system_prompt = (
            "당신은 AI의 기억 관리자(Memory Manager)입니다. "
            "주어진 대화 로그를 분석하여 장기 기억으로 변환할 수 있도록 구조화된 JSON을 생성하세요."
        )

        user_prompt = f"""
        [대화 로그]
        {logs_text}

        Analyze the conversation and output JSON:
        1. episode_summary: 대화 전체를 "누가, 언제, 무엇을 했다" 형태로 1문장 요약.
        2. dominant_feeling: Select the most representative 'feeling' tag found in the logs (or a combined nuance). Keep it under 10 chars. (e.g. "어이없음", "따뜻한 위로")
        3. insights: 대화에서 발견된 유저에 대한 불변의 사실(Fact)이나 성향 리스트.
           형식: list of {{subject, predicate, object, summary}}.
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
            
            # [Evidence Edge] 연결 - config에서 가중치 사용
            # 통찰 <-> 에피소드
            self.graph.connect_nodes(
                ins_node.node_id, ep_node.node_id, 
                weight=config.EVIDENCE_EDGE_TO_EPISODE
            )
            self.graph.connect_nodes(
                ep_node.node_id, ins_node.node_id, 
                weight=config.EVIDENCE_EDGE_TO_INSIGHT
            )
            
        print(f"✅ [Reflection] 저장 완료: Episode({ep_node.node_id[:8]}) + {len(insights)} Insights saved.")