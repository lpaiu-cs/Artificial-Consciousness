import os
import heapq
import time
import json
import threading
import requests
import config

# --- [공통: 기억 객체] ---
class MemoryObject:
    def __init__(self, content, importance=1.0):
        self.content = content
        self.timestamp = time.time()
        self.importance = importance # 1.0 ~ 10.0
        self.access_count = 1

    def calculate_retention_score(self):
        """
        우선순위 큐 정렬 기준: (점수가 낮을수록 먼저 방출됨)
        Score = (중요도 * 0.7) + (최신성 * 0.3)
        """
        time_diff = time.time() - self.timestamp
        recency = 1.0 / (1.0 + time_diff) # 시간이 지날수록 0에 수렴
        return (self.importance * 0.7) + (recency * 0.3)

    # heapq가 객체를 비교할 수 있도록 비교 연산자 오버로딩
    def __lt__(self, other):
        return self.calculate_retention_score() < other.calculate_retention_score()

# --- [Model 1: Persona Model (Fast Path)] ---
class WorkingMemory:
    """
    단기 기억(STM)을 우선순위 큐로 관리.
    용량이 차면 '가장 덜 중요한' 기억을 방출(Evict)하여 Transfer Buffer로 보냄.
    """
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.memory_queue = [] # Min-Heap (우선순위 큐)
        self.transfer_buffer = [] # 장기 기억으로 넘어갈 후보군 (Overflow)

    def add_memory(self, content, importance=1.0):
        mem = MemoryObject(content, importance)
        
        # 큐에 삽입 (Score 기준으로 자동 정렬됨)
        heapq.heappush(self.memory_queue, mem)
        
        # 용량 초과 시 방출 (Eviction)
        if len(self.memory_queue) > self.capacity:
            # 점수가 가장 낮은(가장 오래됐거나 중요하지 않은) 기억이 pop됨
            evicted_mem = heapq.heappop(self.memory_queue)
            self.transfer_buffer.append(evicted_mem)
            print(f"👋 [Evicted to Buffer] '{evicted_mem.content}' (Score 낮은 기억 방출)")

    def get_context(self):
        """현재 작업 기억에 남아있는(살아남은) 기억들 반환"""
        # 힙 구조를 깨지 않고 리스트로 변환하여 반환
        return [m.content for m in sorted(self.memory_queue, reverse=True)]

class PersonaModel:
    def __init__(self):
        self.stm = WorkingMemory(capacity=5) # 테스트용 작은 용량

    def chat(self, user_input):
        """
        사용자와 대화 (실시간).
        대화 내용은 일단 STM에 넣고, 중요도 판단은 약식으로 하거나 별도 모듈 사용.
        """
        # 1. 입력 처리 및 중요도 평가 (여기선 임시로 길이기반 중요도)
        importance = 5.0 if "중요" in user_input else 1.0
        self.stm.add_memory(f"User: {user_input}", importance)
        
        # 2. 답변 생성 (Context: 현재 STM에 살아남은 기억들)
        context = self.stm.get_context()
        # (실제론 여기서 LLM 호출)
        response = f"Simulated Response (Context size: {len(context)})"
        
        # 3. 본인 답변도 기억
        self.stm.add_memory(f"Bot: {response}", 1.0)
        return response

# --- [Model 2: Memory Management Model (Slow Path)] ---
class LongTermMemoryManager:
    """
    주기적으로 호출되어 Transfer Buffer에 쌓인 '밀려난 기억들'을 처리.
    LLM을 사용하여 '요약 저장' 할지 '완전 망각' 할지 결정.
    """
    def __init__(self, db_path="ltm_storage.json"):
        self.db_path = db_path
        self.ltm_data = self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"facts": [], "episodes": []}

    def consolidate_memory(self, buffer_items):
        """
        [핵심 로직]
        방출된 기억들(buffer_items)을 모아서 LLM에게 판단 요청.
        """
        if not buffer_items:
            return

        print(f"🔄 [Consolidation Start] 방출된 기억 {len(buffer_items)}개 처리 중...")
        
        # 분석할 텍스트 묶음
        text_chunk = "\n".join([m.content for m in buffer_items])
        
        # --- LLM API 호출 (Memory Model용 별도 프롬프트) ---
        # 실제로는 config.LLM_API_KEY 등을 사용
        prompt = f"""
        너는 기억 관리자다. 아래는 단기 기억 용량 초과로 인해 '밀려난' 대화 조각들이다.
        이 중에서 장기적으로 기억할 가치가 있는 '사실(Fact)'이나 '중요 사건(Episode)'이 있다면 요약해라.
        쓸데없는 잡담(인사, 리액션 등)은 무시해라.
        
        [입력 데이터]
        {text_chunk}
        
        [출력 형식 JSON]
        {{ "facts": ["유저는 사과를 좋아함"], "episodes": ["2024-01-20에 싸웠음"] }}
        """
        
        # (Mocking LLM Response for demo)
        # 실제 구현시: requests.post(...)
        print("   Thinking... (LLM 호출)")
        simulated_result = {"facts": [], "episodes": []}
        if "중요" in text_chunk:
            simulated_result["facts"].append("중요한 정보가 발견됨")
        
        # DB 업데이트
        if simulated_result["facts"] or simulated_result["episodes"]:
            self.ltm_data["facts"].extend(simulated_result["facts"])
            self.ltm_data["episodes"].extend(simulated_result["episodes"])
            self._save_db()
            print(f"   ✅ 장기 기억 저장 완료: {simulated_result}")
        else:
            print("   🗑️ 저장할 가치 없음. 완전 망각.")

    def _save_db(self):
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.ltm_data, f, ensure_ascii=False, indent=2)

# --- [시스템 통합 실행 예시] ---
if __name__ == "__main__":
    persona = PersonaModel()
    memory_manager = LongTermMemoryManager()

    # 1. 대화 시뮬레이션 (Fast Model)
    # 큐 용량(5)을 넘어가면 자동으로 buffer로 밀려남
    inputs = [
        "안녕", "오늘 날씨 어때?", "나 배고파", 
        "근데 나 사실 딸기 알러지 있어 (중요)", # 중요도 높음
        "아 진짜?", "응 진짜야", "슬프네", "점심 뭐 먹지"
    ]

    print("--- 🗣️ 대화 시작 (Persona Model) ---")
    for txt in inputs:
        persona.chat(txt)
        time.sleep(0.1) # 시간차 시뮬레이션

    # 2. 주기적 메모리 관리 실행 (Slow Model)
    # 실제로는 스케줄러나 별도 스레드가 수행
    print("\n--- 🧠 메모리 정리 시작 (Memory Model) ---")
    
    # STM의 오버플로우 버퍼를 가져와서 처리
    buffer_snapshot = persona.stm.transfer_buffer[:] # 복사
    persona.stm.transfer_buffer = [] # 버퍼 비우기
    
    memory_manager.consolidate_memory(buffer_snapshot)