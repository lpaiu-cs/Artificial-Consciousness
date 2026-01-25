import heapq
import time
from typing import List, Set, Tuple
from memory_structures import MemoryObject

class WorkingMemory:
    """
    [System 1: STM]
    - Priority Queue로 관리 (Activation 기준)
    - 맥락 재구성에 사용된 기억은 강화(Boost)되고, 무시된 기억은 약화(Penalty)됩니다.
    - 일정 점수 이하로 떨어지면 Eviction 됩니다.
    """
    def __init__(self, capacity: int = 15):
        self.capacity = capacity
        self.memory_queue = [] # Min-Heap
        self.eviction_buffer = [] # 방출된 기억 보관소 (LTM 전달용)

        # 튜닝 파라미터 (Scorer Weights)
        self.BOOST_SCORE = 20.0       # 참조 시 가점
        self.PENALTY_SCORE = -5.0     # 비참조 시 감점
        self.TIME_DECAY = 2.0         # 턴당 자연 망각 점수
        self.MIN_THRESHOLD = 10.0     # 이 점수 밑이면 강제 퇴출

    def inject_memories(self, memories: List[MemoryObject]):
        """Sensory Queue에서 넘어온 청크들을 STM에 주입"""
        for mem in memories:
            # 새 기억은 가장 높은 우선순위로 관리하기 위해 힙에 넣음
            # 중복 방지 로직 필요 시 추가 (mem_id 기준)
            heapq.heappush(self.memory_queue, mem)
            
        # 용량 초과 시 단순 제거가 아니라, 나중에 update_activations 후 제거함
        # 여기서는 일단 넘침 허용

    def get_all_memories(self) -> List[MemoryObject]:
        """현재 STM에 있는 모든 기억 반환 (정렬 안됨)"""
        return self.memory_queue

    def update_activations(self, referenced_ids: Set[str]):
        """
        [Dynamic Scorer]
        맥락 재구성에 사용된(referenced) 기억들의 활성도를 갱신하고,
        점수가 낮은 기억들을 방출(Eviction)합니다.
        """
        new_queue = []
        
        while self.memory_queue:
            mem = heapq.heappop(self.memory_queue)
            
            # 1. 참조 여부에 따른 점수 변화
            if mem.mem_id in referenced_ids:
                mem.activation += self.BOOST_SCORE
                # 최대 점수 캡 (선택 사항)
                if mem.activation > 100.0: mem.activation = 100.0
            else:
                mem.activation += self.PENALTY_SCORE
            
            # 2. 시간 경과에 따른 자연 망각 (Time Decay)
            # (복잡한 ACT-R 수식 대신 단순 뺄셈 적용)
            mem.activation -= self.TIME_DECAY
            
            # 3. 생존 여부 결정 (Eviction Logic)
            if mem.activation >= self.MIN_THRESHOLD:
                # 살아남음 -> 새 큐에 추가
                new_queue.append(mem)
            else:
                # 죽음 -> Eviction Buffer로 이동 (LTM 저장 대상)
                self.eviction_buffer.append(mem)
        
        # 4. 큐 재구성 (Heapify)
        self.memory_queue = new_queue
        heapq.heapify(self.memory_queue)
        
        # 5. 용량 관리 (Soft Limit)
        # 점수로 걸러냈는데도 용량이 넘치면, 점수 꼴찌부터 강제 퇴출
        while len(self.memory_queue) > self.capacity:
            evicted = heapq.heappop(self.memory_queue)
            self.eviction_buffer.append(evicted)

    def get_chronological_context(self) -> List[MemoryObject]:
        """LLM 입력용: 시간순 정렬된 리스트 반환"""
        return sorted(self.memory_queue, key=lambda x: x.timestamp)