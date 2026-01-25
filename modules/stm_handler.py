import heapq
from typing import List, Set
from memory_structures import MemoryObject
import config

class WorkingMemory:
    """작업 기억 (Priority Queue based on Activation)"""
    
    def __init__(self):
        self.capacity = config.STM_CAPACITY
        self.memory_queue = [] # Min-Heap
        self.eviction_buffer = []

    def inject_memories(self, memories: List[MemoryObject]):
        for mem in memories:
            heapq.heappush(self.memory_queue, mem)
            
        # Soft Limit: 넘치면 점수 깎기 전이라도 일단 Eviction Buffer로 보낼 수 있음
        # 여기서는 update_activations 호출 시 정리하도록 둠

    def update_activations(self, referenced_ids: Set[str]):
        """참조된 기억 강화, 아닌 기억 약화, 기준 미달 방출"""
        new_queue = []
        
        while self.memory_queue:
            mem = heapq.heappop(self.memory_queue)
            
            # 1. Score Update
            if mem.mem_id in referenced_ids:
                mem.activation += config.BOOST_SCORE
                mem.activation = min(100.0, mem.activation)
            else:
                mem.activation += config.PENALTY_SCORE
            
            mem.activation -= config.TIME_DECAY
            
            # 2. Filter
            if mem.activation >= config.MIN_ACTIVATION:
                new_queue.append(mem)
            else:
                self.eviction_buffer.append(mem)
        
        self.memory_queue = new_queue
        heapq.heapify(self.memory_queue)
        
        # 3. Capacity Control
        while len(self.memory_queue) > self.capacity:
            evicted = heapq.heappop(self.memory_queue)
            self.eviction_buffer.append(evicted)

    def get_chronological_context(self) -> List[MemoryObject]:
        return sorted(self.memory_queue, key=lambda x: x.timestamp)