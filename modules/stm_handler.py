import heapq
import numpy as np
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

    def update_activations(self, query_vector: List[float]):
        """
        [Semantic Attention]
        텍스트 매칭 대신, 현재 쿼리 벡터와 STM 기억들 간의 
        '의미적 유사도'를 계산하여 강화(Boost)함.
        """
        new_queue = []
        q_vec = np.array(query_vector)
        q_norm = np.linalg.norm(q_vec)

        while self.memory_queue:
            mem = heapq.heappop(self.memory_queue)
            
            similarity = 0.0
            if mem.embedding and q_norm > 0:
                m_vec = np.array(mem.embedding)
                m_norm = np.linalg.norm(m_vec)
                if m_norm > 0:
                    similarity = np.dot(q_vec, m_vec) / (q_norm * m_norm)
            
            # [판단 로직] 유사도가 일정 수준 이상이면 "관련된 기억"으로 간주
            # 예: 0.6 이상이면 의미적으로 꽤 가까움
            if similarity >= 0.6: 
                # 유사도가 높을수록 더 많이 강화
                boost = config.BOOST_SCORE * (1.0 + similarity) 
                mem.activation += boost
                # (옵션) 활성도 상한선 캡
                mem.activation = min(100.0, mem.activation)
            else:
                mem.activation += config.PENALTY_SCORE
            
            # 시간 감쇠
            mem.activation -= config.TIME_DECAY
            
            # Eviction 결정
            if mem.activation >= config.MIN_ACTIVATION:
                new_queue.append(mem)
            else:
                self.eviction_buffer.append(mem)
        
        self.memory_queue = new_queue
        heapq.heapify(self.memory_queue)
        
        # Capacity Control
        while len(self.memory_queue) > self.capacity:
            evicted = heapq.heappop(self.memory_queue)
            self.eviction_buffer.append(evicted)

    def get_chronological_context(self) -> List[MemoryObject]:
        return sorted(self.memory_queue, key=lambda x: x.timestamp)