import heapq
import time
import numpy as np
from typing import List, Set, Optional
from memory_structures import MemoryObject
import config

class WorkingMemory:
    """
    [Working Memory (STM)]
    - 역할: 현재 대화의 맥락(Context)을 유지하는 단기 기억 저장소.
    - 구조: Priority Queue (Min-Heap based on Activation Score).
    - 특징: 'Vector-based Semantic Attention'을 통해 중요 기억의 수명을 연장.
    """
    
    def __init__(self):
        self.capacity = config.STM_CAPACITY
        self.memory_queue: List[MemoryObject] = [] # Heap
        self.eviction_buffer: List[MemoryObject] = [] # LTM으로 보낼 후보군

    def inject_memories(self, memories: List[MemoryObject]):
        """
        [Input] 감각 시스템에서 넘어온 새로운 청크들을 STM에 주입
        """
        
        for mem in memories:
            # 신규 기억은 초기 활성도가 높음 (config.py 참조)
            # 힙에 넣기 위해 (activation, timestamp, object) 튜플이 아닌
            # MemoryObject 자체를 넣고 __lt__ 등을 정의하거나,
            # 여기서는 간단히 리스트 관리 후 heapify 방식을 사용.
            
            if not mem.timestamp:
                mem.timestamp = time.time()
            
            heapq.heappush(self.memory_queue, mem) # activation 기준 heap
            
        # 주입 직후 용량 초과 시, 즉시 정리하지 않고
        # update_activations 단계에서 "점수 계산 후" 방출하는 것이 더 스마트함.
        # 다만, 너무 많이 쌓이면 메모리 낭비니 Soft Limit 체크
        if len(self.memory_queue) > self.capacity * 2:
            self._enforce_capacity()

    def update_activations(self, query_vector: List[float]):
        """
        [Attention Mechanism]
        현재 대화의 주제(Query Vector)와 STM 내부 기억들의 '유사도'를 계산하여
        관련된 기억은 강화(Boost)하고, 무관한 기억은 약화(Decay)시킵니다.
        """
        if not query_vector:
            return

        q_vec = np.array(query_vector)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return

        new_queue = []
        
        # 힙을 비우면서 하나씩 꺼내서 검사
        while self.memory_queue:
            mem = heapq.heappop(self.memory_queue)
            
            # 1. 유사도 계산 (Cosine Similarity)
            similarity = 0.0
            if mem.embedding:
                m_vec = np.array(mem.embedding)
                m_norm = np.linalg.norm(m_vec)
                if m_norm > 0:
                    similarity = float(np.dot(q_vec, m_vec) / (q_norm * m_norm))
            
            # 2. 점수 조정 (Reinforcement)
            # 유사도 0.5 이상이면 "관련 있음"으로 간주 (임계값은 조정 가능)
            if similarity >= 0.5:
                # 관련성 높을수록 더 큰 Boost
                boost = config.BOOST_SCORE * (1.0 + similarity)
                mem.activation += boost
                # 상한선 (Max Activation Cap)
                mem.activation = min(100.0, mem.activation)
            else:
                mem.activation += config.PENALTY_SCORE
            
            # 3. 시간 감쇠 (Time Decay)
            # 오래된 기억일수록 점수 하락
            mem.activation -= config.TIME_DECAY
            
            # 4. 생존 여부 결정 (Threshold Check)
            if mem.activation >= config.MIN_ACTIVATION:
                new_queue.append(mem)
            else:
                # 점수가 너무 낮으면 즉시 방출 (망각) -> LTM 저장 후보
                self.eviction_buffer.append(mem)
        
        # 5. 큐 재구성
        self.memory_queue = new_queue
        heapq.heapify(self.memory_queue)
        
        # 6. 용량 초과분 강제 방출 (Hard Limit)
        self._enforce_capacity()

    def get_chronological_context(self) -> List[MemoryObject]:
        """
        [For LLM Prompt]
        STM에 있는 기억들을 '시간 순서'대로 정렬하여 반환합니다.
        (힙 구조는 순서가 섞여 있으므로 정렬 필요)
        """
        # timestamp 기준으로 오름차순 정렬
        return sorted(self.memory_queue, key=lambda x: x.timestamp)

    def _enforce_capacity(self):
        """용량을 초과한 경우, 활성도가 가장 낮은 기억부터 방출"""
        while len(self.memory_queue) > self.capacity:
            # 힙은 최소값이 루트(0번 인덱스)에 있으므로 pop하면 가장 점수 낮은 놈이 나옴
            evicted_mem = heapq.heappop(self.memory_queue)
            self.eviction_buffer.append(evicted_mem)