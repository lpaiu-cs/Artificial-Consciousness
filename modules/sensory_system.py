import time
from typing import List, Dict
from memory_structures import MemoryObject

class SensorySystem:
    """
    [Perception Layer]
    채팅 로그를 '의미 단위(Chunk)'로 병합하여 MemoryObject 리스트로 변환합니다.
    """
    def __init__(self, time_threshold: float = 30.0):
        self.time_threshold = time_threshold

    def process_input(self, history: List[Dict], current_msg: Dict) -> List[MemoryObject]:
        """Raw Log -> Chunked MemoryObjects"""
        if not history and not current_msg:
            return []

        # 1. 로그 합치기
        raw_logs = history + [current_msg] if current_msg else history
        if not raw_logs:
            return []

        chunked_memories = []
        
        # 2. 첫 번째 청크 초기화
        first_log = raw_logs[0]
        current_chunk = {
            "content": [str(first_log.get("msg", ""))],
            "user_id": str(first_log.get("user_id")),
            "user_name": str(first_log.get("user_name", "Unknown")),
            "role": "user", # 편의상 user로 통일 (봇 로그라면 assistant 처리 로직 필요)
            "timestamp": time.time() 
        }

        # 3. 순회하며 병합 (Chunking Loop)
        for i in range(1, len(raw_logs)):
            log = raw_logs[i]
            uid = str(log.get("user_id"))
            msg = str(log.get("msg", ""))
            
            # 같은 유저가 연속으로 말했으면 병합
            if uid == current_chunk["user_id"]:
                current_chunk["content"].append(msg)
            else:
                # 화자가 바뀌면 저장(Commit)
                self._commit_chunk(chunked_memories, current_chunk)
                # 새 청크 시작
                current_chunk = {
                    "content": [msg],
                    "user_id": uid,
                    "user_name": str(log.get("user_name", "Unknown")),
                    "role": "user",
                    "timestamp": time.time()
                }

        # 마지막 청크 저장
        self._commit_chunk(chunked_memories, current_chunk)
        
        return chunked_memories

    def _commit_chunk(self, storage: List[MemoryObject], chunk_data: Dict):
        full_text = " ".join(chunk_data["content"]).strip()
        if full_text:
            mem = MemoryObject(
                content=full_text,
                role=chunk_data["role"],
                user_id=chunk_data["user_id"],
                user_name=chunk_data["user_name"],
                timestamp=chunk_data["timestamp"]
            )
            storage.append(mem)