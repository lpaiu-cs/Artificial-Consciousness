from typing import List, Dict
import time
from memory_structures import MemoryObject

class SensorySystem:
    """
    [Sensory Queue]
    Raw 채팅 로그를 받아 '화자(Speaker)' 단위로 메시지를 병합(Chunking)합니다.
    짧게 끊어 친 카톡 스타일의 대화를 완성된 문장 형태로 STM에 전달합니다.
    """
    def __init__(self, time_threshold: float = 30.0):
        # 30초 이내의 연속 발화는 같은 맥락으로 간주
        self.time_threshold = time_threshold

    def process_input(self, history: List[Dict], current_msg: Dict) -> List[MemoryObject]:
        """
        history + current_msg를 합쳐서 의미 단위의 MemoryObject 리스트로 변환
        """
        if not history and not current_msg:
            return []

        # 1. 전체 로그 병합 (과거 + 현재)
        raw_logs = history + [current_msg] if current_msg else history
        
        chunked_memories = []
        if not raw_logs:
            return chunked_memories

        # 2. 청킹 로직 (Chunking Logic)
        # 첫 번째 메시지로 버퍼 초기화
        current_chunk = {
            "content": [raw_logs[0].get("msg", "")],
            "user_id": str(raw_logs[0].get("user_id")),
            "user_name": raw_logs[0].get("user_name", "Unknown"),
            "role": "user", # 봇인지 유저인지 구분 필요 (여기선 편의상 user)
            "timestamp": time.time() # 실제로는 로그의 시간을 써야 함
        }

        for i in range(1, len(raw_logs)):
            log = raw_logs[i]
            uid = str(log.get("user_id"))
            msg = log.get("msg", "")
            
            # 같은 유저가 연속으로 말했다면 병합
            if uid == current_chunk["user_id"]:
                current_chunk["content"].append(msg)
            else:
                # 화자가 바뀌면 기존 청크 저장(Commit) 후 새 청크 시작
                self._commit_chunk(chunked_memories, current_chunk)
                current_chunk = {
                    "content": [msg],
                    "user_id": uid,
                    "user_name": log.get("user_name", "Unknown"),
                    "role": "user", # 추후 role 구분 로직 필요
                    "timestamp": time.time()
                }

        # 마지막 청크 저장
        self._commit_chunk(chunked_memories, current_chunk)
        
        return chunked_memories

    def _commit_chunk(self, storage, chunk_data):
        """리스트에 MemoryObject로 변환하여 저장"""
        full_text = " ".join(chunk_data["content"]).strip()
        if full_text:
            mem = MemoryObject(
                content=full_text,
                role=chunk_data["role"],
                user_id=chunk_data["user_id"],
                user_name=chunk_data["user_name"],
                timestamp=chunk_data["timestamp"],
                activation=50.0 # 초기 활성화 점수 (신규 기억은 생생함)
            )
            storage.append(mem)