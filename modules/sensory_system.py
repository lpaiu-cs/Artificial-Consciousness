from typing import List, Dict, Any
import time
import re
from memory_structures import MemoryObject
from api_client import UnifiedAPIClient

class SensorySystem:
    """
    [Sensory Queue]
    Raw 채팅 로그를 받아 '화자(Speaker)' 단위로 메시지를 병합(Chunking)합니다.
    
    3-Tier Reflection:
    - 여기서 생성된 MemoryObject는 추후 LTM의 'EpisodeNode'가 됩니다.
    - 추출된 user_id와 user_name은 추후 'EntityNode'를 갱신하거나 연결하는 데 쓰입니다.
    
    TODO: 청킹 관련 로직 고도화
    """
    def __init__(self, api_client: UnifiedAPIClient, time_threshold: float = 30.0):
        self.api = api_client
        self.time_threshold = time_threshold

    def process_input(self, history: List[Dict], current_msg: Dict) -> List[MemoryObject]:
        """
        Input: Raw Logs (History + Current)
        Output: Chunked MemoryObjects (Ready for STM & LTM)
        """
        if not history and not current_msg:
            return []

        # 1. 전체 로그 병합
        raw_logs = history + [current_msg] if current_msg else history
        chunked_memories = []
        
        if not raw_logs:
            return chunked_memories

        # 2. 청킹 초기화 (첫 메시지 기준)
        first_log = raw_logs[0]
        current_chunk = self._init_chunk(first_log)

        # 3. 루프 돌며 병합
        for i in range(1, len(raw_logs)):
            log = raw_logs[i]
            uid = str(log.get("user_id"))
            
            # (조건 1) 화자가 같고
            # (조건 2) 시간 차이가 크지 않으면 -> 병합
            time_diff = log.get("timestamp", time.time()) - current_chunk["timestamp"]
            is_same_speaker = (uid == current_chunk["user_id"])
            is_continuous = (time_diff < self.time_threshold)

            if is_same_speaker and is_continuous:
                current_chunk["content"].append(log.get("msg", ""))
                # (옵션) 닉네임이 중간에 바뀌었을 수도 있으니 최신 걸로 갱신
                current_chunk["user_name"] = log.get("user_name", "Unknown")
            else:
                # 청크 마감(Commit) 및 새 청크 시작
                self._commit_chunk(chunked_memories, current_chunk)
                current_chunk = self._init_chunk(log)

        # 마지막 청크 저장
        self._commit_chunk(chunked_memories, current_chunk)
        
        # 4. 의미 부여 (Embedding)
        # STM 및 LTM 검색을 위한 벡터화
        chunked_memories = self._enrich_chunks_with_embedding(chunked_memories)
        
        return chunked_memories

    def _init_chunk(self, log: Dict) -> Dict:
        """청크 딕셔너리 초기화 헬퍼"""
        return {
            "content": [log.get("msg", "")],
            "user_id": str(log.get("user_id")),
            "user_name": log.get("user_name", "Unknown"),
            "role": log.get("role", "user"), # user vs assistant
            "timestamp": log.get("timestamp", time.time()),
            # [3-Tier] Entity 연결을 위한 기초 데이터
            "mentions": self._extract_mentions(log.get("msg", "")) 
        }

    def _commit_chunk(self, storage: List[MemoryObject], chunk_data: Dict):
        """MemoryObject 변환 및 저장"""
        full_text = " ".join(chunk_data["content"]).strip()
        if not full_text:
            return

        # [3-Tier Logic] 관련 유저 식별
        # 기본적으로 화자(Speaker)는 당연히 관련자
        related_users = [chunk_data["user_id"]]
        # 멘션된 유저가 있다면 추가 (예: @12345) - 필요 시 로직 확장
        # related_users.extend(chunk_data["mentions"]) 

        mem = MemoryObject(
            content=full_text,
            role=chunk_data["role"],
            user_id=chunk_data["user_id"],   # EntityNode 식별 키
            user_name=chunk_data["user_name"], # EntityNode 닉네임 갱신용
            timestamp=chunk_data["timestamp"],
            activation=50.0,
            related_users=list(set(related_users)) # 중복 제거
        )
        storage.append(mem)

    def _enrich_chunks_with_embedding(self, chunks: List[MemoryObject]) -> List[MemoryObject]:
        """각 청크의 텍스트를 벡터로 변환 (Batch 처리 가능하면 더 좋음)"""
        for chunk in chunks:
            # 빈 텍스트 방지
            if chunk.content:
                chunk.embedding = self.api.get_embedding(chunk.content)
            else:
                chunk.embedding = [0.0] * 1536 # Mock size
        return chunks

    def _extract_mentions(self, text: str) -> List[str]:
        """
        (Optional) 텍스트에서 멘션된 유저 ID 등을 추출
        현재는 플레이스홀더. 나중에 '@user_id' 같은 형식을 쓴다면 여기서 파싱.
        """
        return []