from typing import List, Dict
import time
import json
from memory_structures import MemoryObject
from api_client import UnifiedAPIClient
import config

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
        self.bot_id = str(getattr(config, 'BOT_USER_ID', ''))

    def process_input(self, history: List[Dict], current_msg: Dict) -> List[MemoryObject]:
        """
        Input: Raw Logs (History + Current)
        Output: Chunked MemoryObjects (Ready for STM & LTM)
        """
        # 1. 전체 로그 병합
        raw_logs = history + [current_msg] if current_msg else history
        if not raw_logs:
            return []

        chunked_memories = []

        # 첫 번째 유효한 로그를 찾을 때까지 스킵 (시스템 메시지 등 제외)
        start_idx = 0
        while start_idx < len(raw_logs):
            if not self._is_valid_log(raw_logs[start_idx]):
                start_idx += 1
                continue
            break
        
        if start_idx >= len(raw_logs):
            return []
        
        # 2. 첫 청크 초기화
        first_log = raw_logs[start_idx]
        current_chunk = self._init_chunk(first_log)

        # 3. 루프 돌며 병합
        for i in range(start_idx + 1, len(raw_logs)):
            log = raw_logs[i]

            # (A) 노이즈 필터링
            if not self._is_valid_log(log):
                continue

            uid = str(log.get("user_id"))
            
            # (B) 병합 조건 확인
            # 1. 화자가 같은가?
            # 2. 시간 차이가 적은가?
            time_diff = log.get("timestamp", time.time()) - current_chunk["timestamp"]
            is_same_speaker = (uid == current_chunk["user_id"])
            is_continuous = (time_diff < self.time_threshold)

            if is_same_speaker and is_continuous:
                current_chunk["content"].append(log.get("msg", ""))
                # 닉네임이 중간에 바뀌었거나 정보가 더 정확하면 갱신
                if log.get("user_name"):
                    current_chunk["user_name"] = log.get("user_name")
            else:
                # 청크 마감(Commit) 및 새 청크 시작
                self._commit_chunk(chunked_memories, current_chunk)
                current_chunk = self._init_chunk(log)

        # 마지막 청크 저장
        self._commit_chunk(chunked_memories, current_chunk)
        
        # 4. 임베딩 추가 (Embedding)
        # STM 및 LTM 검색을 위한 벡터화
        chunked_memories = self._enrich_chunks_with_embedding(chunked_memories)
        
        return chunked_memories
    
    def _is_valid_log(self, log: Dict) -> bool:
        """
        [Filter Logic]
        처리해서는 안 되는 시스템 메시지나 노이즈를 걸러냅니다.
        TODO: 추후 user_id 0인것 이용 및 방 입출입 메시지 필터링을 통해 상위 api에서 걸러내도록 고도화 여지있음.
        """
        msg = log.get("msg", "")
        if not msg: 
            return False
            
        # JSON 형태의 시스템 로그 차단 (예: {"feedType":4...})
        if "feedType" in msg or (msg.strip().startswith("{") and "userId" in msg):
            return False
            
        return True

    def _init_chunk(self, log: Dict) -> Dict:
        """
        로그 하나를 기준으로 새 청크 딕셔너리를 생성합니다.
        이때 봇 자신인지 확인하여 Role과 Name을 강제합니다.
        """
        uid = str(log.get("user_id"))
        raw_name = log.get("user_name", "Unknown")
        
        # [Self-Recognition] 봇 자신의 메시지 처리
        if self.bot_id and uid == self.bot_id:
            role = "assistant"
            user_name = config.BOT_NAME  # 봇의 자아를 강제 주입
        else:
            role = "user"
            user_name = raw_name

        return {
            "content": [log.get("msg", "")],
            "user_id": uid,
            "user_name": user_name,
            "role": role,
            "timestamp": log.get("timestamp", time.time())
        }

    def _commit_chunk(self, storage, chunk_data):
        """청크를 MemoryObject로 변환하여 리스트에 추가"""
        full_text = " ".join(chunk_data["content"]).strip()
        if not full_text:
            return

        # [3-Tier Logic] 관련 유저 식별
        # 멘션된 유저가 있다면 추가 (예: @12345) - 필요 시 로직 확장
        # related_users.extend(chunk_data["mentions"]) 

        mem = MemoryObject(
            content=full_text,
            role=chunk_data["role"],
            user_id=chunk_data["user_id"],       # EntityNode 식별 키
            user_name=chunk_data["user_name"],   # EntityNode 닉네임 갱신용
            timestamp=chunk_data["timestamp"],
            activation=50.0,
            # [3-Tier] 화자 본인은 당연히 관련자
            related_users=[chunk_data["user_id"]] 
        )
        storage.append(mem)

    def _enrich_chunks_with_embedding(self, chunks: List[MemoryObject]) -> List[MemoryObject]:
        """각 청크의 텍스트를 벡터로 변환 (Batch 처리 가능하면 더 좋음)"""
        for chunk in chunks:
            # 빈 텍스트 방지
            if chunk.content:
                # 봇의 말(assistant)도 임베딩을 해야 '내가 무슨 말을 했는지' 문맥을 파악 가능
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