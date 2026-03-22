from typing import List, Dict
import hashlib
import hmac
import os
import time
import json
from cogbot.memory_structures import MemoryObject
from cogbot.api_client import UnifiedAPIClient
from cogbot import config

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
        self.cursor_path = getattr(config, "SENSORY_CURSOR_PATH", "sensory_seen_logs.json")
        self.max_seen_entries = int(getattr(config, "SENSORY_CURSOR_MAX_ENTRIES", 50000))
        self.cursor_hmac_key = getattr(config, "SENSORY_CURSOR_HMAC_KEY", "")
        self._seen_log_keys: Dict[str, float] = {}
        self._load_cursor()

    def process_input(self, history: List[Dict], current_msg: Dict) -> List[MemoryObject]:
        """
        Input: Raw Logs (History + Current)
        Output: Chunked MemoryObjects (Ready for STM & LTM)
        """
        # 1. 전체 로그 병합 후 아직 처리하지 않은 delta만 남김
        raw_logs = history + [current_msg] if current_msg else history
        raw_logs = self._filter_new_logs(raw_logs)
        if not raw_logs:
            return []
        raw_logs.sort(key=lambda x: x.get("timestamp", 0.0))  # 타임스탬프 기준  정렬

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
        if log.get("role") == "assistant" or uid in {"bot", "assistant"} or (self.bot_id and uid == self.bot_id):
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

    def _filter_new_logs(self, logs: List[Dict]) -> List[Dict]:
        """이전 턴에 이미 처리한 로그는 건너뛰어 history 전체 재주입을 막는다."""
        delta_logs = []
        cursor_updated = False
        for log in logs:
            if not self._is_valid_log(log):
                continue

            key = self._make_log_key(log)
            if key in self._seen_log_keys:
                self._touch_seen_key(key)
                continue

            self._touch_seen_key(key)
            cursor_updated = True
            delta_logs.append(log)

        if cursor_updated:
            self._save_cursor()
        return delta_logs

    def _make_log_key(self, log: Dict) -> str:
        message_id = self._extract_message_id(log)
        if message_id is not None:
            return f"msgid:{str(log.get('user_id', ''))}:{message_id}"

        payload = json.dumps(
            [
                str(log.get("user_id", "")),
                log.get("timestamp", 0.0),
                log.get("msg", "").strip(),
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if self.cursor_hmac_key:
            digest = hmac.new(
                self.cursor_hmac_key.encode("utf-8"),
                payload,
                hashlib.sha256,
            ).hexdigest()
            return f"hmac-sha256:{digest}"
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    def _extract_message_id(self, log: Dict):
        for key in ("message_id", "msg_id", "id"):
            value = log.get(key)
            if isinstance(value, (str, int, float)) and value != "":
                return str(value)
        return None

    def _touch_seen_key(self, key: str):
        if key in self._seen_log_keys:
            self._seen_log_keys.pop(key, None)
        self._seen_log_keys[key] = time.time()
        while len(self._seen_log_keys) > max(self.max_seen_entries, 0):
            oldest_key = next(iter(self._seen_log_keys))
            self._seen_log_keys.pop(oldest_key, None)

    def _load_cursor(self):
        try:
            if not os.path.exists(self.cursor_path):
                return
            with open(self.cursor_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            self._seen_log_keys = {}
            return

        keys = payload.get("keys", []) if isinstance(payload, dict) else payload
        if not isinstance(keys, list):
            self._seen_log_keys = {}
            return

        self._seen_log_keys = {}
        for key in keys[-max(self.max_seen_entries, 0):]:
            if not isinstance(key, str):
                continue
            self._seen_log_keys[key] = time.time()

    def _save_cursor(self):
        parent = os.path.dirname(os.path.abspath(self.cursor_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {
            "keys": list(self._seen_log_keys.keys())[-max(self.max_seen_entries, 0):],
            "updated_at": time.time(),
        }
        try:
            with open(self.cursor_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
        except OSError:
            return

    def _extract_mentions(self, text: str) -> List[str]:
        """
        (Optional) 텍스트에서 멘션된 유저 ID 등을 추출
        현재는 플레이스홀더. 나중에 '@user_id' 같은 형식을 쓴다면 여기서 파싱.
        """
        return []
