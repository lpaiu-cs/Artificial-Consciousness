"""
CogBot Test Configuration
- Fixtures for mocking external APIs
- Shared test utilities
"""
import sys
import os
import json
from unittest.mock import MagicMock, patch
from typing import List
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from cryptography.fernet import Fernet


# =============================================================================
# Mock Classes
# =============================================================================

class MockUnifiedAPIClient:
    """Mock API Client for testing without actual API calls"""
    
    def __init__(self, enable_logging: bool = None):
        self._embedding_counter = 0
        # Mock logger (no-op)
        self.logger = type('MockLogger', (), {
            'set_enabled': lambda self, x: None,
            'log_embedding_request': lambda self, *args: None,
            'log_embedding_response': lambda self, *args, **kwargs: None,
            'log_chat_request': lambda self, *args, **kwargs: None,
            'log_chat_response': lambda self, *args, **kwargs: None,
        })()
    
    def set_logging(self, enabled: bool):
        """로깅 on/off (테스트에서는 no-op)"""
        pass
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate deterministic mock embeddings based on text content.
        Returns a 1536-dimensional vector (OpenAI embedding size).
        """
        if not text:
            return [0.0] * 1536
        
        # Create a simple deterministic embedding based on text hash
        import hashlib
        text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
        
        # Generate normalized vector
        import numpy as np
        np.random.seed(text_hash % (2**32 - 1))
        vec = np.random.randn(1536)
        vec = vec / np.linalg.norm(vec)
        
        return vec.tolist()
    
    def chat_fast(self, system_prompt: str, user_prompt: str) -> str:
        """Mock fast LLM response (System 1 - Groq)"""
        return f"[MOCK_FAST] Processed: {user_prompt[:50]}..."
    
    def chat_slow(self, system_prompt: str, user_prompt: str, json_mode: bool = False):
        """Mock slow LLM response (System 2 - GPT-4)"""
        if json_mode:
            return {
                "episode_summary": "Mock episode summary",
                "dominant_emotion": "neutral",
                "claims": [
                    {
                        "subject_id": "user_001",
                        "facet": "interaction.preference",
                        "value": {"dimension": "language", "value": "ko"},
                        "qualifiers": {},
                        "source_type": "explicit",
                        "confidence": 0.91,
                        "status": "active",
                        "nl_summary": "사용자는 한국어 답변을 선호한다."
                    },
                    {
                        "subject_id": "user_001",
                        "facet": "commitment.open_loop",
                        "value": {"kind": "followup_needed", "text": "다음에 다시 알려주기", "priority": 7},
                        "qualifiers": {},
                        "source_type": "explicit",
                        "confidence": 0.88,
                        "status": "active",
                        "nl_summary": "다음에 다시 알려줘야 하는 열린 루프가 있다."
                    }
                ],
                "notes": [
                    {
                        "note_type": "impression",
                        "summary": "민트초코 이야기를 즐거워했다.",
                        "tags": ["legacy_mood"],
                        "confidence": 0.73,
                        "related_entity_ids": ["user_001"]
                    }
                ],
                "insights": ["Mock insight 1"]
            }
        # Return a response with emotion tag for social module testing
        return "[MOCK_SLOW] This is a mock response. [FEELING:calm contentment]"


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_api():
    """Provides a mock API client"""
    return MockUnifiedAPIClient()


@pytest.fixture
def temp_graph_files(tmp_path):
    """Provides temporary file paths for graph storage"""
    return {
        "graph_path": str(tmp_path / "test_graph.json"),
        "embeddings_path": str(tmp_path / "test_embeddings.json"),
    }


@pytest.fixture
def temp_canonical_db(tmp_path):
    """Provides a temporary SQLite path for canonical memory."""
    return str(tmp_path / "memory_state.sqlite3")


@pytest.fixture
def canonical_keyring(monkeypatch, tmp_path):
    """Configure a temporary canonical payload keyring for reproducible tests."""
    keyring_path = tmp_path / "canonical_payload_keys.json"

    def configure(keys: dict | None = None, active_key_id: str = "local-v1", auto_reencrypt: bool = False):
        effective_keys = keys or {active_key_id: Fernet.generate_key().decode("utf-8")}
        payload = {
            "active_key_id": active_key_id,
            "keys": {str(key_id): str(key_value) for key_id, key_value in effective_keys.items()},
        }
        keyring_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "CANONICAL_ACTIVE_KEY_ID", active_key_id, raising=False)
        monkeypatch.setattr(config, "CANONICAL_ENCRYPTION_KEYS_JSON", "", raising=False)
        monkeypatch.setattr(config, "CANONICAL_ENCRYPTION_KEYRING_PATH", str(keyring_path), raising=False)
        monkeypatch.setattr(config, "CANONICAL_ENCRYPTION_KEY", "", raising=False)
        monkeypatch.setattr(config, "CANONICAL_ENCRYPTION_KEY_ID", active_key_id, raising=False)
        monkeypatch.setattr(config, "CANONICAL_ENCRYPTION_KEY_PATH", "", raising=False)
        monkeypatch.setattr(config, "CANONICAL_AUTO_REENCRYPT_PROTECTED_PAYLOADS", auto_reencrypt, raising=False)
        return {
            "path": str(keyring_path),
            "active_key_id": active_key_id,
            "keys": effective_keys,
        }

    return configure


@pytest.fixture
def sample_chat_history():
    """Provides sample chat history for testing"""
    import time
    base_time = time.time()
    
    return [
        {
            "user_id": "user_001",
            "user_name": "테스터",
            "msg": "안녕하세요! 처음 인사드립니다.",
            "role": "user",
            "timestamp": base_time
        },
        {
            "user_id": "user_001",
            "user_name": "테스터",
            "msg": "날씨가 좋네요.",
            "role": "user",
            "timestamp": base_time + 5
        },
        {
            "user_id": "bot",
            "user_name": "CogBot",
            "msg": "안녕하세요! 정말 좋은 날씨네요.",
            "role": "assistant",
            "timestamp": base_time + 10
        },
    ]


@pytest.fixture
def sample_calling_message():
    """Provides a sample calling message for testing"""
    import time
    return {
        "user_id": "user_001",
        "user_name": "테스터",
        "msg": "오늘 뭐해?",
        "role": "user",
        "timestamp": time.time()
    }


@pytest.fixture
def sample_memory_objects():
    """Provides sample MemoryObject instances for testing"""
    from memory_structures import MemoryObject
    import time
    
    return [
        MemoryObject(
            content="첫 번째 테스트 메모리",
            role="user",
            user_id="user_001",
            user_name="테스터",
            timestamp=time.time(),
            activation=50.0
        ),
        MemoryObject(
            content="두 번째 테스트 메모리",
            role="user",
            user_id="user_001",
            user_name="테스터",
            timestamp=time.time() + 1,
            activation=60.0
        ),
        MemoryObject(
            content="세 번째 테스트 메모리 - 관련 주제",
            role="user",
            user_id="user_001",
            user_name="테스터",
            timestamp=time.time() + 2,
            activation=70.0
        ),
    ]
