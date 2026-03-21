import json
import sqlite3
import time

import pytest

import config
from api_client import UnifiedAPIClient
from bot_orchestrator import BotOrchestrator
from memory.canonical_store import CanonicalMemoryStore
from memory.fast_path import FastPathMemoryWriter
from memory_structures import ClaimNode
from modules.ltm_graph import MemoryGraph
from tests.conftest import MockUnifiedAPIClient


class CountingMockAPI(MockUnifiedAPIClient):
    def __init__(self):
        super().__init__()
        self.embedding_call_count = 0

    def get_embedding(self, text: str):
        self.embedding_call_count += 1
        return super().get_embedding(text)


@pytest.fixture
def smoke_orchestrator(mock_api, temp_graph_files, temp_canonical_db):
    graph = MemoryGraph(
        graph_path=temp_graph_files["graph_path"],
        embeddings_path=temp_graph_files["embeddings_path"],
    )
    store = CanonicalMemoryStore(temp_canonical_db)
    bot = BotOrchestrator(api_client=mock_api, graph_db=graph, canonical_store=store)
    bot.reflector.stop()
    return bot


def test_model_eval_gate_matches_pinned_config():
    with open(config.MODEL_EVAL_GATE_PATH, "r", encoding="utf-8") as handle:
        gate = json.load(handle)

    assert gate["models"]["embedding"] == config.EMBEDDING_MODEL
    assert gate["models"]["smart"] == config.SMART_MODEL
    assert gate["models"]["fast"] == config.FAST_MODEL

    client = UnifiedAPIClient(enable_logging=False, exclude_embedding_log=True)
    assert client is not None


def test_boundary_payload_is_encrypted_and_runtime_behavior_is_split(smoke_orchestrator, temp_canonical_db):
    bot = smoke_orchestrator

    do_not_store_msg = {
        "user_id": "user_001",
        "user_name": "테스터",
        "msg": "내 병력 얘기는 저장하지 마",
        "role": "user",
        "timestamp": time.time(),
    }
    bot._run_slow_generation = lambda *args, **kwargs: ("알겠다. 병력 이야기는 저장하지 않겠다.", "차분함")
    response = bot.process_trigger([], do_not_store_msg)
    assert "병력" in response
    assistant_memories = [m.content for m in bot.stm.memory_queue if m.role == "assistant"]
    assert any("저장하지 않아야" in memory for memory in assistant_memories)
    assert all("병력" not in memory for memory in assistant_memories)

    with sqlite3.connect(temp_canonical_db) as conn:
        row = conn.execute(
            "SELECT value_json, qualifiers_json, encrypted_payload_blob FROM claims WHERE facet = 'boundary.rule' ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
    assert row[2] is not None
    assert '"target"' not in row[0]
    assert "병력" not in row[1]

    avoid_topic_msg = {
        "user_id": "user_001",
        "user_name": "테스터",
        "msg": "내 병력 얘기는 다시 꺼내지 마",
        "role": "user",
        "timestamp": time.time() + 1,
    }
    bot._run_slow_generation = lambda *args, **kwargs: ("알겠다. 병력 이야기는 다시 꺼내지 않겠다.", "차분함")
    repaired = bot.process_trigger([], avoid_topic_msg)
    assert "병력" not in repaired
    assert "꺼내지 않아야" in repaired or "피해야 한다" in repaired


def test_boundary_semantic_fallback_is_bounded_and_cached(temp_graph_files, temp_canonical_db):
    graph = MemoryGraph(
        graph_path=temp_graph_files["graph_path"],
        embeddings_path=temp_graph_files["embeddings_path"],
    )
    store = CanonicalMemoryStore(temp_canonical_db)
    api = CountingMockAPI()
    writer = FastPathMemoryWriter(graph, store, api)

    rules = [
        {
            "kind": "avoid_topic",
            "topic_label": "personal",
            "target": "",
            "sensitive_tokens": [f"치료이력{i}"],
            "semantic_terms": [],
        }
        for i in range(8)
    ]
    text = "치료 관련 우회 표현"

    writer.enforce_assistant_boundaries(text, boundary_rules=rules)
    first_pass_calls = api.embedding_call_count
    writer.enforce_assistant_boundaries(text, boundary_rules=rules)
    second_pass_calls = api.embedding_call_count

    assert first_pass_calls <= 1 + config.BOUNDARY_SEMANTIC_MAX_CANDIDATES
    assert second_pass_calls == first_pass_calls
