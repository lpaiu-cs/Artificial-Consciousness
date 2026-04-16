import json
import sqlite3
import time
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

import config
from api_client import APILogger, UnifiedAPIClient
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


def test_model_eval_gate_resolves_relative_to_config_module(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = UnifiedAPIClient(enable_logging=False, exclude_embedding_log=True)
    assert client is not None


def test_api_logger_writes_paired_chat_call(monkeypatch, tmp_path):
    log_file = tmp_path / "api_calls.jsonl"
    monkeypatch.setattr(config, "API_LOGGING_ENABLED", True)
    monkeypatch.setattr(config, "API_LOG_FILE", str(log_file))
    monkeypatch.setattr(config, "API_LOG_LEVEL", "DEBUG")
    monkeypatch.setattr(config, "API_LOG_INCLUDE_PAIR", True)
    monkeypatch.setattr(config, "API_LOG_INCLUDE_CONTENT", True)
    monkeypatch.setattr(config, "API_LOG_INCLUDE_PARSED_RESPONSE", True)
    monkeypatch.setattr(config, "API_LOG_MAX_CONTENT_LENGTH", 2000)
    monkeypatch.setattr(config, "API_LOG_EXCLUDE_EMBEDDING", True)

    logger = APILogger()
    call_id = logger.log_chat_request(
        "OpenAI",
        "gpt-test",
        "system ontology prompt",
        "user message",
        json_mode=True,
        response_format="json_schema",
        operation="chat_slow",
        json_schema={"name": "ontology_response", "schema": {"type": "object"}},
    )
    logger.log_chat_response(
        "OpenAI",
        "gpt-test",
        '{"claims": [{"facet": "preference"}]}',
        12.34,
        success=True,
        token_usage={"total_tokens": 42},
        call_id=call_id,
        operation="chat_slow",
        json_mode=True,
        response_format="json_schema",
        parsed_response={"claims": [{"facet": "preference"}]},
    )
    logger.log_chat_pair(
        "OpenAI",
        "gpt-test",
        "system ontology prompt",
        "user message",
        '{"claims": [{"facet": "preference"}]}',
        12.34,
        success=True,
        token_usage={"total_tokens": 42},
        call_id=call_id,
        operation="chat_slow",
        json_mode=True,
        response_format="json_schema",
        parsed_response={"claims": [{"facet": "preference"}]},
        json_schema={"name": "ontology_response", "schema": {"type": "object"}},
    )

    entries = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert [entry["type"] for entry in entries] == ["CHAT_REQUEST", "CHAT_RESPONSE", "CHAT_CALL"]
    assert len({entry["call_id"] for entry in entries}) == 1

    pair = entries[-1]
    assert pair["schema_name"] == "ontology_response"
    assert pair["request"]["system_prompt"]["preview"] == "system ontology prompt"
    assert pair["response"]["parsed"]["claims"][0]["facet"] == "preference"
    assert pair["token_usage"]["total_tokens"] == 42


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


def test_boundary_payload_is_not_plaintext_in_graph_snapshot_or_delta(
    smoke_orchestrator,
    temp_graph_files,
    temp_canonical_db,
):
    bot = smoke_orchestrator
    boundary_msg = {
        "user_id": "user_001",
        "user_name": "테스터",
        "msg": "내 병력 얘기는 저장하지 마",
        "role": "user",
        "timestamp": time.time(),
    }
    bot._run_slow_generation = lambda *args, **kwargs: ("알겠다. 병력 이야기는 저장하지 않겠다.", "차분함")
    bot.process_trigger([], boundary_msg)

    delta_path = Path(bot.ltm_graph.delta_path)
    delta_text = delta_path.read_text(encoding="utf-8")
    assert "병력" not in delta_text
    for marker in [
        "sensitive_tokens",
        "semantic_terms",
        "target_roles",
        "target_alias_hashes",
        "target_aliases",
        "target_entity_id",
    ]:
        assert marker not in delta_text

    bot.ltm_graph.save_all()
    snapshot_text = Path(temp_graph_files["graph_path"]).read_text(encoding="utf-8")
    delta_after_checkpoint = delta_path.read_text(encoding="utf-8")
    combined = snapshot_text + delta_after_checkpoint

    assert "병력" not in combined
    assert '"facet": "boundary.rule"' in snapshot_text
    for marker in [
        "sensitive_tokens",
        "semantic_terms",
        "target_roles",
        "target_alias_hashes",
        "target_aliases",
        "target_entity_id",
    ]:
        assert marker not in combined

    with sqlite3.connect(temp_canonical_db) as conn:
        row = conn.execute(
            """
            SELECT value_json, qualifiers_json, encrypted_payload_blob
            FROM claims
            WHERE facet = 'boundary.rule'
            ORDER BY updated_at DESC
            LIMIT 1
            """
        ).fetchone()
    assert row[2] is not None
    assert "병력" not in row[0]
    assert "병력" not in row[1]


def test_wrapped_double_quotes_are_removed_from_final_response(smoke_orchestrator):
    bot = smoke_orchestrator

    quoted_msg = {
        "user_id": "user_001",
        "user_name": "테스터",
        "msg": "답변 따옴표 버그를 재현해봐",
        "role": "user",
        "timestamp": time.time(),
    }
    bot._run_slow_generation = lambda *args, **kwargs: ('"이제 따옴표 없이 보일 거야."', "차분함")

    response = bot.process_trigger([], quoted_msg)

    assert response == "이제 따옴표 없이 보일 거야."
    assistant_memories = [m.content for m in bot.stm.memory_queue if m.role == "assistant"]
    assert assistant_memories[-1] == "이제 따옴표 없이 보일 거야."
    assert bot._strip_wrapping_double_quotes('그는 "안녕"이라고 말했다.') == '그는 "안녕"이라고 말했다.'


def test_legacy_boundary_persistence_is_scrubbed_on_startup(temp_graph_files):
    graph_path = Path(temp_graph_files["graph_path"])
    delta_path = Path(str(graph_path).replace(".json", "_delta.jsonl"))
    leaked_snapshot_claim = {
        "node_id": "legacy-boundary-snapshot",
        "subject_id": "user_001",
        "facet": "boundary.rule",
        "merge_key": "user_001|boundary.rule|kind=do_not_store_sensitive|target=health-fingerprint",
        "value": {
            "kind": "do_not_store_sensitive",
            "policy_kind": "do_not_store_sensitive",
            "target": "health-fingerprint",
            "target_entity_id": "user_001",
        },
        "qualifiers": {
            "topic_label": "health",
            "sensitive_tokens": ["병력"],
            "semantic_terms": ["병력", "진료", "의료"],
            "target_aliases": ["본인"],
            "target_alias_hashes": ["abc123"],
            "target_roles": ["self"],
        },
        "nl_summary": "health 관련 민감 주제를 저장하지 말라는 경계 요청이 있음",
        "source_type": "explicit",
        "confidence": 0.98,
        "status": "active",
        "valid_from": None,
        "valid_to": None,
        "last_confirmed_at": time.time(),
        "evidence_episode_ids": [],
        "sensitivity": "high",
        "scope": "user_private",
        "type": "claim",
        "edges": {},
    }
    leaked_delta_claim = {
        **leaked_snapshot_claim,
        "node_id": "legacy-boundary-delta",
        "merge_key": "user_001|boundary.rule|kind=avoid_topic|target=health-fingerprint",
        "value": {
            "kind": "avoid_topic",
            "policy_kind": "avoid_topic",
            "target": "health-fingerprint",
            "target_entity_id": "user_001",
        },
    }

    snapshot_payload = {
        "episodes": {},
        "insights": {},
        "notes": {},
        "claims": {leaked_snapshot_claim["node_id"]: leaked_snapshot_claim},
        "entities": {},
    }
    graph_path.write_text(
        json.dumps(snapshot_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    delta_entry = {
        "timestamp": time.time(),
        "action": "UPSERT_NODE",
        "payload": {
            "category": "claims",
            "data": leaked_delta_claim,
        },
    }
    delta_path.write_text(
        json.dumps(delta_entry, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    migrated_graph = MemoryGraph(
        graph_path=temp_graph_files["graph_path"],
        embeddings_path=temp_graph_files["embeddings_path"],
    )

    snapshot_text = graph_path.read_text(encoding="utf-8")
    delta_text = delta_path.read_text(encoding="utf-8")
    combined = snapshot_text + delta_text
    assert delta_text.strip() == ""
    assert "병력" not in combined
    for marker in [
        "sensitive_tokens",
        "semantic_terms",
        "target_roles",
        "target_alias_hashes",
        "target_aliases",
        "target_entity_id",
    ]:
        assert marker not in combined

    snapshot_claim = migrated_graph.claims["legacy-boundary-snapshot"]
    delta_claim = migrated_graph.claims["legacy-boundary-delta"]
    assert snapshot_claim.qualifiers == {"topic_label": "health"}
    assert delta_claim.qualifiers == {"topic_label": "health"}
    assert snapshot_claim.value == {
        "kind": "do_not_store_sensitive",
        "policy_kind": "do_not_store_sensitive",
    }
    assert delta_claim.value == {
        "kind": "avoid_topic",
        "policy_kind": "avoid_topic",
    }


def test_boundary_payload_supports_keyring_decode_and_rotation(temp_canonical_db, canonical_keyring):
    key_a = Fernet.generate_key().decode("utf-8")
    key_b = Fernet.generate_key().decode("utf-8")
    canonical_keyring({"key-a": key_a}, active_key_id="key-a")

    boundary_claim = ClaimNode(
        node_id="boundary-keyring-claim",
        subject_id="user_001",
        facet="boundary.rule",
        merge_key="user_001|boundary.rule|kind=do_not_store_sensitive|target=health-fingerprint",
        value={
            "kind": "do_not_store_sensitive",
            "policy_kind": "do_not_store_sensitive",
            "target": "health-fingerprint",
            "target_entity_id": "user_001",
        },
        qualifiers={
            "topic_label": "health",
            "sensitive_tokens": ["병력"],
            "semantic_terms": ["병력", "진료", "의료"],
            "target_aliases": ["본인"],
            "target_alias_hashes": ["abc123"],
            "target_roles": ["self"],
        },
        nl_summary="health 관련 민감 주제를 저장하지 말라는 경계 요청이 있음",
        source_type="explicit",
        confidence=0.98,
        status="active",
        last_confirmed_at=time.time(),
        sensitivity="high",
        scope="user_private",
    )

    store_a = CanonicalMemoryStore(temp_canonical_db)
    store_a.upsert_claim(boundary_claim)
    with sqlite3.connect(temp_canonical_db) as conn:
        payload_row = conn.execute(
            "SELECT payload_key_id FROM claims WHERE claim_id = ?",
            (boundary_claim.node_id,),
        ).fetchone()
    assert payload_row[0] == "key-a"

    canonical_keyring({"key-a": key_a, "key-b": key_b}, active_key_id="key-a")
    store_ab = CanonicalMemoryStore(temp_canonical_db)
    decoded_claim = store_ab.get_active_claims(
        subject_id="user_001",
        facets=["boundary.rule"],
        viewer_id="user_001",
        limit=1,
    )[0]
    assert decoded_claim.qualifiers["sensitive_tokens"] == ["병력"]
    assert decoded_claim.value["target_entity_id"] == "user_001"

    canonical_keyring({"key-b": key_b}, active_key_id="key-b")
    store_missing = CanonicalMemoryStore(temp_canonical_db)
    with pytest.raises(RuntimeError, match="missing key_id=key-a"):
        store_missing.get_active_claims(
            subject_id="user_001",
            facets=["boundary.rule"],
            viewer_id="user_001",
            limit=1,
        )

    canonical_keyring({"key-a": key_a, "key-b": key_b}, active_key_id="key-b")
    store_rotator = CanonicalMemoryStore(temp_canonical_db)
    rotated = store_rotator.reencrypt_protected_claim_payloads()
    assert rotated >= 1
    with sqlite3.connect(temp_canonical_db) as conn:
        rotated_row = conn.execute(
            "SELECT payload_key_id FROM claims WHERE claim_id = ?",
            (boundary_claim.node_id,),
        ).fetchone()
    assert rotated_row[0] == "key-b"

    canonical_keyring({"key-b": key_b}, active_key_id="key-b")
    store_b = CanonicalMemoryStore(temp_canonical_db)
    rotated_claim = store_b.get_active_claims(
        subject_id="user_001",
        facets=["boundary.rule"],
        viewer_id="user_001",
        limit=1,
    )[0]
    assert rotated_claim.qualifiers["sensitive_tokens"] == ["병력"]
    assert rotated_claim.value["target"] == "health-fingerprint"


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
