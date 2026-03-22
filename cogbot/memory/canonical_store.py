import base64
import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional

from cogbot import config
from cryptography.fernet import Fernet, InvalidToken
from cogbot.memory.ontology import get_facet_spec
from cogbot.memory_structures import ClaimNode, OpenLoop, RelationState


class CanonicalMemoryStore:
    """SQLite-backed canonical state store for claims, open loops, and relation state."""

    PROTECTED_BOUNDARY_VALUE_FIELDS = frozenset()
    PROTECTED_BOUNDARY_QUALIFIER_FIELDS = frozenset()

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or getattr(config, "CANONICAL_MEMORY_DB_PATH", "memory_state.sqlite3")
        self._lock = threading.RLock()
        self._open_loop_listener: Optional[Callable[[Dict[str, Any]], None]] = None
        self._payload_ciphers: Dict[str, Fernet] = {}
        self._active_payload_key_id = str(
            getattr(
                config,
                "CANONICAL_ACTIVE_KEY_ID",
                getattr(config, "CANONICAL_ENCRYPTION_KEY_ID", "local-v1"),
            )
        )
        self._payload_keyring_path = ""
        self._init_payload_keyring()
        self._ensure_parent_dir()
        self._init_db()
        if getattr(config, "CANONICAL_AUTO_REENCRYPT_PROTECTED_PAYLOADS", False):
            self.reencrypt_protected_claim_payloads()

    def _ensure_parent_dir(self):
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_payload_keyring(self):
        keyring_data, keyring_path = self._resolve_payload_keyring_data()
        keys = keyring_data.get("keys", {})
        self._payload_ciphers = {
            str(key_id): Fernet(self._normalize_fernet_key(str(raw_key).encode("utf-8")))
            for key_id, raw_key in keys.items()
            if str(key_id).strip() and str(raw_key).strip()
        }
        if not self._payload_ciphers:
            raise RuntimeError("Canonical payload keyring is empty.")

        active_key_id = str(
            keyring_data.get("active_key_id")
            or self._active_payload_key_id
            or getattr(config, "CANONICAL_ENCRYPTION_KEY_ID", "local-v1")
        ).strip()
        if active_key_id not in self._payload_ciphers:
            raise RuntimeError(f"Canonical payload keyring missing active key_id={active_key_id}")

        self._active_payload_key_id = active_key_id
        self._payload_keyring_path = keyring_path

    def _resolve_payload_keyring_data(self) -> tuple[Dict[str, Any], str]:
        keyring_path = self._default_keyring_path()
        env_keys_json = str(getattr(config, "CANONICAL_ENCRYPTION_KEYS_JSON", "") or "").strip()
        if env_keys_json:
            parsed = json.loads(env_keys_json)
            if not isinstance(parsed, dict):
                raise RuntimeError("CANONICAL_ENCRYPTION_KEYS_JSON must be a JSON object.")
            active_key_id = str(
                getattr(config, "CANONICAL_ACTIVE_KEY_ID", "")
                or getattr(config, "CANONICAL_ENCRYPTION_KEY_ID", "")
                or next(iter(parsed.keys()), "local-v1")
            ).strip()
            return {
                "active_key_id": active_key_id,
                "keys": {str(key_id): str(key_value) for key_id, key_value in parsed.items()},
            }, keyring_path

        configured_keyring_path = str(getattr(config, "CANONICAL_ENCRYPTION_KEYRING_PATH", "") or "").strip()
        if configured_keyring_path:
            keyring_path = os.path.abspath(os.path.expanduser(configured_keyring_path))
        os.makedirs(os.path.dirname(keyring_path), exist_ok=True)

        if os.path.exists(keyring_path):
            keyring_data = self._load_keyring_file(keyring_path)
            self._persist_keyring_file(keyring_path, keyring_data)
            return keyring_data, keyring_path

        configured_key = str(getattr(config, "CANONICAL_ENCRYPTION_KEY", "") or "").strip()
        if configured_key:
            active_key_id = str(
                getattr(config, "CANONICAL_ACTIVE_KEY_ID", "")
                or getattr(config, "CANONICAL_ENCRYPTION_KEY_ID", "local-v1")
            ).strip()
            return {
                "active_key_id": active_key_id,
                "keys": {active_key_id: configured_key},
            }, keyring_path

        legacy_key_path = str(getattr(config, "CANONICAL_ENCRYPTION_KEY_PATH", "") or "").strip()
        if legacy_key_path:
            legacy_key_path = os.path.abspath(os.path.expanduser(legacy_key_path))
            if os.path.exists(legacy_key_path):
                with open(legacy_key_path, "rb") as handle:
                    legacy_key = handle.read().strip()
                active_key_id = str(
                    getattr(config, "CANONICAL_ACTIVE_KEY_ID", "")
                    or getattr(config, "CANONICAL_ENCRYPTION_KEY_ID", "local-v1")
                ).strip()
                keyring_data = {
                    "active_key_id": active_key_id,
                    "keys": {active_key_id: legacy_key.decode("utf-8")},
                }
                self._persist_keyring_file(keyring_path, keyring_data)
                return keyring_data, keyring_path

        active_key_id = str(
            getattr(config, "CANONICAL_ACTIVE_KEY_ID", "")
            or getattr(config, "CANONICAL_ENCRYPTION_KEY_ID", "local-v1")
        ).strip()
        keyring_data = {
            "active_key_id": active_key_id,
            "keys": {active_key_id: Fernet.generate_key().decode("utf-8")},
        }
        self._persist_keyring_file(keyring_path, keyring_data)
        return keyring_data, keyring_path

    def _default_keyring_path(self) -> str:
        configured_keyring_path = str(getattr(config, "CANONICAL_ENCRYPTION_KEYRING_PATH", "") or "").strip()
        if configured_keyring_path:
            return os.path.abspath(os.path.expanduser(configured_keyring_path))
        return f"{os.path.abspath(self.db_path)}.keyring.json"

    def _load_keyring_file(self, keyring_path: str) -> Dict[str, Any]:
        with open(keyring_path, "r", encoding="utf-8") as handle:
            raw_content = handle.read().strip()
        if not raw_content:
            raise RuntimeError(f"Canonical keyring file is empty: {keyring_path}")

        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError:
            active_key_id = str(
                getattr(config, "CANONICAL_ACTIVE_KEY_ID", "")
                or getattr(config, "CANONICAL_ENCRYPTION_KEY_ID", "local-v1")
            ).strip()
            return {
                "active_key_id": active_key_id,
                "keys": {active_key_id: raw_content},
            }

        if "keys" in data:
            keys = data.get("keys", {})
            active_key_id = str(
                data.get("active_key_id")
                or getattr(config, "CANONICAL_ACTIVE_KEY_ID", "")
                or getattr(config, "CANONICAL_ENCRYPTION_KEY_ID", "")
                or next(iter(keys.keys()), "local-v1")
            ).strip()
            return {
                "active_key_id": active_key_id,
                "keys": {str(key_id): str(key_value) for key_id, key_value in keys.items()},
            }

        if isinstance(data, dict):
            active_key_id = str(
                getattr(config, "CANONICAL_ACTIVE_KEY_ID", "")
                or getattr(config, "CANONICAL_ENCRYPTION_KEY_ID", "")
                or next(iter(data.keys()), "local-v1")
            ).strip()
            return {
                "active_key_id": active_key_id,
                "keys": {str(key_id): str(key_value) for key_id, key_value in data.items()},
            }

        raise RuntimeError(f"Unsupported canonical keyring format: {keyring_path}")

    def _persist_keyring_file(self, keyring_path: str, keyring_data: Dict[str, Any]):
        os.makedirs(os.path.dirname(keyring_path), exist_ok=True)
        payload = {
            "active_key_id": str(keyring_data.get("active_key_id") or "").strip(),
            "keys": {
                str(key_id): str(key_value)
                for key_id, key_value in (keyring_data.get("keys") or {}).items()
            },
        }
        with open(keyring_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        try:
            os.chmod(keyring_path, 0o600)
        except OSError:
            pass

    def _normalize_fernet_key(self, raw_key: bytes) -> bytes:
        raw_key = (raw_key or b"").strip()
        try:
            Fernet(raw_key)
            return raw_key
        except Exception:
            digest = hashlib.sha256(raw_key).digest()
            return base64.urlsafe_b64encode(digest)

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS claims (
                    claim_id TEXT PRIMARY KEY,
                    subject_id TEXT NOT NULL,
                    facet TEXT NOT NULL,
                    merge_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    value_json TEXT NOT NULL,
                    qualifiers_json TEXT NOT NULL,
                    encrypted_payload_blob BLOB,
                    payload_key_id TEXT,
                    valid_from REAL,
                    valid_to REAL,
                    last_confirmed_at REAL,
                    source_type TEXT NOT NULL DEFAULT 'explicit',
                    evidence_json TEXT NOT NULL DEFAULT '[]',
                    sensitivity TEXT,
                    scope TEXT,
                    nl_summary TEXT,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_claims_active
                ON claims(subject_id, facet, status, valid_to);

                CREATE TABLE IF NOT EXISTS open_loops (
                    loop_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    due_at REAL,
                    status TEXT NOT NULL,
                    source_type TEXT NOT NULL DEFAULT 'explicit',
                    overdue_notified_at REAL,
                    priority INTEGER NOT NULL,
                    evidence_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_open_loops_owner
                ON open_loops(owner_id, status, due_at);

                CREATE TABLE IF NOT EXISTS relation_states (
                    entity_id TEXT PRIMARY KEY,
                    trust REAL NOT NULL,
                    warmth REAL NOT NULL,
                    familiarity REAL NOT NULL,
                    respect REAL NOT NULL,
                    tension REAL NOT NULL,
                    reliability REAL NOT NULL,
                    last_interaction_at REAL NOT NULL
                );
                """
            )
            self._migrate_claim_columns(conn)
            self._migrate_claim_scopes(conn)
            self._migrate_sensitive_claim_payloads(conn)
            self._migrate_open_loop_columns(conn)

    def set_open_loop_listener(self, listener: Optional[Callable[[Dict[str, Any]], None]]):
        self._open_loop_listener = listener

    def _migrate_claim_columns(self, conn: sqlite3.Connection):
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(claims)").fetchall()
        }
        if "source_type" not in columns:
            conn.execute(
                "ALTER TABLE claims ADD COLUMN source_type TEXT NOT NULL DEFAULT 'explicit'"
            )
        if "evidence_json" not in columns:
            conn.execute(
                "ALTER TABLE claims ADD COLUMN evidence_json TEXT NOT NULL DEFAULT '[]'"
            )
        if "encrypted_payload_blob" not in columns:
            conn.execute(
                "ALTER TABLE claims ADD COLUMN encrypted_payload_blob BLOB"
            )
        if "payload_key_id" not in columns:
            conn.execute(
                "ALTER TABLE claims ADD COLUMN payload_key_id TEXT"
            )

    def _migrate_claim_scopes(self, conn: sqlite3.Connection):
        rows = conn.execute(
            """
            SELECT claim_id, subject_id, facet, value_json, qualifiers_json, scope
            FROM claims
            WHERE scope = 'shared'
            """
        ).fetchall()
        for row in rows:
            value = json.loads(row["value_json"] or "{}")
            qualifiers = json.loads(row["qualifiers_json"] or "{}")
            scope, qualifiers = self._normalize_scope_and_qualifiers(
                row["subject_id"],
                row["scope"],
                value,
                qualifiers,
            )
            conn.execute(
                "UPDATE claims SET scope = ?, qualifiers_json = ? WHERE claim_id = ?",
                (
                    scope,
                    json.dumps(qualifiers, ensure_ascii=False, sort_keys=True),
                    row["claim_id"],
                ),
            )

    def _migrate_sensitive_claim_payloads(self, conn: sqlite3.Connection):
        rows = conn.execute(
            """
            SELECT claim_id, facet, sensitivity, value_json, qualifiers_json, encrypted_payload_blob, payload_key_id
            FROM claims
            WHERE facet = ?
            """,
            ("boundary.rule",),
        ).fetchall()
        for row in rows:
            value = json.loads(row["value_json"] or "{}")
            qualifiers = json.loads(row["qualifiers_json"] or "{}")
            public_value, public_qualifiers, protected_payload = self._split_sensitive_payload(
                row["facet"],
                row["sensitivity"],
                value,
                qualifiers,
            )
            existing_protected_payload = {}
            if row["encrypted_payload_blob"] and row["payload_key_id"] and not protected_payload:
                continue
            if row["encrypted_payload_blob"]:
                existing_protected_payload = self._decode_protected_payload(
                    row["encrypted_payload_blob"],
                    row["payload_key_id"],
                )
            if not protected_payload and not row["encrypted_payload_blob"]:
                continue
            merged_payload = self._merge_protected_payloads(existing_protected_payload, protected_payload)
            encrypted_blob, payload_key_id = self._encode_protected_payload(merged_payload)
            conn.execute(
                """
                UPDATE claims
                SET value_json = ?, qualifiers_json = ?, encrypted_payload_blob = ?, payload_key_id = ?
                WHERE claim_id = ?
                """,
                (
                    json.dumps(public_value, ensure_ascii=False, sort_keys=True),
                    json.dumps(public_qualifiers, ensure_ascii=False, sort_keys=True),
                    encrypted_blob,
                    payload_key_id,
                    row["claim_id"],
                ),
            )

    def _migrate_open_loop_columns(self, conn: sqlite3.Connection):
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(open_loops)").fetchall()
        }
        if "source_type" not in columns:
            conn.execute(
                "ALTER TABLE open_loops ADD COLUMN source_type TEXT NOT NULL DEFAULT 'explicit'"
            )
        if "overdue_notified_at" not in columns:
            conn.execute(
                "ALTER TABLE open_loops ADD COLUMN overdue_notified_at REAL"
            )

    def upsert_claim(self, claim: ClaimNode):
        claim.scope, claim.qualifiers = self._normalize_scope_and_qualifiers(
            claim.subject_id,
            claim.scope,
            claim.value,
            claim.qualifiers,
        )
        public_value, public_qualifiers, encrypted_blob, payload_key_id = self._prepare_claim_storage(claim)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO claims (
                    claim_id, subject_id, facet, merge_key, status, confidence,
                    value_json, qualifiers_json, encrypted_payload_blob, payload_key_id, valid_from, valid_to,
                    last_confirmed_at, source_type, evidence_json,
                    sensitivity, scope, nl_summary, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(claim_id) DO UPDATE SET
                    subject_id=excluded.subject_id,
                    facet=excluded.facet,
                    merge_key=excluded.merge_key,
                    status=excluded.status,
                    confidence=excluded.confidence,
                    value_json=excluded.value_json,
                    qualifiers_json=excluded.qualifiers_json,
                    encrypted_payload_blob=excluded.encrypted_payload_blob,
                    payload_key_id=excluded.payload_key_id,
                    valid_from=excluded.valid_from,
                    valid_to=excluded.valid_to,
                    last_confirmed_at=excluded.last_confirmed_at,
                    source_type=excluded.source_type,
                    evidence_json=excluded.evidence_json,
                    sensitivity=excluded.sensitivity,
                    scope=excluded.scope,
                    nl_summary=excluded.nl_summary,
                    updated_at=excluded.updated_at
                """,
                (
                    claim.node_id,
                    claim.subject_id,
                    claim.facet,
                    claim.merge_key,
                    claim.status,
                    claim.confidence,
                    json.dumps(public_value, ensure_ascii=False, sort_keys=True),
                    json.dumps(public_qualifiers, ensure_ascii=False, sort_keys=True),
                    encrypted_blob,
                    payload_key_id,
                    claim.valid_from,
                    claim.valid_to,
                    claim.last_confirmed_at,
                    claim.source_type,
                    json.dumps(claim.evidence_episode_ids, ensure_ascii=False),
                    claim.sensitivity,
                    claim.scope,
                    claim.nl_summary,
                    time.time(),
                ),
            )
        self.sync_open_loop_with_claim(claim)

    def sync_open_loop_with_claim(self, claim: ClaimNode) -> Optional[OpenLoop]:
        if claim.facet != "commitment.open_loop":
            return None
        if claim.status == "active":
            return self.upsert_open_loop_from_claim(claim)
        self.close_open_loop_for_claim(claim)
        return None

    def get_active_claims(self, subject_id: str, facets: Optional[Iterable[str]] = None,
                          search_text: str = "", limit: int = 10,
                          viewer_id: Optional[str] = None) -> List[ClaimNode]:
        with self._lock, self._connect() as conn:
            query = """
                SELECT *
                FROM claims
                WHERE subject_id = ?
                  AND status = 'active'
                  AND (valid_to IS NULL OR valid_to >= ?)
            """
            params: List[Any] = [subject_id, time.time()]

            if facets:
                facet_list = list(facets)
                placeholders = ", ".join("?" for _ in facet_list)
                query += f" AND facet IN ({placeholders})"
                params.extend(facet_list)

            for token in self._tokenize(search_text):
                like = f"%{token}%"
                query += " AND (nl_summary LIKE ? OR value_json LIKE ? OR qualifiers_json LIKE ?)"
                params.extend([like, like, like])

            query += " ORDER BY last_confirmed_at DESC LIMIT ?"
            params.append(max(limit * 3, limit))

            rows = conn.execute(query, params).fetchall()
            claims = [self._row_to_claim(row) for row in rows]
            if viewer_id is not None:
                claims = [
                    claim for claim in claims
                    if self._claim_visible_to_viewer(claim, viewer_id)
                ]
            claims.sort(
                key=lambda claim: (
                    get_facet_spec(claim.facet).retrieval_priority,
                    -claim.confidence,
                    -(claim.last_confirmed_at or 0.0),
                )
            )
            return claims[:limit]

    def upsert_open_loop(self, owner_id: str, kind: str, text: str,
                         due_at: Optional[float] = None, status: str = "open",
                         priority: int = 0,
                         evidence_episode_ids: Optional[List[str]] = None,
                         loop_id: Optional[str] = None,
                         source_type: str = "explicit") -> OpenLoop:
        evidence_episode_ids = list(dict.fromkeys(evidence_episode_ids or []))
        lifecycle_event: Optional[Dict[str, Any]] = None
        with self._lock, self._connect() as conn:
            existing = None
            if loop_id:
                existing = conn.execute(
                    """
                    SELECT *
                    FROM open_loops
                    WHERE loop_id = ?
                    LIMIT 1
                    """,
                    (loop_id,),
                ).fetchone()
            if not existing:
                existing = conn.execute(
                    """
                    SELECT *
                    FROM open_loops
                    WHERE owner_id = ? AND kind = ? AND text = ?
                    ORDER BY CASE WHEN status = 'open' THEN 0 ELSE 1 END, updated_at DESC
                    LIMIT 1
                    """,
                    (owner_id, kind, text),
                ).fetchone()

            loop_id = existing["loop_id"] if existing else (loop_id or str(uuid.uuid4()))
            updated_at = time.time()
            overdue_notified_at = existing["overdue_notified_at"] if existing else None
            created = existing is None
            reopened = bool(existing and existing["status"] != "open" and status == "open")
            if status == "open" and (created or reopened):
                overdue_notified_at = None
            conn.execute(
                """
                INSERT INTO open_loops (
                    loop_id, owner_id, kind, text, due_at, status, source_type,
                    overdue_notified_at, priority, evidence_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(loop_id) DO UPDATE SET
                    owner_id=excluded.owner_id,
                    kind=excluded.kind,
                    text=excluded.text,
                    due_at=excluded.due_at,
                    status=excluded.status,
                    source_type=excluded.source_type,
                    overdue_notified_at=excluded.overdue_notified_at,
                    priority=excluded.priority,
                    evidence_json=excluded.evidence_json,
                    updated_at=excluded.updated_at
                """,
                (
                    loop_id,
                    owner_id,
                    kind,
                    text,
                    due_at,
                    status,
                    source_type,
                    overdue_notified_at,
                    priority,
                    json.dumps(evidence_episode_ids, ensure_ascii=False),
                    updated_at,
                ),
            )
            if created or reopened:
                lifecycle_event = {
                    "event_type": "reopened" if reopened else "opened",
                    "owner_id": owner_id,
                    "kind": kind,
                    "text": text,
                    "due_at": due_at,
                    "source_type": source_type,
                    "occurred_at": updated_at,
                }

        if lifecycle_event:
            self._emit_open_loop_event(lifecycle_event)

        return OpenLoop(
                loop_id=loop_id,
                owner_id=owner_id,
                kind=kind,
                text=text,
                due_at=due_at,
                status=status,
                priority=priority,
                evidence_episode_ids=evidence_episode_ids,
            )

    def upsert_open_loop_from_claim(self, claim: ClaimNode) -> Optional[OpenLoop]:
        if claim.facet != "commitment.open_loop" or claim.status != "active":
            return None

        kind = (
            claim.value.get("kind")
            or claim.qualifiers.get("kind")
            or "followup_needed"
        )
        text = (
            claim.value.get("text")
            or claim.qualifiers.get("text")
            or claim.nl_summary
        )
        if not text:
            return None

        due_at = claim.value.get("due_at") or claim.qualifiers.get("due_at") or claim.valid_to
        priority = int(
            claim.value.get("priority")
            or claim.qualifiers.get("priority")
            or 5
        )
        return self.upsert_open_loop(
            owner_id=claim.subject_id,
            kind=kind,
            text=text,
            due_at=due_at,
            status="open",
            priority=priority,
            evidence_episode_ids=claim.evidence_episode_ids,
            loop_id=claim.node_id,
            source_type=claim.source_type or "explicit",
        )

    def close_open_loop_for_claim(self, claim: ClaimNode) -> bool:
        if claim.facet != "commitment.open_loop":
            return False

        terminal_status = self._loop_status_from_claim(claim)
        lifecycle_event: Optional[Dict[str, Any]] = None
        with self._lock, self._connect() as conn:
            now = time.time()
            target_row = conn.execute(
                """
                SELECT *
                FROM open_loops
                WHERE loop_id = ? AND status = 'open'
                LIMIT 1
                """,
                (claim.node_id,),
            ).fetchone()
            kind = (
                claim.value.get("kind")
                or claim.qualifiers.get("kind")
                or "followup_needed"
            )
            text = (
                claim.value.get("text")
                or claim.qualifiers.get("text")
                or claim.nl_summary
            )
            if not target_row:
                target_row = conn.execute(
                    """
                    SELECT *
                    FROM open_loops
                    WHERE owner_id = ? AND kind = ? AND text = ? AND status = 'open'
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (claim.subject_id, kind, text),
                ).fetchone()
            if not target_row:
                return False

            updated = conn.execute(
                """
                UPDATE open_loops
                SET status = ?, updated_at = ?
                WHERE loop_id = ? AND status = 'open'
                """,
                (terminal_status, now, target_row["loop_id"]),
            )
            if updated.rowcount:
                lifecycle_event = {
                    "event_type": "closed",
                    "owner_id": target_row["owner_id"],
                    "kind": target_row["kind"],
                    "text": target_row["text"],
                    "due_at": target_row["due_at"],
                    "source_type": target_row["source_type"] or claim.source_type or "explicit",
                    "occurred_at": now,
                    "terminal_status": terminal_status,
                    "was_overdue": bool(
                        target_row["overdue_notified_at"] is not None
                        or (
                            target_row["due_at"] is not None
                            and float(target_row["due_at"]) < now
                        )
                    ),
                }

        if lifecycle_event:
            self._emit_open_loop_event(lifecycle_event)
            return True
        return False

    def get_open_loops(self, owner_id: str, search_text: str = "", limit: int = 5) -> List[OpenLoop]:
        lifecycle_events: List[Dict[str, Any]] = []
        with self._lock, self._connect() as conn:
            lifecycle_events = self._mark_overdue_open_loops_locked(conn, owner_id)
            query = """
                SELECT *
                FROM open_loops
                WHERE owner_id = ? AND status = 'open'
            """
            params: List[Any] = [owner_id]

            for token in self._tokenize(search_text):
                query += " AND text LIKE ?"
                params.append(f"%{token}%")

            query += " ORDER BY priority DESC, COALESCE(due_at, 9999999999) ASC, updated_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            loops = [self._row_to_open_loop(row) for row in rows]

        for event in lifecycle_events:
            self._emit_open_loop_event(event)
        return loops

    def get_interaction_policy(self, subject_id: str) -> Dict[str, Any]:
        policy: Dict[str, Any] = {}
        claims = self.get_active_claims(
            subject_id=subject_id,
            facets=["interaction.preference"],
            limit=20,
        )
        for claim in claims:
            dimension = claim.value.get("dimension")
            value = claim.value.get("value")
            if dimension and value is not None:
                policy[dimension] = value
        return policy

    def get_relation_state(self, entity_id: str) -> RelationState:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM relation_states WHERE entity_id = ?",
                (entity_id,),
            ).fetchone()
            if row:
                return RelationState(
                    entity_id=row["entity_id"],
                    trust=row["trust"],
                    warmth=row["warmth"],
                    familiarity=row["familiarity"],
                    respect=row["respect"],
                    tension=row["tension"],
                    reliability=row["reliability"],
                    last_interaction_at=row["last_interaction_at"],
                )

        return RelationState(
            entity_id=entity_id,
            trust=0.5,
            warmth=0.5,
            familiarity=0.1,
            respect=0.5,
            tension=0.0,
            reliability=0.5,
            last_interaction_at=0.0,
        )

    def save_relation_state(self, state: RelationState) -> RelationState:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO relation_states (
                    entity_id, trust, warmth, familiarity, respect, tension, reliability, last_interaction_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    trust=excluded.trust,
                    warmth=excluded.warmth,
                    familiarity=excluded.familiarity,
                    respect=excluded.respect,
                    tension=excluded.tension,
                    reliability=excluded.reliability,
                    last_interaction_at=excluded.last_interaction_at
                """,
                (
                    state.entity_id,
                    state.trust,
                    state.warmth,
                    state.familiarity,
                    state.respect,
                    state.tension,
                    state.reliability,
                    state.last_interaction_at,
                ),
            )
        return state

    def update_relation_state(self, entity_id: str, deltas: Dict[str, float]) -> RelationState:
        state = self.get_relation_state(entity_id)
        now = time.time()

        state.trust = self._clamp01(state.trust + deltas.get("trust", 0.0))
        state.warmth = self._clamp01(state.warmth + deltas.get("warmth", 0.0))
        state.familiarity = self._clamp01(state.familiarity + deltas.get("familiarity", 0.0))
        state.respect = self._clamp01(state.respect + deltas.get("respect", 0.0))
        state.tension = self._clamp01(state.tension + deltas.get("tension", 0.0))
        state.reliability = self._clamp01(state.reliability + deltas.get("reliability", 0.0))
        state.last_interaction_at = now

        return self.save_relation_state(state)

    def _prepare_claim_storage(self, claim: ClaimNode) -> tuple[Dict[str, Any], Dict[str, Any], Optional[bytes], Optional[str]]:
        public_value, public_qualifiers, protected_payload = self._split_sensitive_payload(
            claim.facet,
            claim.sensitivity,
            claim.value,
            claim.qualifiers,
        )
        encrypted_blob, payload_key_id = self._encode_protected_payload(protected_payload)
        return public_value, public_qualifiers, encrypted_blob, payload_key_id

    def _split_sensitive_payload(self, facet: str, sensitivity: Optional[str],
                                 value: Dict[str, Any], qualifiers: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        public_value = dict(value or {})
        public_qualifiers = dict(qualifiers or {})
        protected_value: Dict[str, Any] = {}
        protected_qualifiers: Dict[str, Any] = {}

        if facet == "boundary.rule" and (sensitivity or "high") == "high":
            public_value = {
                key: field_value
                for key, field_value in public_value.items()
                if key in ClaimNode.BOUNDARY_PUBLIC_VALUE_FIELDS
            }
            protected_value = {
                key: field_value
                for key, field_value in (value or {}).items()
                if key not in ClaimNode.BOUNDARY_PUBLIC_VALUE_FIELDS and field_value not in (None, "", [], {})
            }
            public_qualifiers = {
                key: field_value
                for key, field_value in public_qualifiers.items()
                if key in ClaimNode.BOUNDARY_PUBLIC_QUALIFIER_FIELDS
            }
            protected_qualifiers = {
                key: field_value
                for key, field_value in (qualifiers or {}).items()
                if key not in ClaimNode.BOUNDARY_PUBLIC_QUALIFIER_FIELDS and field_value not in (None, "", [], {})
            }

        protected_payload = {}
        if protected_value:
            protected_payload["value"] = protected_value
        if protected_qualifiers:
            protected_payload["qualifiers"] = protected_qualifiers
        return public_value, public_qualifiers, protected_payload

    def _encode_protected_payload(self, protected_payload: Dict[str, Any]) -> tuple[Optional[bytes], Optional[str]]:
        if not protected_payload:
            return None, None
        payload_json = json.dumps(protected_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        cipher = self._payload_ciphers.get(self._active_payload_key_id)
        if cipher is None:
            raise RuntimeError(f"Canonical payload keyring missing active key_id={self._active_payload_key_id}")
        return cipher.encrypt(payload_json), self._active_payload_key_id

    def _decode_protected_payload(self, encrypted_blob: Any, payload_key_id: Optional[str]) -> Dict[str, Any]:
        if not encrypted_blob:
            return {}
        if payload_key_id:
            cipher = self._payload_ciphers.get(str(payload_key_id))
            if cipher is None:
                raise RuntimeError(
                    f"Unable to decrypt canonical memory payload: missing key_id={payload_key_id}"
                )
            try:
                decrypted = cipher.decrypt(bytes(encrypted_blob))
            except InvalidToken as exc:
                raise RuntimeError(
                    f"Unable to decrypt canonical memory payload with key_id={payload_key_id}"
                ) from exc
            return json.loads(decrypted.decode("utf-8"))

        for legacy_key_id, cipher in self._payload_ciphers.items():
            try:
                decrypted = cipher.decrypt(bytes(encrypted_blob))
                return json.loads(decrypted.decode("utf-8"))
            except InvalidToken:
                continue
        try:
            if len(self._payload_ciphers) == 1:
                only_key_id, cipher = next(iter(self._payload_ciphers.items()))
                decrypted = cipher.decrypt(bytes(encrypted_blob))
                return json.loads(decrypted.decode("utf-8"))
        except InvalidToken:
            pass
        raise RuntimeError(
            "Unable to decrypt canonical memory payload with legacy key fallback"
        )

    def reencrypt_protected_claim_payloads(self, target_key_id: Optional[str] = None) -> int:
        desired_key_id = str(target_key_id or self._active_payload_key_id or "").strip()
        if desired_key_id not in self._payload_ciphers:
            raise RuntimeError(
                f"Cannot re-encrypt protected payloads: missing key_id={desired_key_id}"
            )

        updated_rows = 0
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT claim_id, encrypted_payload_blob, payload_key_id
                FROM claims
                WHERE encrypted_payload_blob IS NOT NULL
                """
            ).fetchall()
            for row in rows:
                protected_payload = self._decode_protected_payload(
                    row["encrypted_payload_blob"],
                    row["payload_key_id"],
                )
                encrypted_blob, payload_key_id = self._encode_with_key_id(protected_payload, desired_key_id)
                conn.execute(
                    """
                    UPDATE claims
                    SET encrypted_payload_blob = ?, payload_key_id = ?, updated_at = ?
                    WHERE claim_id = ?
                    """,
                    (encrypted_blob, payload_key_id, time.time(), row["claim_id"]),
                )
                updated_rows += 1
        return updated_rows

    def _encode_with_key_id(self, protected_payload: Dict[str, Any], key_id: str) -> tuple[Optional[bytes], Optional[str]]:
        if not protected_payload:
            return None, None
        cipher = self._payload_ciphers.get(str(key_id))
        if cipher is None:
            raise RuntimeError(f"Canonical payload keyring missing key_id={key_id}")
        payload_json = json.dumps(protected_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return cipher.encrypt(payload_json), str(key_id)

    def _merge_protected_payloads(self, base_payload: Dict[str, Any], incoming_payload: Dict[str, Any]) -> Dict[str, Any]:
        merged = {
            "value": dict(base_payload.get("value", {})),
            "qualifiers": dict(base_payload.get("qualifiers", {})),
        }
        merged["value"].update(incoming_payload.get("value", {}))
        merged["qualifiers"].update(incoming_payload.get("qualifiers", {}))
        if not merged["value"]:
            merged.pop("value")
        if not merged["qualifiers"]:
            merged.pop("qualifiers")
        return merged

    def _row_to_claim(self, row: sqlite3.Row) -> ClaimNode:
        value = json.loads(row["value_json"])
        qualifiers = json.loads(row["qualifiers_json"])
        protected_payload = self._decode_protected_payload(
            row["encrypted_payload_blob"],
            row["payload_key_id"],
        )
        value.update(protected_payload.get("value", {}))
        qualifiers.update(protected_payload.get("qualifiers", {}))
        scope, qualifiers = self._normalize_scope_and_qualifiers(
            row["subject_id"],
            row["scope"] or "user_private",
            value,
            qualifiers,
        )
        return ClaimNode(
            node_id=row["claim_id"],
            subject_id=row["subject_id"],
            facet=row["facet"],
            merge_key=row["merge_key"],
            value=value,
            qualifiers=qualifiers,
            nl_summary=row["nl_summary"] or "",
            source_type=row["source_type"] or "explicit",
            confidence=row["confidence"],
            status=row["status"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            last_confirmed_at=row["last_confirmed_at"],
            evidence_episode_ids=json.loads(row["evidence_json"] or "[]"),
            sensitivity=row["sensitivity"] or "personal",
            scope=scope,
        )

    def _row_to_open_loop(self, row: sqlite3.Row) -> OpenLoop:
        return OpenLoop(
            loop_id=row["loop_id"],
            owner_id=row["owner_id"],
            kind=row["kind"],
            text=row["text"],
            due_at=row["due_at"],
            status=row["status"],
            priority=row["priority"],
            evidence_episode_ids=json.loads(row["evidence_json"] or "[]"),
        )

    def _mark_overdue_open_loops_locked(self, conn: sqlite3.Connection, owner_id: str) -> List[Dict[str, Any]]:
        now = time.time()
        rows = conn.execute(
            """
            SELECT *
            FROM open_loops
            WHERE owner_id = ?
              AND status = 'open'
              AND due_at IS NOT NULL
              AND due_at < ?
              AND overdue_notified_at IS NULL
            """,
            (owner_id, now),
        ).fetchall()
        if not rows:
            return []

        conn.executemany(
            """
            UPDATE open_loops
            SET overdue_notified_at = ?, updated_at = ?
            WHERE loop_id = ?
            """,
            [(now, now, row["loop_id"]) for row in rows],
        )
        return [
            {
                "event_type": "overdue",
                "owner_id": row["owner_id"],
                "kind": row["kind"],
                "text": row["text"],
                "due_at": row["due_at"],
                "source_type": row["source_type"] or "explicit",
                "occurred_at": now,
            }
            for row in rows
        ]

    def _tokenize(self, search_text: str) -> List[str]:
        return [token.strip() for token in (search_text or "").split() if token.strip()]

    def _claim_visible_to_viewer(self, claim: ClaimNode, viewer_id: str) -> bool:
        scope = claim.scope or "user_private"
        if claim.subject_id == viewer_id:
            return True
        if scope != "participants":
            return False
        audience_ids = {
            str(audience_id)
            for audience_id in (claim.qualifiers.get("audience_ids") or [])
            if audience_id
        }
        return viewer_id in audience_ids

    def _normalize_scope_and_qualifiers(self, subject_id: str, scope: Optional[str],
                                        value: Dict[str, Any], qualifiers: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        normalized_scope = "participants" if scope == "shared" else (scope or "user_private")
        normalized_qualifiers = dict(qualifiers or {})
        if normalized_scope != "participants":
            return normalized_scope, normalized_qualifiers

        audience_ids = [
            str(audience_id)
            for audience_id in normalized_qualifiers.get("audience_ids", [])
            if audience_id
        ]
        if not audience_ids:
            audience_ids.append(str(subject_id))
            target_entity_id = value.get("target_entity_id")
            if target_entity_id:
                audience_ids.append(str(target_entity_id))
        normalized_qualifiers["audience_ids"] = list(dict.fromkeys(audience_ids))
        return normalized_scope, normalized_qualifiers

    def _loop_status_from_claim(self, claim: ClaimNode) -> str:
        explicit_status = (
            claim.value.get("loop_status")
            or claim.qualifiers.get("loop_status")
            or claim.value.get("status")
            or claim.qualifiers.get("status")
        )
        if explicit_status in {"open", "done", "abandoned"}:
            return explicit_status
        if claim.status == "superseded":
            return "done"
        if claim.status in {"retracted", "expired", "uncertain"}:
            return "abandoned"
        return "done"

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _emit_open_loop_event(self, event: Dict[str, Any]):
        if not self._open_loop_listener:
            return
        try:
            self._open_loop_listener(event)
        except Exception as exc:
            print(f"⚠️ Open loop listener error: {exc}")
