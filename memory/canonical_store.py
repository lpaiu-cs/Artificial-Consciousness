import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

import config
from memory.ontology import get_facet_spec
from memory_structures import ClaimNode, OpenLoop, RelationState


class CanonicalMemoryStore:
    """SQLite-backed canonical state store for claims, open loops, and relation state."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or getattr(config, "CANONICAL_MEMORY_DB_PATH", "memory_state.sqlite3")
        self._lock = threading.RLock()
        self._ensure_parent_dir()
        self._init_db()

    def _ensure_parent_dir(self):
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

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

    def upsert_claim(self, claim: ClaimNode):
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO claims (
                    claim_id, subject_id, facet, merge_key, status, confidence,
                    value_json, qualifiers_json, valid_from, valid_to,
                    last_confirmed_at, source_type, evidence_json,
                    sensitivity, scope, nl_summary, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(claim_id) DO UPDATE SET
                    subject_id=excluded.subject_id,
                    facet=excluded.facet,
                    merge_key=excluded.merge_key,
                    status=excluded.status,
                    confidence=excluded.confidence,
                    value_json=excluded.value_json,
                    qualifiers_json=excluded.qualifiers_json,
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
                    json.dumps(claim.value, ensure_ascii=False, sort_keys=True),
                    json.dumps(claim.qualifiers, ensure_ascii=False, sort_keys=True),
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
                         loop_id: Optional[str] = None) -> OpenLoop:
        evidence_episode_ids = list(dict.fromkeys(evidence_episode_ids or []))
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                """
                SELECT *
                FROM open_loops
                WHERE owner_id = ? AND kind = ? AND text = ? AND status = 'open'
                LIMIT 1
                """,
                (owner_id, kind, text),
            ).fetchone()

            loop_id = existing["loop_id"] if existing else (loop_id or str(uuid.uuid4()))
            updated_at = time.time()
            conn.execute(
                """
                INSERT INTO open_loops (
                    loop_id, owner_id, kind, text, due_at, status, priority, evidence_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(loop_id) DO UPDATE SET
                    owner_id=excluded.owner_id,
                    kind=excluded.kind,
                    text=excluded.text,
                    due_at=excluded.due_at,
                    status=excluded.status,
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
                    priority,
                    json.dumps(evidence_episode_ids, ensure_ascii=False),
                    updated_at,
                ),
            )

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
        )

    def close_open_loop_for_claim(self, claim: ClaimNode) -> bool:
        if claim.facet != "commitment.open_loop":
            return False

        terminal_status = self._loop_status_from_claim(claim)
        with self._lock, self._connect() as conn:
            updated = conn.execute(
                """
                UPDATE open_loops
                SET status = ?, updated_at = ?
                WHERE loop_id = ? AND status = 'open'
                """,
                (terminal_status, time.time(), claim.node_id),
            )
            if updated.rowcount:
                return True

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
            updated = conn.execute(
                """
                UPDATE open_loops
                SET status = ?, updated_at = ?
                WHERE owner_id = ? AND kind = ? AND text = ? AND status = 'open'
                """,
                (terminal_status, time.time(), claim.subject_id, kind, text),
            )
            return updated.rowcount > 0

    def get_open_loops(self, owner_id: str, search_text: str = "", limit: int = 5) -> List[OpenLoop]:
        with self._lock, self._connect() as conn:
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
            return [self._row_to_open_loop(row) for row in rows]

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

    def _row_to_claim(self, row: sqlite3.Row) -> ClaimNode:
        return ClaimNode(
            node_id=row["claim_id"],
            subject_id=row["subject_id"],
            facet=row["facet"],
            merge_key=row["merge_key"],
            value=json.loads(row["value_json"]),
            qualifiers=json.loads(row["qualifiers_json"]),
            nl_summary=row["nl_summary"] or "",
            source_type=row["source_type"] or "explicit",
            confidence=row["confidence"],
            status=row["status"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            last_confirmed_at=row["last_confirmed_at"],
            evidence_episode_ids=json.loads(row["evidence_json"] or "[]"),
            sensitivity=row["sensitivity"] or "personal",
            scope=row["scope"] or "user_private",
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

    def _tokenize(self, search_text: str) -> List[str]:
        return [token.strip() for token in (search_text or "").split() if token.strip()]

    def _claim_visible_to_viewer(self, claim: ClaimNode, viewer_id: str) -> bool:
        scope = claim.scope or "user_private"
        return claim.subject_id == viewer_id or scope == "shared"

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
