"""Durable delegation records in the existing profile-scoped SessionDB.

The service owns capability policy and tool execution. This module owns atomic
state transitions, lease fencing and idempotency; it never resolves credentials.
"""

from __future__ import annotations

import json
import math
import time
import uuid


_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS orchestration_workers (
        worker_id TEXT PRIMARY KEY, owner_session_id TEXT NOT NULL,
        parent_worker_id TEXT, root_worker_id TEXT NOT NULL, depth INTEGER NOT NULL,
        profile TEXT, config_revision TEXT NOT NULL, policy TEXT NOT NULL,
        frozen_prompt TEXT NOT NULL, frozen_prompt_hash TEXT NOT NULL,
        history TEXT NOT NULL DEFAULT '[]', uncertain_side_effect INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL, updated_at REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS orchestration_runs (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT UNIQUE NOT NULL,
        worker_id TEXT NOT NULL REFERENCES orchestration_workers(worker_id),
        request_id TEXT NOT NULL, previous_run_id TEXT,
        goal TEXT NOT NULL, context TEXT NOT NULL, status TEXT NOT NULL,
        lease_token TEXT, lease_expires_at REAL, tool_inflight INTEGER NOT NULL DEFAULT 0,
        uncertain_side_effect INTEGER NOT NULL DEFAULT 0, result TEXT,
        completion_ack INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL, updated_at REAL NOT NULL,
        UNIQUE(worker_id, request_id))""",
    """CREATE UNIQUE INDEX IF NOT EXISTS orchestration_one_active
        ON orchestration_runs(worker_id) WHERE status = 'RUNNING'""",
    """CREATE TABLE IF NOT EXISTS orchestration_messages (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT, message_id TEXT UNIQUE NOT NULL,
        worker_id TEXT NOT NULL REFERENCES orchestration_workers(worker_id),
        sender_id TEXT, content TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING',
        delivered_run_id TEXT, created_at REAL NOT NULL, delivered_at REAL)""",
    "CREATE INDEX IF NOT EXISTS orchestration_owner ON orchestration_workers(owner_session_id)",
    "CREATE INDEX IF NOT EXISTS orchestration_queue ON orchestration_runs(worker_id, status, sequence)",
    "CREATE INDEX IF NOT EXISTS orchestration_mailbox ON orchestration_messages(worker_id, status, sequence)",
)
_JSON_FIELDS = {"policy", "history", "result"}
_SECRET_KEYS = {"api_key", "access_token", "refresh_token", "token", "password", "authorization", "cookie", "credentials"}
_TERMINAL = {"SUCCEEDED", "FAILED", "INTERRUPTED", "CANCELLED"}


def _row(row):
    if row is None:
        return None
    value = dict(row)
    for key in _JSON_FIELDS.intersection(value):
        value[key] = json.loads(value[key]) if value[key] is not None else None
    for key in {"uncertain_side_effect", "tool_inflight", "completion_ack"}.intersection(value):
        value[key] = bool(value[key])
    return value


def _policy_json(value):
    def check(item):
        if isinstance(item, dict):
            if any(str(key).lower() in _SECRET_KEYS for key in item):
                raise ValueError("Worker policy must not contain credentials")
            for child in item.values():
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)
    check(value)
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


class WorkerStore:
    def __init__(self, session_db):
        self.db = session_db

    def ensure_schema(self):
        def migrate(conn):
            for statement in _SCHEMA:
                conn.execute(statement)
        self.db._execute_write(migrate)

    @staticmethod
    def _worker(conn, worker_id, owner_session_id):
        worker = _row(conn.execute("SELECT * FROM orchestration_workers WHERE worker_id=?", (worker_id,)).fetchone())
        if worker is None or not owner_session_id or worker["owner_session_id"] != owner_session_id:
            raise PermissionError("Unknown worker or foreign owner")
        return worker

    @classmethod
    def _run(cls, conn, run_id, owner_session_id):
        run = _row(conn.execute("SELECT * FROM orchestration_runs WHERE run_id=?", (run_id,)).fetchone())
        if run is None:
            raise PermissionError("Unknown run or foreign owner")
        cls._worker(conn, run["worker_id"], owner_session_id)
        return run

    @classmethod
    def _lease(cls, conn, run_id, owner_session_id, lease_token):
        run = cls._run(conn, run_id, owner_session_id)
        if (not lease_token or run["status"] != "RUNNING" or run["lease_token"] != lease_token
                or run["lease_expires_at"] <= time.time()):
            raise PermissionError("Worker execution lease is no longer valid")
        return run

    def create_worker(self, owner_session_id, *, profile=None, config_revision="", policy=None,
                      frozen_prompt="", parent_worker_id=None, worker_id=None):
        import hashlib

        if not isinstance(owner_session_id, str) or not owner_session_id.strip():
            raise ValueError("A stable owner session is required")
        worker_id = worker_id or "worker-" + uuid.uuid4().hex
        policy_text = _policy_json(policy or {})
        prompt_hash = hashlib.sha256(frozen_prompt.encode()).hexdigest()
        def create(conn):
            parent = self._worker(conn, parent_worker_id, owner_session_id) if parent_worker_id else None
            now = time.time()
            conn.execute("""INSERT INTO orchestration_workers
                (worker_id,owner_session_id,parent_worker_id,root_worker_id,depth,profile,
                 config_revision,policy,frozen_prompt,frozen_prompt_hash,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (worker_id, owner_session_id, parent_worker_id,
                 parent["root_worker_id"] if parent else worker_id, parent["depth"] + 1 if parent else 1,
                 profile, config_revision, policy_text, frozen_prompt, prompt_hash, now, now))
            return self._worker(conn, worker_id, owner_session_id)
        return self.db._execute_write(create)

    def get_worker(self, worker_id, owner_session_id):
        with self.db._read_ctx() as conn:
            return self._worker(conn, worker_id, owner_session_id)

    def list_workers(self, owner_session_id):
        with self.db._read_ctx() as conn:
            return [_row(r) for r in conn.execute(
                "SELECT * FROM orchestration_workers WHERE owner_session_id=? ORDER BY created_at,worker_id",
                (owner_session_id,))]

    def get_run(self, run_id, owner_session_id):
        with self.db._read_ctx() as conn:
            return self._run(conn, run_id, owner_session_id)

    def enqueue_run(self, worker_id, owner_session_id, *, goal, context="", request_id=None, previous_run_id=None):
        if not isinstance(goal, str) or not goal.strip() or not isinstance(context, str):
            raise ValueError("A run needs a nonempty goal and text context")
        run_id, request_id = "run-" + uuid.uuid4().hex, request_id or uuid.uuid4().hex
        def enqueue(conn):
            worker = self._worker(conn, worker_id, owner_session_id)
            existing = _row(conn.execute("SELECT * FROM orchestration_runs WHERE worker_id=? AND request_id=?",
                                         (worker_id, request_id)).fetchone())
            if existing:
                if (existing["goal"], existing["context"], existing["previous_run_id"]) != (goal, context, previous_run_id):
                    raise ValueError("Run request ID was already used for different content")
                return existing
            if worker["uncertain_side_effect"]:
                raise ValueError("Reconcile the interrupted tool outcome before resuming this worker")
            if previous_run_id:
                previous = self._run(conn, previous_run_id, owner_session_id)
                if previous["worker_id"] != worker_id or previous["status"] not in _TERMINAL:
                    raise ValueError("Resume must link to this worker's terminal run")
            now = time.time()
            conn.execute("""INSERT INTO orchestration_runs
                (run_id,worker_id,request_id,previous_run_id,goal,context,status,created_at,updated_at)
                VALUES (?,?,?,?,?,?,'PENDING',?,?)""",
                (run_id, worker_id, request_id, previous_run_id, goal, context, now, now))
            return self._run(conn, run_id, owner_session_id)
        return self.db._execute_write(enqueue)

    def claim_next_run(self, worker_id, owner_session_id, *, lease_seconds=60, max_concurrent=10):
        _positive(lease_seconds, "lease_seconds")
        if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int) or max_concurrent < 1:
            raise ValueError("max_concurrent must be a positive integer")
        token = uuid.uuid4().hex
        def claim(conn):
            worker = self._worker(conn, worker_id, owner_session_id)
            if worker["uncertain_side_effect"]:
                return None
            # Expired RUNNING rows still count until explicit recovery fences them.
            active = conn.execute("SELECT 1 FROM orchestration_runs WHERE worker_id=? AND status='RUNNING'", (worker_id,)).fetchone()
            count = conn.execute("""SELECT count(*) FROM orchestration_runs r JOIN orchestration_workers w
                ON r.worker_id=w.worker_id WHERE w.owner_session_id=? AND r.status='RUNNING'""",
                (owner_session_id,)).fetchone()[0]
            if active or count >= max_concurrent:
                return None
            run = conn.execute("SELECT run_id FROM orchestration_runs WHERE worker_id=? AND status='PENDING' ORDER BY sequence LIMIT 1", (worker_id,)).fetchone()
            if run is None:
                return None
            now = time.time()
            conn.execute("UPDATE orchestration_runs SET status='RUNNING',lease_token=?,lease_expires_at=?,updated_at=? WHERE run_id=?",
                         (token, now + lease_seconds, now, run[0]))
            return self._run(conn, run[0], owner_session_id)
        return self.db._execute_write(claim)

    def heartbeat_run(self, run_id, owner_session_id, lease_token, *, lease_seconds=60):
        _positive(lease_seconds, "lease_seconds")
        def heartbeat(conn):
            self._lease(conn, run_id, owner_session_id, lease_token)
            now = time.time()
            conn.execute("UPDATE orchestration_runs SET lease_expires_at=?,updated_at=? WHERE run_id=?", (now + lease_seconds, now, run_id))
        self.db._execute_write(heartbeat)

    def checkpoint_run(self, run_id, owner_session_id, lease_token, *, history, tool_inflight=False, delivered_message_ids=()):
        history_json = json.dumps(history, allow_nan=False)
        def checkpoint(conn):
            run = self._lease(conn, run_id, owner_session_id, lease_token)
            now = time.time()
            conn.execute("UPDATE orchestration_workers SET history=?,updated_at=? WHERE worker_id=?", (history_json, now, run["worker_id"]))
            conn.execute("UPDATE orchestration_runs SET tool_inflight=?,updated_at=? WHERE run_id=?", (int(tool_inflight), now, run_id))
            for message_id in delivered_message_ids:
                self._ack_message(conn, run, message_id)
        self.db._execute_write(checkpoint)

    def finish_run(self, run_id, owner_session_id, lease_token, *, status, result, history=None):
        if status not in _TERMINAL:
            raise ValueError("Invalid terminal run status")
        result_json = _policy_json(result)
        def finish(conn):
            run = self._lease(conn, run_id, owner_session_id, lease_token)
            now = time.time()
            uncertain = bool(run["tool_inflight"]) and status != "SUCCEEDED"
            conn.execute("""UPDATE orchestration_runs SET status=?,result=?,uncertain_side_effect=?,
                lease_token=NULL,lease_expires_at=NULL,updated_at=? WHERE run_id=?""",
                (status, result_json, int(uncertain), now, run_id))
            conn.execute("UPDATE orchestration_workers SET uncertain_side_effect=?,updated_at=? WHERE worker_id=?", (int(uncertain), now, run["worker_id"]))
            if history is not None:
                conn.execute("UPDATE orchestration_workers SET history=? WHERE worker_id=?", (json.dumps(history, allow_nan=False), run["worker_id"]))
            return self._run(conn, run_id, owner_session_id)
        return self.db._execute_write(finish)

    def enqueue_message(self, worker_id, owner_session_id, content, *, message_id=None, sender_id=None):
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Message must be nonempty text")
        message_id = message_id or "message-" + uuid.uuid4().hex
        def enqueue(conn):
            self._worker(conn, worker_id, owner_session_id)
            existing = _row(conn.execute("SELECT * FROM orchestration_messages WHERE message_id=?", (message_id,)).fetchone())
            if existing:
                if (existing["worker_id"], existing["content"], existing["sender_id"]) != (worker_id, content, sender_id):
                    raise ValueError("Message ID was already used for different content")
                return existing
            conn.execute("INSERT INTO orchestration_messages(message_id,worker_id,sender_id,content,created_at) VALUES (?,?,?,?,?)",
                         (message_id, worker_id, sender_id, content, time.time()))
            return _row(conn.execute("SELECT * FROM orchestration_messages WHERE message_id=?", (message_id,)).fetchone())
        return self.db._execute_write(enqueue)

    def claim_messages(self, run_id, owner_session_id, lease_token):
        # Delivery becomes final only alongside a persisted conversation checkpoint.
        with self.db._read_ctx() as conn:
            run = self._lease(conn, run_id, owner_session_id, lease_token)
            return [_row(r) for r in conn.execute("SELECT * FROM orchestration_messages WHERE worker_id=? AND status='PENDING' ORDER BY sequence", (run["worker_id"],))]

    @staticmethod
    def _ack_message(conn, run, message_id):
        message = conn.execute("SELECT worker_id FROM orchestration_messages WHERE message_id=?", (message_id,)).fetchone()
        if not message or message[0] != run["worker_id"]:
            raise PermissionError("Unknown message or foreign worker")
        conn.execute("""UPDATE orchestration_messages SET status='DELIVERED',delivered_run_id=?,delivered_at=?
            WHERE message_id=? AND status='PENDING'""", (run["run_id"], time.time(), message_id))

    def ack_message(self, run_id, owner_session_id, lease_token, message_id):
        def ack(conn):
            self._ack_message(conn, self._lease(conn, run_id, owner_session_id, lease_token), message_id)
        self.db._execute_write(ack)

    def recover_expired_runs(self, owner_session_id):
        def recover(conn):
            expired = conn.execute("""SELECT r.run_id FROM orchestration_runs r JOIN orchestration_workers w
                ON r.worker_id=w.worker_id WHERE w.owner_session_id=? AND r.status='RUNNING' AND r.lease_expires_at<=?""",
                (owner_session_id, time.time())).fetchall()
            for row in expired:
                run = self._run(conn, row[0], owner_session_id)
                uncertain = int(run["tool_inflight"])
                conn.execute("""UPDATE orchestration_runs SET status='INTERRUPTED',uncertain_side_effect=?,
                    lease_token=NULL,lease_expires_at=NULL,result=?,updated_at=? WHERE run_id=?""",
                    (uncertain, json.dumps({"reason": "execution_lease_expired", "uncertain_side_effect": bool(uncertain)}), time.time(), row[0]))
                conn.execute("UPDATE orchestration_workers SET uncertain_side_effect=?,updated_at=? WHERE worker_id=?", (uncertain, time.time(), run["worker_id"]))
            return [self._run(conn, r[0], owner_session_id) for r in expired]
        return self.db._execute_write(recover)

    def reconcile_run(self, run_id, owner_session_id):
        def reconcile(conn):
            run = self._run(conn, run_id, owner_session_id)
            if run["status"] not in _TERMINAL:
                raise ValueError("Only a terminal run can be reconciled")
            latest = conn.execute("SELECT run_id FROM orchestration_runs WHERE worker_id=? AND status IN ('INTERRUPTED','FAILED','CANCELLED','SUCCEEDED') ORDER BY sequence DESC LIMIT 1", (run["worker_id"],)).fetchone()
            if not latest or latest[0] != run_id:
                raise ValueError("Reconcile the latest terminal run")
            conn.execute("UPDATE orchestration_workers SET uncertain_side_effect=0,updated_at=? WHERE worker_id=?", (time.time(), run["worker_id"]))
        self.db._execute_write(reconcile)

    def pending_completions(self, owner_session_id):
        with self.db._read_ctx() as conn:
            return [_row(r) for r in conn.execute("""SELECT r.* FROM orchestration_runs r JOIN orchestration_workers w
                ON r.worker_id=w.worker_id WHERE w.owner_session_id=? AND r.completion_ack=0
                AND r.status IN ('SUCCEEDED','FAILED','INTERRUPTED','CANCELLED') ORDER BY r.sequence""", (owner_session_id,))]

    def ack_completion(self, run_id, owner_session_id):
        def ack(conn):
            run = self._run(conn, run_id, owner_session_id)
            if run["status"] not in _TERMINAL:
                raise ValueError("A running task has no completion to acknowledge")
            conn.execute("UPDATE orchestration_runs SET completion_ack=1 WHERE run_id=?", (run_id,))
        self.db._execute_write(ack)
