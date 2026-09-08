"""Storage contracts against real profile-scoped SQLite, including reopen recovery."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from agent.worker_store import WorkerStore
from hermes_state import SessionDB


@pytest.fixture
def store(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    storage = WorkerStore(db)
    storage.ensure_schema()
    yield storage
    db.close()


def worker(store, owner="owner", **kwargs):
    return store.create_worker(owner, profile="research", policy={"allowed_toolsets": ["file"]}, **kwargs)["worker_id"]


def test_fifo_and_owner_scoped_concurrency_are_transactional(store):
    first, second = worker(store), worker(store)
    run = store.enqueue_run(first, "owner", goal="first", request_id="one")
    assert store.enqueue_run(first, "owner", goal="first", request_id="one")["run_id"] == run["run_id"]
    with pytest.raises(ValueError):
        store.enqueue_run(first, "owner", goal="different", request_id="one")
    followup = store.enqueue_run(first, "owner", goal="next")
    store.enqueue_run(second, "owner", goal="other")
    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(lambda wid: store.claim_next_run(wid, "owner", max_concurrent=1), [first, second]))
    assert sum(c is not None for c in claims) == 1
    selected = next(c for c in claims if c is not None)
    assert store.claim_next_run(selected["worker_id"], "owner") is None
    store.finish_run(selected["run_id"], "owner", selected["lease_token"], status="SUCCEEDED", result={"summary": "done"})
    if selected["worker_id"] != first:
        claimed = store.claim_next_run(first, "owner")
        store.finish_run(claimed["run_id"], "owner", claimed["lease_token"], status="SUCCEEDED", result={})
    assert store.claim_next_run(first, "owner")["run_id"] == followup["run_id"]


def test_foreign_owner_cannot_read_message_claim_or_reparent(store):
    wid = worker(store)
    run = store.enqueue_run(wid, "owner", goal="private")
    actions = [
        lambda: store.get_worker(wid, "foreign"),
        lambda: store.get_run(run["run_id"], "foreign"),
        lambda: store.enqueue_message(wid, "foreign", "inject"),
        lambda: store.claim_next_run(wid, "foreign"),
        lambda: store.create_worker("foreign", parent_worker_id=wid),
    ]
    for action in actions:
        with pytest.raises(PermissionError):
            action()
    assert store.list_workers("foreign") == []
    assert store.pending_completions("foreign") == []
    child = store.get_worker(worker(store, parent_worker_id=wid), "owner")
    assert child["root_worker_id"] == wid and child["depth"] == 2


def test_message_ack_is_atomic_with_checkpoint_and_retries_are_idempotent(store):
    wid = worker(store)
    queued = store.enqueue_run(wid, "owner", goal="work")
    run = store.claim_next_run(wid, "owner")
    msg = store.enqueue_message(wid, "owner", "important", message_id="dedup")
    assert store.enqueue_message(wid, "owner", "important", message_id="dedup") == msg
    assert [m["message_id"] for m in store.claim_messages(run["run_id"], "owner", run["lease_token"])] == ["dedup"]
    history = [{"role": "user", "content": "work"}, {"role": "assistant", "content": "important received"}]
    with pytest.raises(PermissionError):
        store.checkpoint_run(run["run_id"], "owner", run["lease_token"], history=history, delivered_message_ids=["dedup", "foreign"])
    assert store.get_worker(wid, "owner")["history"] == []
    assert len(store.claim_messages(run["run_id"], "owner", run["lease_token"])) == 1
    store.checkpoint_run(run["run_id"], "owner", run["lease_token"], history=history, delivered_message_ids=["dedup"])
    assert store.claim_messages(run["run_id"], "owner", run["lease_token"]) == []
    assert store.get_worker(wid, "owner")["history"] == history
    assert queued["run_id"] == run["run_id"]


@pytest.mark.parametrize("tool_inflight", [False, True])
def test_restart_fences_old_executor_and_preserves_conversation_and_completions(tmp_path, monkeypatch, tool_inflight):
    import agent.worker_store as module

    now = [1000.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0]))
    path = tmp_path / "state.db"
    db = SessionDB(path)
    first = WorkerStore(db)
    first.ensure_schema()
    wid = worker(first, frozen_prompt="fixed")
    first.enqueue_run(wid, "owner", goal="work")
    run = first.claim_next_run(wid, "owner", lease_seconds=10)
    first.enqueue_message(wid, "owner", "undelivered", message_id="pending")
    history = [{"role": "user", "content": "work"}]
    first.checkpoint_run(run["run_id"], "owner", run["lease_token"], history=history, tool_inflight=tool_inflight)
    db.close()
    now[0] += 11
    reopened = SessionDB(path)
    try:
        recovered = WorkerStore(reopened)
        recovered.ensure_schema()
        records = recovered.recover_expired_runs("owner")
        assert len(records) == 1 and records[0]["status"] == "INTERRUPTED"
        assert records[0]["uncertain_side_effect"] is tool_inflight
        assert recovered.get_worker(wid, "owner")["history"] == history
        with pytest.raises(PermissionError):
            recovered.heartbeat_run(run["run_id"], "owner", run["lease_token"])
        if tool_inflight:
            with pytest.raises(ValueError):
                recovered.enqueue_run(wid, "owner", goal="resume", previous_run_id=run["run_id"])
            recovered.reconcile_run(run["run_id"], "owner")
        next_run = recovered.enqueue_run(wid, "owner", goal="resume", previous_run_id=run["run_id"])
        active = recovered.claim_next_run(wid, "owner")
        assert active["run_id"] == next_run["run_id"] != run["run_id"]
        assert recovered.claim_messages(active["run_id"], "owner", active["lease_token"])[0]["message_id"] == "pending"
        assert len(recovered.pending_completions("owner")) == 1
        recovered.ack_completion(run["run_id"], "owner")
        recovered.ack_completion(run["run_id"], "owner")
        assert recovered.pending_completions("owner") == []
    finally:
        reopened.close()


def test_credentials_rejected_and_invalid_lease_cannot_change_history(store):
    with pytest.raises(ValueError, match="credentials"):
        store.create_worker("owner", policy={"route": {"api_key": "synthetic-secret"}})
    wid = worker(store)
    run = store.enqueue_run(wid, "owner", goal="work")
    store.claim_next_run(wid, "owner")
    with pytest.raises(PermissionError):
        store.checkpoint_run(run["run_id"], "owner", "wrong-token", history=[{"role": "user", "content": "changed"}])
    assert store.get_worker(wid, "owner")["history"] == []


def test_tool_boundary_is_fenced_without_rewriting_history(store):
    wid = worker(store, frozen_prompt="fixed")
    store.enqueue_run(wid, "owner", goal="work")
    run = store.claim_next_run(wid, "owner")
    store.mark_tool_boundary(run["run_id"], "owner", run["lease_token"], tool_inflight=True)
    assert store.get_run(run["run_id"], "owner")["tool_inflight"] is True
    assert store.get_worker(wid, "owner")["history"] == []
    store.mark_tool_boundary(run["run_id"], "owner", run["lease_token"], tool_inflight=False)
    assert store.get_run(run["run_id"], "owner")["tool_inflight"] is False


def test_parallel_tool_checkpoint_keeps_other_inflight_effect_uncertain(tmp_path, monkeypatch):
    import agent.worker_store as module

    now = [2000.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0]))
    path = tmp_path / "state.db"
    db = SessionDB(path)
    first = WorkerStore(db)
    first.ensure_schema()
    wid = worker(first)
    first.enqueue_run(wid, "owner", goal="parallel")
    run = first.claim_next_run(wid, "owner", lease_seconds=10)
    first.mark_tool_boundary(run["run_id"], "owner", run["lease_token"], tool_inflight=True)
    first.mark_tool_boundary(run["run_id"], "owner", run["lease_token"], tool_inflight=True)
    history = [{"role": "tool", "tool_call_id": "a", "content": "done"}]
    first.checkpoint_tool_result(
        run["run_id"], "owner", run["lease_token"], history=history, settled=True)
    snapshot = first.get_run(run["run_id"], "owner")
    assert snapshot["tool_inflight_count"] == 1
    assert snapshot["tool_inflight"] is True
    db.close()

    now[0] += 11
    reopened = SessionDB(path)
    try:
        recovered = WorkerStore(reopened)
        recovered.ensure_schema()
        interrupted = recovered.recover_expired_runs("owner")[0]
        assert interrupted["status"] == "INTERRUPTED"
        assert interrupted["uncertain_side_effect"] is True
        assert recovered.get_worker(wid, "owner")["history"] == history
    finally:
        reopened.close()
