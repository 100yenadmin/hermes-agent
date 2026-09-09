"""Focused increment-3 proof over the real Kanban and WorkerStore owners."""

from contextvars import ContextVar
import json
from types import SimpleNamespace
import threading

import pytest

from agent.shared_discovery import build_local_discovery_scope
from agent.team_orchestration import TeamOrchestrationService
from agent.worker_interfaces import (
    CANONICAL_TEAM_TOOL,
    InterfaceSelection,
    bind_worker_interface,
    canonical_worker_capability,
    dispatch_worker_interface_call,
    normalize_worker_call,
    project_worker_tool_definitions,
)
from agent.worker_store import WorkerStore
from gateway import hosted_rooms
from hermes_state import SessionDB
from tui_gateway.session_discovery import build_gateway_discovery_scope


TEAM_TOOLS = {
    "kanban_team", "delegate_task", "kanban_create", "kanban_heartbeat",
    "kanban_request_review", "kanban_request_changes", "kanban_complete",
    "kanban_block",
}


class FakeLifecycle:
    """Scheduling stub; persistence and task state remain real in these tests."""

    def __init__(self):
        self.admissions = {}
        self.statuses = {}
        self.messages = []
        self.sequence = 0

    def admit_team_execution(self, request, *, worker_id, request_id, previous_run_id=None):
        immutable = (request.goal, request.context, request.profile, previous_run_id)
        if request_id in self.admissions:
            worker, run, expected = self.admissions[request_id]
            if immutable != expected:
                raise ValueError("conflicting deterministic admission")
            return worker, run
        self.sequence += 1
        run_id = f"run-synthetic-{self.sequence}"
        worker = {"worker_id": worker_id, "profile": request.profile}
        run = {"run_id": run_id, "status": "PENDING"}
        self.statuses[run_id] = "PENDING"
        self.admissions[request_id] = (worker, run, immutable)
        return worker, run

    def schedule_team_execution(self, worker_id, run_id):
        self.statuses[run_id] = "RUNNING"
        return {"worker_id": worker_id, "run_id": run_id, "status": "RUNNING"}

    def control(self, action, *, worker_id, run_id=None, **kwargs):
        if action == "wait":
            return {"worker_id": worker_id, "run_id": run_id, "status": self.statuses[run_id]}
        if action == "message":
            self.messages.append((worker_id, run_id, kwargs["message"]))
            return {"accepted": True, "delivery": "RUNNING_STEER_PENDING_CHECKPOINT"}
        if action == "cancel":
            self.statuses[run_id] = "CANCELLED"
            return {"cancel_requested": True}
        raise AssertionError(action)

    def succeed(self, run_ref):
        self.statuses[run_ref.partition(":")[2]] = "SUCCEEDED"


def _definitions(*extra):
    names = ["delegate_task", "kanban_team", *extra]
    return [
        {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
        for name in names
    ]


def _agent(scope, *, tools=TEAM_TOOLS):
    return SimpleNamespace(
        session_id="owner-synthetic",
        _shared_discovery_scope=scope,
        _worker_effective_tool_names=set(tools),
        _executable_tool_names=set(tools),
        valid_tool_names=set(tools),
        provider="fixture",
        model="fixture",
    )


def _service(tmp_path, monkeypatch, *, tools=TEAM_TOOLS):
    home = tmp_path / "home"
    home.mkdir(parents=True)
    board_db = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(board_db))
    scope = build_local_discovery_scope()
    service = TeamOrchestrationService(_agent(scope, tools=tools))
    lifecycle = FakeLifecycle()
    service.lifecycle = lifecycle
    service._start_monitor = lambda *args, **kwargs: None
    return service, lifecycle, board_db


def test_team_schema_projection_is_collision_safe_and_keeps_native_authority(monkeypatch):
    selection = bind_worker_interface(
        InterfaceSelection("codex", "explicit", "experimental_unqualified", "fixture", "fixture"),
        _definitions("team_task"),
    )
    aliases = dict(selection.aliases)
    assert aliases["team_task"] == "hermes_worker_team_task"
    projected = project_worker_tool_definitions(_definitions("team_task"), selection)
    names = [item["function"]["name"] for item in projected]
    assert names.count("team_task") == 1
    assert "hermes_worker_team_task" in names
    assert canonical_worker_capability(selection, "team_task") == "team_task"
    assert canonical_worker_capability(selection, "hermes_worker_team_task") == CANONICAL_TEAM_TOOL
    call = normalize_worker_call(
        selection, "hermes_worker_team_task", {"action": "start", "task_ref": "task:one"},
    )
    assert call.operation == "team"
    assert call.arguments == {"action": "start", "task_ref": "task:one"}

    parent = _agent(SimpleNamespace())
    parent._worker_interface_selection = selection

    class Routed:
        def __init__(self, _agent):
            pass

        def dispatch(self, arguments):
            return {"routed": arguments["task_ref"]}

    monkeypatch.setattr("agent.team_orchestration.TeamOrchestrationService", Routed)
    payload = json.loads(dispatch_worker_interface_call(
        parent, "hermes_worker_team_task", {"action": "start", "task_ref": "task:one"},
        lambda _args: pytest.fail("team call reached delegate dispatch"),
    ))
    assert payload["routed"] == "task:one"
    assert payload["orchestration_interface"]["canonical_tool"] == "kanban_team"


def test_parent_mode_claim_attachment_and_worker_admission_are_immutable(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    conn = kbc.connect()
    try:
        task_id = kb.create_task(
            conn, title="Parent run", assignee="alpha", execution_mode="parent",
        )
        task = kb.claim_task(conn, task_id, claimer="parent-lock", expected_execution_mode="parent")
        assert task is not None and task.execution_mode == "parent"
        assert kb.claim_task(
            conn, task_id, claimer="dispatcher", expected_execution_mode="dispatcher",
        ) is None
        assert not kbd._has_spawnable(conn, "ready")
        reference = {
            "version": "kanban-team-v1", "worker_ref": "worker:one", "run_ref": "run:one",
            "admission_hash": "a" * 64, "role": "implementer", "profile": "alpha",
        }
        attached = kb.attach_execution_reference(
            conn, task_id, task.current_run_id, claim_lock="parent-lock", reference=reference,
        )
        assert attached == reference
        assert kb.attach_execution_reference(
            conn, task_id, task.current_run_id, claim_lock="parent-lock", reference=reference,
        ) == reference
        with pytest.raises(ValueError, match="different execution"):
            kb.attach_execution_reference(
                conn, task_id, task.current_run_id, claim_lock="parent-lock",
                reference={**reference, "run_ref": "run:other"},
            )
    finally:
        conn.close()

    db = SessionDB(tmp_path / "state.db")
    try:
        store = WorkerStore(db)
        store.ensure_schema()
        worker, run = store.admit_team_run(
            "owner", worker_id="worker-one", request_id="request-one", profile="alpha",
            policy={"role": "leaf"}, frozen_prompt="fixed", goal="Implement", context="Task",
        )
        repeated = store.admit_team_run(
            "owner", worker_id="worker-one", request_id="request-one", profile="alpha",
            policy={"role": "leaf"}, frozen_prompt="fixed", goal="Implement", context="Task",
        )
        assert repeated[1]["run_id"] == run["run_id"]
        assert worker["worker_id"] == "worker-one"
        with pytest.raises(ValueError, match="immutable request"):
            store.admit_team_run(
                "owner", worker_id="worker-one", request_id="request-one", profile="alpha",
                policy={"role": "leaf"}, frozen_prompt="fixed", goal="Changed", context="Task",
            )
    finally:
        db.close()


def test_two_worker_dependency_review_rejection_retained_correction_and_acceptance(tmp_path, monkeypatch):
    service, lifecycle, _board_db = _service(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    first = service.dispatch({
        "action": "create", "title": "Build", "profile": "alpha", "idempotency_key": "build",
    })
    dependent = service.dispatch({
        "action": "create", "title": "Package", "profile": "beta",
        "parent_refs": [first["task_ref"]], "idempotency_key": "package",
    })
    assert first["status"] == "ready" and dependent["status"] == "todo"

    implementation = service.dispatch({"action": "start", "task_ref": first["task_ref"]})
    assert implementation["worker_status"] == "RUNNING"
    guidance = service.dispatch({
        "action": "guide", "targets": [first["task_ref"]], "message": "Preserve the API.",
        "idempotency_key": "guide-one",
    })
    assert guidance["outcomes"][0]["delivery"] == "RUNNING_STEER_PENDING_CHECKPOINT"
    lifecycle.succeed(implementation["run_ref"])
    submitted = service.dispatch({
        "action": "submit_review", "task_ref": first["task_ref"],
        "summary": "Implemented and checked.", "reviewer": "beta",
    })
    assert submitted["status"] == "review"

    reviewer = service.dispatch({"action": "start", "task_ref": first["task_ref"]})
    assert reviewer["profile"] == "beta"
    lifecycle.succeed(reviewer["run_ref"])
    correction = service.dispatch({
        "action": "request_changes", "task_ref": first["task_ref"],
        "message": "Correct the boundary condition.",
    })
    assert correction["worker_ref"] == implementation["worker_ref"]
    assert correction["run_ref"] != implementation["run_ref"]
    correction_admission = next(
        row for row in lifecycle.admissions.values() if row[1]["run_id"] == correction["run_ref"].partition(":")[2]
    )
    assert correction_admission[2][3] == implementation["run_ref"].partition(":")[2]
    lifecycle.succeed(correction["run_ref"])
    assert service.dispatch({
        "action": "submit_review", "task_ref": first["task_ref"],
        "summary": "Correction complete.", "reviewer": "beta",
    })["status"] == "review"
    second_review = service.dispatch({"action": "start", "task_ref": first["task_ref"]})
    lifecycle.succeed(second_review["run_ref"])
    accepted = service.dispatch({
        "action": "accept", "task_ref": first["task_ref"], "summary": "Accepted.",
    })
    assert accepted["status"] == "done"

    conn = kbc.connect()
    try:
        assert kb.get_task(conn, dependent["task_ref"].partition(":")[2]).status == "ready"
        roles = [
            event.payload.get("role")
            for event in kb.list_events(conn, first["task_ref"].partition(":")[2])
            if event.kind == "execution_attached"
        ]
        assert roles == ["implementer", "reviewer", "correction", "reviewer"]
    finally:
        conn.close()


def test_cancel_waits_for_terminal_worker_and_list_only_policy_cannot_mutate(tmp_path, monkeypatch):
    denied, _lifecycle, _ = _service(tmp_path, monkeypatch, tools={"kanban_list"})
    result = denied.dispatch({"action": "create", "title": "Denied", "profile": "alpha"})
    assert "current executable tool policy" in result["error"]

    service, lifecycle, _ = _service(tmp_path / "allowed", monkeypatch)
    created = service.dispatch({"action": "create", "title": "Cancel", "profile": "alpha"})
    running = service.dispatch({"action": "start", "task_ref": created["task_ref"]})
    cancelled = service.dispatch({
        "action": "cancel", "task_ref": created["task_ref"], "timeout_seconds": 0,
    })
    assert lifecycle.statuses[running["run_ref"].partition(":")[2]] == "CANCELLED"
    assert cancelled["status"] == "blocked"


def test_room_guidance_requires_current_message_grant(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    room_db = tmp_path / "rooms.db"
    hosted_rooms.create_room(
        room_db, room_id="room-one", name="Room", members=[{"profile": "alpha"}],
        authority_gateway_id="gateway-one",
    )
    monkeypatch.setattr(hosted_rooms, "local_authority_gateway_id_existing", lambda: "gateway-one")

    sent = []
    service = SimpleNamespace(
        db_path=room_db,
        runtime=SimpleNamespace(status=lambda: {"running": True, "stopping": False}),
        send=lambda **kwargs: sent.append(kwargs) or {"event_id": kwargs["event_id"]},
    )
    from tui_gateway import methods_groups
    monkeypatch.setattr(methods_groups, "_service", service)
    cfg = {"orchestration": {"discovery": {"rooms": [{
        "id": "room-one", "actions": ["inspect", "message"], "participants": ["alpha"],
    }]}}}
    current = ContextVar("team-room-record", default=None)
    server = SimpleNamespace(
        _sessions={}, _sessions_lock=threading.RLock(), _current_runtime_session_record=current,
        _current_profile_name=lambda: "default", _load_cfg=lambda: cfg,
    )
    scope = build_gateway_discovery_scope(server, sid="sid-one", cfg=cfg, source="tui")
    agent = _agent(scope)
    record = {"agent": agent, "discovery_scope": scope, "source": "tui"}
    server._sessions["sid-one"] = record
    token = current.set(record)
    try:
        result = scope.room_provider.send(
            agent, "room:room-one", event_id="team-event", payload={"text": "Coordinate."},
        )
        assert result["event_id"] == "team-event"
        assert sent[0]["room_id"] == "room-one"
        cfg["orchestration"]["discovery"]["rooms"][0]["actions"] = ["inspect"]
        with pytest.raises(PermissionError, match="Unknown or unavailable"):
            scope.room_provider.send(
                agent, "room:room-one", event_id="other", payload={"text": "Denied."},
            )
    finally:
        current.reset(token)
