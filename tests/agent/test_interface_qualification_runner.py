import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from scripts.evals import interface_qualification as runner
from scripts.evals.interface_qualification import load_fixture, run_suite


def _fake_observation(scenario_id, secret):
    common = {
        "parent_provider": "provider-parent",
        "parent_model": "model-parent",
        "selected_interface": "codex",
        "interface_version": "worker-interface-v1",
        "advertised_tool_names": ["worker_capabilities", "spawn_agent"],
        "parent_completed": True,
        "elapsed_seconds": 1.25,
        "token_usage": {"input_tokens": 120, "output_tokens": 30, "provider_blob": secret},
        "cost": {"status": "unknown"},
        "wire_route_agreement": True,
        "tool_receipts_match": True,
        "duplicate_owned_executions": 0,
        "authorization_violations": 0,
        "malformed_tool_calls": 0,
        "corrective_turns": 0,
        "provider_body": secret,
    }
    if scenario_id == "two_provider_retained_workflow":
        return {
            **common,
            "events": [
                {"tool_name": "spawn_agent", "profiles": ["route-a", "route-b"], "accepted": True},
                {
                    "tool_name": "send_message", "accepted": True,
                    "effective_action": "message", "target_was_running": True,
                },
                {
                    "tool_name": "followup_task", "accepted": True,
                    "effective_action": "resume", "linked_followup": True,
                    "followup_avoids_labels": True,
                },
            ],
            "workers": [
                {
                    "profile": "route-a", "provider": "provider-a", "run_count": 2,
                    "all_terminal_acked": True,
                },
                {
                    "profile": "route-b", "provider": "provider-b", "run_count": 1,
                    "all_terminal_acked": True,
                },
            ],
            "retained_answer_match": True,
        }
    if scenario_id == "invalid_profile_rejected":
        return {
            **common,
            "events": [{
                "tool_name": "spawn_agent", "accepted": False,
                "classification": "EXPECTED_INVALID_PROFILE_DENIAL",
            }],
            "workers": [],
            "counts_before": {"workers": 0, "runs": 0, "messages": 0},
            "counts_after": {"workers": 0, "runs": 0, "messages": 0},
        }
    if scenario_id == "foreign_worker_denied":
        return {
            **common,
            "events": [{
                "tool_name": "inspect_agent", "accepted": False,
                "classification": "EXPECTED_FOREIGN_OWNER_DENIAL",
            }],
            "workers": [],
            "foreign_preimage_unchanged": True,
        }
    return {
        **common,
        "events": [
            {
                "tool_name": "send_message", "accepted": True,
                "effective_action": "message", "run_count_delta": 0,
            },
            {
                "tool_name": "followup_task", "accepted": True,
                "effective_action": "resume", "run_count_delta": 1,
                "linked_followup": True,
            },
            {
                "tool_name": "worker_control", "accepted": True,
                "effective_action": "ack", "targets_latest_run": True,
            },
        ],
        "workers": [{
            "profile": "route-a", "provider": "provider-a", "run_count": 2,
            "all_terminal_acked": True,
        }],
        "retained_answer_match": True,
        "prior_run_replayed": False,
    }


def test_fake_transport_qualifies_all_scenarios_without_leaking_transport_data():
    fixture = load_fixture()
    assert runner.REPO_ROOT == Path(runner.__file__).resolve().parents[2]
    assert sys.path[0] == str(runner.REPO_ROOT)
    secret = "synthetic-provider-body-must-not-appear"
    common = {
        "candidate_sha": "a" * 40,
        "parent_provider": "provider-parent",
        "parent_model": "model-parent",
        "requested_interface": "codex",
        "required_receipt_fields": fixture["required_receipt_fields"],
        "limits": {
            "max_parent_iterations": 16,
            "max_child_iterations": 3,
            "max_children": 2,
            "max_seconds_per_scenario": 240,
            "max_child_seconds": 90,
            "max_parent_tokens": 4096,
        },
    }

    report = run_suite(
        fixture,
        lambda scenario: _fake_observation(scenario["id"], secret),
        common,
    )

    assert report["qualified"] is True
    assert [item["scenario_id"] for item in report["receipts"]] == [
        item["id"] for item in fixture["scenarios"]
    ]
    assert all(item["task_completed"] for item in report["receipts"])
    assert all(
        set(fixture["required_receipt_fields"]).issubset(item)
        for item in report["receipts"]
    )
    assert secret not in json.dumps(report)

    def malformed_denial(scenario):
        observation = _fake_observation(scenario["id"], secret)
        if scenario["id"] == "invalid_profile_rejected":
            observation["events"][0]["classification"] = "INVALID_TOOL_CALL"
        return observation

    failed = run_suite(fixture, malformed_denial, common)
    invalid = next(item for item in failed["receipts"] if item["scenario_id"] == "invalid_profile_rejected")
    assert failed["qualified"] is False
    assert invalid["task_completed"] is False
    assert invalid["invalid_tool_calls"]["structured"] == 1


def test_timeout_and_controller_failures_are_sanitized(monkeypatch, capsys):
    secret = "provider-error-body-must-not-appear"
    packet = {"parent_provider": "provider-parent", "parent_model": "model-parent"}

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("qualification-child", 1)

    monkeypatch.setattr(runner.subprocess, "run", timeout)
    receipt = runner._run_child_process(packet, timeout=1)
    assert receipt["parent_completed"] is False
    assert receipt["error_type"] == "ScenarioTimeout"

    def fail_controller():
        raise RuntimeError(secret)

    monkeypatch.setattr(runner, "_main", fail_controller)
    assert runner.main() == 1
    output = capsys.readouterr().out
    assert json.loads(output)["qualification"] == "unqualified"
    assert secret not in output


def test_running_guidance_releases_only_after_an_accepted_message():
    runs = [{"run_id": "run-1", "status": "RUNNING"}]

    class Store:
        def list_runs(self, worker_id, owner):
            assert (worker_id, owner) == ("worker-1", "owner")
            return list(runs)

    responses = [
        {"error": "denied", "orchestration_interface": {"effective_action": "message"}},
        {"success": True, "orchestration_interface": {"effective_action": "resume"}},
        {"success": True, "orchestration_interface": {"effective_action": "message"}},
    ]
    parent = SimpleNamespace(
        _dispatch_worker_interface=lambda _name, _args: json.dumps(responses.pop(0))
    )
    released = []
    events, restore = runner._instrument_interface(
        parent, Store(), "owner", accepted_running_guidance=lambda: released.append(True)
    )
    try:
        parent._dispatch_worker_interface("send_message", {"target": "worker-1"})
        parent._dispatch_worker_interface("followup_task", {"target": "worker-1"})
        assert released == []
        parent._dispatch_worker_interface("send_message", {"target": "worker-1"})
        assert released == [True]
        assert [event["accepted"] for event in events] == [False, True, True]
    finally:
        restore()
