"""Worker tool ceilings at the real registry and deferred-call boundaries."""

import json
from types import SimpleNamespace

from agent.delegation_model_routing import ToolPolicy
from tools.delegate_tool_toolsets import _apply_exact_tool_policy


def _schema(name):
    return {
        "name": name,
        "description": "Synthetic worker tool.",
        "parameters": {"type": "object", "properties": {}},
    }


def _tool(name):
    return {"type": "function", "function": _schema(name)}


def test_model_dispatch_forwards_worker_limit_to_execute_code(monkeypatch):
    """The ordinary model_tools signature reaches execute_code's nested RPC scope."""
    import model_tools
    from tools import code_execution_tool

    captured = {}

    def fake_execute_code(**kwargs):
        captured.update(kwargs)
        return json.dumps({"ok": True})

    monkeypatch.setattr(code_execution_tool, "execute_code", fake_execute_code)
    result = model_tools.handle_function_call(
        "execute_code",
        {"code": "print('bounded')"},
        task_id="worker-task",
        session_id="worker-session",
        enabled_tools=["read_file"],
        worker_max_tool_calls=2,
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    )

    assert json.loads(result) == {"ok": True}
    assert captured["enabled_tools"] == ["read_file"]
    assert captured["worker_max_tool_calls"] == 2


def test_model_dispatch_accepts_worker_limit_for_ordinary_registry_tool():
    """R1's original call contract no longer raises before a native handler."""
    import model_tools
    from tools.registry import registry

    name = "pytest_worker_native_read"
    registry.register(
        name=name,
        toolset="pytest-worker",
        schema=_schema(name),
        handler=lambda _args, **_kwargs: json.dumps({"called": True}),
    )
    try:
        result = model_tools.handle_function_call(
            name,
            {},
            worker_max_tool_calls=1,
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        )
        assert json.loads(result) == {"called": True}
    finally:
        registry.deregister(name)


def test_exact_policy_intersects_ancestor_profile_and_empty_request():
    child = SimpleNamespace(
        valid_tool_names={"read_file", "write_file", "execute_code", "delegate_task"},
        tools=[_tool(name) for name in ("read_file", "write_file", "execute_code", "delegate_task")],
    )
    _apply_exact_tool_policy(
        child,
        ToolPolicy(allowed_tools=("read_file", "write_file")),
        request_blocked_tools=["write_file"],
        ancestor_allowed_tools={"read_file", "write_file", "delegate_task"},
    )
    assert child.valid_tool_names == {"read_file", "delegate_task"}

    empty = SimpleNamespace(
        valid_tool_names={"read_file", "delegate_task"},
        tools=[_tool("read_file"), _tool("delegate_task")],
    )
    _apply_exact_tool_policy(
        empty,
        ToolPolicy(),
        request_toolsets=[],
        ancestor_allowed_tools={"read_file", "delegate_task"},
    )
    assert empty.valid_tool_names == {"delegate_task"}
    assert [item["function"]["name"] for item in empty.tools] == ["delegate_task"]


def test_deferred_mcp_call_cannot_bypass_exact_worker_names():
    from agent.tool_executor import _parse_tool_call
    from tools.registry import registry

    name = "mcp__pytest_worker__private_read"
    registry.register(
        name=name,
        toolset="mcp-pytest-worker",
        schema=_schema(name),
        handler=lambda _args, **_kwargs: json.dumps({"leaked": True}),
    )
    try:
        agent = SimpleNamespace(
            enabled_toolsets=["mcp-pytest-worker"],
            disabled_toolsets=[],
            valid_tool_names={"read_file"},
        )
        call = SimpleNamespace(
            id="mcp-denied",
            function=SimpleNamespace(
                name="tool_call",
                arguments=json.dumps({"name": name, "arguments": {}}),
            ),
        )
        parsed = _parse_tool_call(agent, call)
        assert parsed.name == "tool_call"
        assert "not available in this session" in parsed.scope_block
    finally:
        registry.deregister(name)
