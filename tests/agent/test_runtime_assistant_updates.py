"""Real SQLite proof for the acknowledged runtime commentary seam."""

import asyncio
from types import SimpleNamespace

import pytest

from agent.runtime_api import RuntimeAssistantUpdate
from agent.runtime_dispatch import HermesRuntimeHostServices, RuntimeToolPersistenceError
from hermes_state import SessionDB


@pytest.fixture
def bound(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="synthetic-parent", source="cli")
    messages = []
    def flush(rows):
        for row in rows:
            if not row.get("_db_persisted"):
                db.append_message("synthetic-parent", role=row["role"], content=row["content"])
                row["_db_persisted"] = True
        return True
    agent = SimpleNamespace(session_id="synthetic-parent", _session_db=db,
        _flush_messages_to_session_db=flush, tools=[], valid_tool_names=[])
    host = HermesRuntimeHostServices(agent, task_id="synthetic-turn",
        runtime_id="example-runtime", turn_messages=messages, correlation_id="turn-1")
    yield host, db, messages
    db.close()


def test_long_ordered_content_and_identity_dedup_survive_reload(bound):
    host, db, messages = bound
    messages.append({"role": "user", "content": "Synthetic prompt"})
    text = "Unicode λ\n    indented code\n" * 400
    async def run():
        await host.persist_assistant(RuntimeAssistantUpdate("a", 0, text[:4000]))
        await host.persist_assistant(RuntimeAssistantUpdate("a", 1, text[4000:]))
        await host.persist_assistant(RuntimeAssistantUpdate("a", 2, "", "final"))
        await host.persist_assistant(RuntimeAssistantUpdate("a", 2, "", "final"))
        await host.persist_assistant(RuntimeAssistantUpdate("b", 0, text, "final"))
    asyncio.run(run())
    saved = db.get_messages("synthetic-parent")
    assert [row["content"] for row in saved] == ["Synthetic prompt", text, text]
    assert saved[1]["platform_message_id"] != saved[2]["platform_message_id"]
    assert [row["content"] for row in messages] == ["Synthetic prompt", text, text]


def test_update_cannot_rewrite_commentary_after_tool_boundary(bound):
    host, db, messages = bound
    asyncio.run(host.persist_assistant(RuntimeAssistantUpdate("a", 0, "before")))
    db.append_message("synthetic-parent", "tool", "saved result", tool_call_id="tool-1")
    with pytest.raises(RuntimeToolPersistenceError):
        asyncio.run(host.persist_assistant(RuntimeAssistantUpdate("a", 1, "rewrite", "snapshot")))
    assert db.get_messages("synthetic-parent")[0]["content"] == "before"


def test_sequence_conflict_does_not_append_or_change_saved_content(bound):
    host, db, _ = bound
    asyncio.run(host.persist_assistant(RuntimeAssistantUpdate("a", 0, "first")))
    with pytest.raises(RuntimeToolPersistenceError):
        asyncio.run(host.persist_assistant(RuntimeAssistantUpdate("a", 0, "different")))
    assert [row["content"] for row in db.get_messages("synthetic-parent")] == ["first"]


def test_failed_prior_flush_produces_no_ack_or_new_row(bound):
    host, db, _ = bound
    host._agent._flush_messages_to_session_db = lambda rows: False
    with pytest.raises(RuntimeToolPersistenceError):
        asyncio.run(host.persist_assistant(RuntimeAssistantUpdate("a", 0, "not saved")))
    assert db.get_messages("synthetic-parent") == []
