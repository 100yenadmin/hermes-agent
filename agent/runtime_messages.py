"""Acknowledged visible-message persistence for whole-turn runtime plugins."""

import hashlib
import json


def persist_assistant(host, update):
    """Return only after prior rows and this update are durably saved.

    The caller holds the host delivery lock. Tool execution uses the same live
    transcript and cannot manufacture a durability acknowledgment itself.
    """
    from agent.runtime_api import RuntimeAssistantUpdate
    from agent.runtime_dispatch import RuntimeToolPersistenceError

    if not isinstance(update, RuntimeAssistantUpdate):
        raise TypeError("invalid assistant update")
    agent = host._agent
    database = getattr(agent, "_session_db", None)
    messages = host._turn_messages
    flush = getattr(agent, "_flush_messages_to_session_db", None)
    if database is None or messages is None or not callable(flush):
        raise RuntimeToolPersistenceError("assistant updates require durable Hermes state")
    identity = "runtime-assistant-" + hashlib.sha256(json.dumps(
        [host._runtime_id, host._turn_namespace, update.message_id],
    ).encode()).hexdigest()
    try:
        if flush(messages) is False:
            raise RuntimeToolPersistenceError("prior transcript flush failed")
        saved = database.apply_runtime_assistant_update(
            host._parent_session_id, identity=identity, update=update,
            compression_lock_holder=getattr(agent, "_active_compression_lock_holder", None),
            turn_lease_holder=getattr(agent, "_active_session_turn_lease_holder", None),
            turn_lease_ttl_seconds=getattr(agent, "_active_session_turn_lease_ttl_seconds", 300) or 300,
        )
    except Exception as exc:
        agent._incremental_persistence_failed = True
        agent._last_persistence_error_cause = "runtime_assistant"
        raise RuntimeToolPersistenceError("runtime assistant persistence failed") from exc
    existing = next((m for m in messages if m.get("platform_message_id") == identity), None)
    if existing is None:
        existing = {"role": "assistant", "platform_message_id": identity}
        messages.append(existing)
    existing.update(content=saved["content"], _row_id=saved["id"], _db_persisted=True)
    # Failure classification includes observed durable commentary, even though
    # no duplicate public content event is required to make it persistent.
    host._side_effect_count += int(saved["changed"])

