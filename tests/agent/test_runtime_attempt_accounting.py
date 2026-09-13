"""Attempt receipts are cumulative, idempotent, and not HTTP request guesses."""

from dataclasses import replace
import sqlite3

import pytest

from agent.runtime_api import RuntimeUsageReceipt
from hermes_state import SessionDB


def receipt(attempt="attempt-a", **kwargs):
    return RuntimeUsageReceipt(runtime_id="example-runtime", provider="example",
        model="example-model", billing_mode="subscription_included", cost_status="included",
        correlation_id="one-host-turn", attempt_id=attempt, input_tokens=10, output_tokens=5, **kwargs)


def test_distinct_attempts_same_turn_and_duplicate_totals_apply_once(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="synthetic", source="cli")
        a = receipt(runtime_turn_count=17)
        assert db.record_runtime_usage_receipt("synthetic", a, project_usage=True)
        assert not db.record_runtime_usage_receipt("synthetic", a, project_usage=True)
        with pytest.raises(ValueError, match="conflicting"):
            db.record_runtime_usage_receipt("synthetic", replace(a, output_tokens=8), project_usage=True)
        assert db.record_runtime_usage_receipt("synthetic", receipt("attempt-b"), project_usage=True)
        rows = db.list_runtime_usage_receipts("synthetic")
        assert len(rows) == 2 and rows[0].runtime_turn_count == 17
        assert rows[0].request_count is None
        state = db.get_session("synthetic")
        assert state["input_tokens"] == 20 and state["output_tokens"] == 10
        assert state["api_call_count"] == 0  # known subtotal only
        assert db.runtime_request_accounting("synthetic") == {
            "attempts": 2, "request_count": None, "known_requests": 0, "request_count_exact": False}
    finally:
        db.close()


def test_known_zero_and_known_requests_are_distinct_from_unknown(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="synthetic", source="cli")
        db.record_runtime_usage_receipt("synthetic", receipt(request_count=0), project_usage=True)
        assert db.runtime_request_accounting("synthetic")["request_count"] == 0
        db.record_runtime_usage_receipt("synthetic", receipt("b", request_count=2), project_usage=True)
        assert db.runtime_request_accounting("synthetic")["request_count"] == 2
        db.record_runtime_usage_receipt("synthetic", receipt("c"), project_usage=True)
        assert db.runtime_request_accounting("synthetic")["request_count"] is None
        assert db.runtime_request_accounting("synthetic")["known_requests"] == 2
    finally:
        db.close()


def test_legacy_correlation_dedup_remains_separate_from_new_attempts(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    db.create_session(session_id="synthetic", source="cli")
    legacy = receipt(None)
    assert db.record_runtime_usage_receipt("synthetic", legacy)
    assert not db.record_runtime_usage_receipt("synthetic", legacy)
    db.close()
    # Reproduce the prior index shape; opening the successor reconciles it
    # without backfilling false attempt identities onto historical rows.
    with sqlite3.connect(path) as conn:
        conn.execute("DROP INDEX idx_runtime_usage_receipts_legacy_correlation")
        conn.execute("CREATE UNIQUE INDEX idx_runtime_usage_receipts_correlation ON runtime_usage_receipts(session_id,runtime_id,correlation_id) WHERE correlation_id IS NOT NULL")
    db = SessionDB(db_path=path)
    try:
        assert db.record_runtime_usage_receipt("synthetic", receipt())
        assert len(db.list_runtime_usage_receipts("synthetic")) == 2
    finally:
        db.close()
