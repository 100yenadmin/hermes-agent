"""Private publication admission shares the invocation's host cancellation fence."""

from types import SimpleNamespace

import pytest

from agent.conversation_compression import (
    CompressionCommitFence,
    _claim_compressor_attempt,
    _clear_compression_cancelled_check_if_owner,
    _install_compression_cancelled_check,
    _run_summary_dispatch,
)


@pytest.mark.parametrize("cancel", ["cancel_before_commit", "revoke_commit_admission", "deadline"])
def test_dispatch_publication_is_repeatable_until_cancel_and_cleans_up(cancel, monkeypatch):
    compressor = SimpleNamespace()
    agent = SimpleNamespace(context_compressor=compressor, session_id="synthetic")
    generation = _claim_compressor_attempt(compressor)
    fence = CompressionCommitFence()
    published = []
    messages = [{"role": "user", "content": "synthetic"}]

    def compress(incoming):
        # A real consumer captures this once, before any provider work.
        publication = compressor._compression_publication_fence
        assert publication is fence
        for page in range(2):
            assert publication.begin_lock_setup()
            try:
                # Cancellation cannot win halfway through this short transaction.
                assert fence.try_cancel_before_commit() is None
                assert not fence.commit_in_flight
                published.append(page)
            finally:
                publication.finish_lock_setup()
        if cancel == "deadline":
            fence.set_total_ceiling_seconds(10)
            monkeypatch.setattr("agent.conversation_compression.time.monotonic",
                                lambda: fence.deadline_monotonic + 1)
        else:
            result = getattr(fence, cancel)()
            if cancel == "cancel_before_commit":
                assert result is True  # Publication never set sticky commit_started.
        assert compressor._compression_cancelled_check()
        assert not publication.begin_lock_setup()
        return incoming

    assert _run_summary_dispatch(
        agent, messages, compress, {}, commit_fence=fence,
        attempt_generation=generation, hard_cancel_event=None,
    ) is messages
    assert published == [0, 1]
    assert compressor._compression_publication_fence is None
    assert compressor._compression_cancelled_check is None
    assert compressor._compression_cancelled_check_owner is None


def test_late_dispatch_unwind_preserves_newer_invocation_publication_fence():
    compressor = SimpleNamespace()
    agent = SimpleNamespace(context_compressor=compressor, session_id="synthetic")
    old_generation = _claim_compressor_attempt(compressor)
    old_fence, new_fence = CompressionCommitFence(), CompressionCommitFence()
    new_generation = None

    def detached_compress(messages):
        nonlocal new_generation
        captured = compressor._compression_publication_fence
        assert captured is old_fence
        assert old_fence.cancel_before_commit()
        new_generation = _claim_compressor_attempt(compressor)
        _install_compression_cancelled_check(
            compressor, lambda: new_fence.is_cancelled, new_generation,
            publication_fence=new_fence,
        )
        assert not captured.begin_lock_setup()
        raise RuntimeError("old invocation unwound")

    with pytest.raises(RuntimeError, match="old invocation unwound"):
        _run_summary_dispatch(
            agent, [], detached_compress, {}, commit_fence=old_fence,
            attempt_generation=old_generation, hard_cancel_event=None,
        )
    assert compressor._compression_publication_fence is new_fence
    assert not compressor._compression_cancelled_check()
    assert compressor._compression_cancelled_check_owner == new_generation
    assert new_fence.begin_lock_setup()
    new_fence.finish_lock_setup()
    assert _clear_compression_cancelled_check_if_owner(compressor, new_generation)
    assert compressor._compression_publication_fence is None
