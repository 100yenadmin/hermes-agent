"""Compression ``fallback_policy=none`` stays on the active Codex identity."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _write_config(home, *, provider="openai-codex", model="", fallback_policy="none"):
    (home / "config.yaml").write_text(
        "\n".join(
            (
                "model:",
                "  provider: openai-codex",
                "  default: gpt-6-astra",
                "auxiliary:",
                "  compression:",
                f"    provider: {provider}",
                f"    model: {model!r}",
                f"    fallback_policy: {fallback_policy}",
            )
        )
        + "\n",
        encoding="utf-8",
    )


class _FakeResponses:
    def __init__(self, *, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeOpenAI:
    def __init__(self, responses, token):
        self.responses = responses
        self.api_key = token
        self.base_url = "https://chatgpt.com/backend-api/codex"

    def close(self):
        return None


def _runtime(model="gpt-6-astra", provider="openai-codex", token="synthetic-codex-token"):
    return {
        "provider": provider,
        "model": model,
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_key": token,
        "api_mode": "codex_responses",
    }


@contextmanager
def _fallback_patches(ac):
    patches = (
        patch.object(ac, "_try_configured_fallback_chain", side_effect=AssertionError("configured fallback used")),
        patch.object(ac, "_try_main_fallback_chain", side_effect=AssertionError("main fallback used")),
        patch.object(ac, "_try_discovery_chain", side_effect=AssertionError("discovery fallback used")),
        patch.object(ac, "_try_payment_fallback", side_effect=AssertionError("payment fallback used")),
        patch.object(ac, "_try_main_agent_model_fallback", side_effect=AssertionError("main-agent fallback used")),
        patch.object(ac, "_build_codex_client", side_effect=AssertionError("stored Codex resolver used")),
    )
    started = [item.start() for item in patches]
    try:
        yield started
    finally:
        for item in reversed(patches):
            item.stop()


def test_compression_fallback_policy_defaults_to_legacy_chain():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["auxiliary"]["compression"]["fallback_policy"] == "default"


def test_none_getter_uses_active_astra_and_sol_without_stale_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, model="")
    from agent import auxiliary_client as ac

    for active_model in ("gpt-6-astra", "gpt-5.6-sol"):
        responses = _FakeResponses()
        fake = _FakeOpenAI(responses, f"synthetic-{active_model}")
        with patch.object(ac, "_create_openai_client", return_value=fake) as build:
            with _fallback_patches(ac) as fallback_mocks:
                client, resolved = ac.get_text_auxiliary_client(
                    "compression", main_runtime=_runtime(active_model, token=f"synthetic-{active_model}")
                )
        assert resolved == active_model
        assert client._real_client is fake
        build.assert_called_once()
        assert all(mock.call_count == 0 for mock in fallback_mocks)


def test_none_actual_compression_call_propagates_active_model_to_codex_transport(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, model="")
    from agent import auxiliary_client as ac

    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="active summary")],
            )
        ]
    )
    responses = _FakeResponses(response=response)
    fake = _FakeOpenAI(responses, "synthetic-codex-token")
    with patch.object(ac, "_create_openai_client", return_value=fake):
        with _fallback_patches(ac) as fallback_mocks:
            result = ac.call_llm(
                task="compression",
                main_runtime=_runtime("gpt-6-astra"),
                messages=[{"role": "user", "content": "summarize this"}],
            )
    assert result.choices[0].message.content == "active summary"
    assert responses.calls[0]["model"] == "gpt-6-astra"
    assert all(mock.call_count == 0 for mock in fallback_mocks)


def test_none_async_compression_call_stays_on_active_codex_route(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, model="")
    from agent import auxiliary_client as ac

    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="async active summary")],
            )
        ]
    )
    responses = _FakeResponses(response=response)
    fake = _FakeOpenAI(responses, "synthetic-codex-token")
    with patch.object(ac, "_create_openai_client", return_value=fake):
        with _fallback_patches(ac) as fallback_mocks:
            result = asyncio.run(
                ac.async_call_llm(
                    task="compression",
                    main_runtime=_runtime("gpt-5.6-sol"),
                    messages=[{"role": "user", "content": "summarize this"}],
                )
            )
    assert result.choices[0].message.content == "async active summary"
    assert responses.calls[0]["model"] == "gpt-5.6-sol"
    assert all(mock.call_count == 0 for mock in fallback_mocks)


@pytest.mark.parametrize(
    "error",
    [
        type("Auth401", (Exception,), {"status_code": 401})("unauthorized"),
        type("Payment402", (Exception,), {"status_code": 402})("payment required"),
        type("Rate429", (Exception,), {"status_code": 429})("rate limited"),
        type("APITimeoutError", (Exception,), {})("timed out"),
        type("Model400", (Exception,), {"status_code": 400})("model is not supported with this route"),
        RuntimeError("Auxiliary compression LLM returned invalid response: choices[0].message"),
    ],
    ids=("auth401", "payment402", "rate429", "timeout", "incompatible", "invalid-output"),
)
def test_none_errors_fail_closed_without_any_fallback_transport(tmp_path, monkeypatch, error):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, model="")
    from agent import auxiliary_client as ac

    responses = _FakeResponses(error=error)
    fake = _FakeOpenAI(responses, "synthetic-codex-token")
    with patch.object(ac, "_create_openai_client", return_value=fake):
        with _fallback_patches(ac) as fallback_mocks:
            with pytest.raises(Exception):
                ac.call_llm(
                    task="compression",
                    main_runtime=_runtime("gpt-6-astra"),
                    messages=[{"role": "user", "content": "summarize this"}],
                )
    assert len(responses.calls) >= 1
    assert all(mock.call_count == 0 for mock in fallback_mocks)


@pytest.mark.parametrize(
    "runtime, config_provider, config_model",
    [
        (None, "openai-codex", ""),
        (_runtime("gpt-6-astra", provider="openrouter"), "openai-codex", ""),
        (_runtime("gpt-6-astra"), "openrouter", ""),
        (_runtime("gpt-6-astra"), "openai-codex", "gpt-5.6-sol"),
    ],
    ids=("missing-runtime", "mismatched-runtime", "mismatched-config-provider", "mismatched-config-model"),
)
def test_none_refuses_missing_or_mismatched_identity_before_transport(
    tmp_path, monkeypatch, runtime, config_provider, config_model
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, provider=config_provider, model=config_model)
    from agent import auxiliary_client as ac
    if runtime is None:
        ac.clear_runtime_main()

    with patch.object(ac, "_create_openai_client") as build:
        with _fallback_patches(ac) as fallback_mocks:
            with pytest.raises(RuntimeError):
                ac.get_text_auxiliary_client("compression", main_runtime=runtime)
    build.assert_not_called()
    assert all(mock.call_count == 0 for mock in fallback_mocks)
