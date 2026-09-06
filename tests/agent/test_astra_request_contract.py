"""Astra restrictions apply to the serialized request, not just SDK kwargs."""

import copy
import json
from types import SimpleNamespace

import httpx
import openai
import pytest
from openai._models import FinalRequestOptions

from agent.auxiliary_client import _CodexCompletionsAdapter
from agent.codex_runtime import _bypass_sdk_request_transform
from agent.transports.codex import ResponsesApiTransport


ROUTES = {
    "direct": {"provider": "openai", "base_url": "https://api.openai.com/v1"},
    "oauth": {
        "provider": "openai-codex",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "is_codex_backend": True,
    },
}
MESSAGES = [{"role": "user", "content": "Offline request fixture"}]


def _build(path, route, config, model="gpt-6-astra"):
    if path == "main":
        return ResponsesApiTransport().build_kwargs(
            model=model, messages=MESSAGES, reasoning_config=config, **ROUTES[route]
        )
    adapter = _CodexCompletionsAdapter(
        SimpleNamespace(base_url=ROUTES[route]["base_url"]), model
    )
    return adapter._build_responses_kwargs(
        {"messages": MESSAGES, "extra_body": {"reasoning": config}}
    )[0]


def _send(kwargs, base_url, *, raw=False):
    captured = []

    def handle(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text="data: [DONE]\n\n"
        )

    with openai.OpenAI(
        api_key="offline-fixture", base_url=base_url,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as client:
        if raw:
            # Isolate body-precedence failures from SDK keyword compatibility.
            body = dict(kwargs)
            extra = body.pop("extra_body", None)
            headers = body.pop("extra_headers", {})
            request = client._build_request(FinalRequestOptions.construct(
                method="post", url="/responses", json_data=body, extra_json=extra, headers=headers,
            ))
            client._client.send(request)
        else:
            stream = client.responses.create(**_bypass_sdk_request_transform(
                {**kwargs, "stream": True}
            ))
            list(stream)
            stream.close()
    assert len(captured) == 1
    return captured[0]


@pytest.mark.parametrize("path", ["main", "aux"])
@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6-astra-900k"])
def test_direct_request_reaches_sdk_transport_with_cache_contract(path, model):
    kwargs = ResponsesApiTransport().preflight_kwargs(
        _build(path, "direct", {"effort": "max"}, model)
    )
    body = _send(kwargs, ROUTES["direct"]["base_url"])
    assert body["model"] == "gpt-6-astra"
    assert body["reasoning"]["effort"] == "max"
    assert body["prompt_cache_options"] == {"ttl": "30m"}
    assert "prompt_cache_retention" not in body


@pytest.mark.parametrize("route", ["direct", "oauth"])
@pytest.mark.parametrize("effort,expected", [("none", "low"), ("minimal", "low"), ("max", "max")])
@pytest.mark.parametrize("nested", [False, True])
def test_wire_overrides_cannot_bypass_astra_contract(route, effort, expected, nested):
    overrides = {
        "reasoning": {"effort": effort, "summary": "detailed"},
        "temperature": 0.7, "top_p": 0.8, "logprobs": True, "top_logprobs": 4,
        "include": ["reasoning.encrypted_content", "message.output_text.logprobs"],
        "prompt_cache_retention": "24h", "prompt_cache_options": {"ttl": "1h"},
        "service_tier": "priority",
    }
    if nested:
        overrides = {**overrides, "reasoning": {"effort": "high"}, "extra_body": copy.deepcopy(overrides)}
    original = copy.deepcopy(overrides)
    transport = ResponsesApiTransport()
    kwargs = transport.build_kwargs(
        model="gpt-6-astra", messages=MESSAGES, reasoning_config={"effort": "high"},
        request_overrides=overrides, **ROUTES[route],
    )
    body = _send(kwargs, ROUTES[route]["base_url"], raw=True)
    assert body["reasoning"] == {"effort": expected, "summary": "detailed"}
    assert body["service_tier"] == "priority"
    assert body["include"] == ["reasoning.encrypted_content"]
    assert not {"temperature", "top_p", "logprobs", "top_logprobs", "prompt_cache_retention"} & body.keys()
    if route == "direct":
        assert body["prompt_cache_options"] == {"ttl": "30m"}
    else:
        assert "prompt_cache_options" not in body
    assert overrides == original
    assert _send(transport.preflight_kwargs(kwargs), ROUTES[route]["base_url"]) == {**body, "stream": True}


@pytest.mark.parametrize("path", ["main", "aux"])
@pytest.mark.parametrize("config,expected", [
    ({"enabled": False}, "low"),
    ({"enabled": False, "effort": "max"}, "low"),
    ({"effort": "none"}, "low"),
    ({"effort": "minimal"}, "low"),
    ({"effort": "max"}, "max"),
])
def test_oauth_main_and_aux_share_effort_contract_without_direct_cache(path, config, expected):
    kwargs = ResponsesApiTransport().preflight_kwargs(_build(path, "oauth", config))
    body = _send(kwargs, ROUTES["oauth"]["base_url"])
    assert body.get("reasoning", {}).get("effort") == expected
    assert "prompt_cache_options" not in body
    assert "prompt_cache_retention" not in body


@pytest.mark.parametrize("base_url", [
    "https://responses.example.com/v1",
    "https://evil.api.openai.com/v1",
    "https://chatgpt.com.example.com/backend-api/codex",
    "https://chatgpt.com/other-api",
])
def test_foreign_routes_keep_their_wire_override_semantics(base_url):
    extra = {
        "reasoning": {"effort": "none"}, "temperature": 0.7,
        "include": ["message.output_text.logprobs"],
        "prompt_cache_options": {"ttl": "1h"}, "vendor_option": "preserved",
    }
    kwargs = ResponsesApiTransport().build_kwargs(
        model="gpt-6-astra", messages=MESSAGES, base_url=base_url,
        request_overrides={"extra_body": extra},
    )
    body = _send(kwargs, base_url)
    assert all(body[key] == value for key, value in extra.items())


INVALID_EXTRA_BODIES = [False, 0, "", [], [["vendor_option", "value"]], "bad", 1, True]


@pytest.mark.parametrize("entry", ["main", "shared_sanitizer"])
@pytest.mark.parametrize("route", ["direct", "oauth"])
@pytest.mark.parametrize("extra_body", INVALID_EXTRA_BODIES)
def test_invalid_outer_extra_body_preserves_object_validation(entry, route, extra_body):
    from agent.transports.codex import _sanitize_astra_request_kwargs

    kwargs = {"extra_body": extra_body, "reasoning": {"effort": "max"}}
    original = copy.deepcopy(kwargs)
    with pytest.raises(ValueError) as exc:
        if entry == "main":
            ResponsesApiTransport().build_kwargs(
                model="gpt-6-astra", messages=MESSAGES,
                request_overrides=kwargs, **ROUTES[route],
            )
        else:
            _sanitize_astra_request_kwargs(kwargs, "gpt-6-astra", ROUTES[route]["base_url"])
    assert str(exc.value) == "Codex Responses request 'extra_body' must be an object."
    assert kwargs == original


@pytest.mark.parametrize("path", ["main", "aux"])
@pytest.mark.parametrize("route", ["direct", "oauth"])
@pytest.mark.parametrize("extra_body", [None, {}])
def test_empty_outer_extra_body_keeps_official_wire_contract(path, route, extra_body):
    from agent.transports.codex import _sanitize_astra_request_kwargs

    kwargs = _build(path, route, {"effort": "max"})
    kwargs["extra_body"] = extra_body
    original = copy.deepcopy(extra_body)
    _sanitize_astra_request_kwargs(kwargs, "gpt-6-astra", ROUTES[route]["base_url"])
    body = _send(ResponsesApiTransport().preflight_kwargs(kwargs), ROUTES[route]["base_url"])
    assert body["reasoning"]["effort"] == "max"
    if route == "direct":
        assert body["prompt_cache_options"] == {"ttl": "30m"}
    else:
        assert "prompt_cache_options" not in body
    assert extra_body == original
