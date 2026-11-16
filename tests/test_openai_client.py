"""The real OpenAI SDK client, pointed at a local stand-in for OpenRouter's API.

Nothing leaves 127.0.0.1: the stand-in answers /chat/completions with FakeLLM's replies,
or with the errors a test asks for.
"""

import json
from types import SimpleNamespace

import pytest
from flask import Flask, jsonify, request

import agent
import dummy_customer_api
import main
from fake_llm import FakeLLM
from helpers import OHIO_OVER_500, OHIO_OVER_500_OUTPUT

stand_in = Flask("openrouter_stand_in")
received = []  # the JSON body and Authorization header of each request
scripted = []  # (status, body) to answer with before the normal replies


@stand_in.post("/api/v1/chat/completions")
def chat_completions():
    body = request.get_json()
    received.append({"body": body, "authorization": request.headers.get("Authorization")})
    if scripted:
        status, reply = scripted.pop(0)
        if status != 200:
            return jsonify(reply), status, {"retry-after-ms": "10"}
        content = reply
    else:
        content = FakeLLM().reply(body["messages"])
    message = {"role": "assistant", "content": content}
    return jsonify(
        {
            "id": f"stand-in-{len(received)}",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        }
    )


@pytest.fixture
def openrouter(monkeypatch):
    received.clear()
    scripted.clear()
    with main.serve(stand_in) as base_url:
        monkeypatch.setattr(agent, "OPENROUTER_BASE_URL", f"{base_url}/api/v1")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        yield


def test_a_run_through_the_sdk(openrouter, capsys):
    assert main.main(["-q", OHIO_OVER_500]) == main.EXIT_OK
    assert json.loads(capsys.readouterr().out) == OHIO_OVER_500_OUTPUT
    assert len(received) == 6
    first = received[0]
    assert first["authorization"] == "Bearer test-key"
    assert first["body"]["model"] == "openai/gpt-oss-120b:exacto"
    assert first["body"]["temperature"] == 0
    assert first["body"]["seed"] == 0
    assert first["body"]["response_format"] == agent.response_format(agent.OrderFilters)


def test_the_retry_reaches_the_api_with_the_rejected_reply(openrouter, capsys):
    scripted.append((200, "Sure, here are the filters."))
    assert main.main(["-q", OHIO_OVER_500]) == main.EXIT_OK
    assert json.loads(capsys.readouterr().out) == OHIO_OVER_500_OUTPUT
    messages = received[1]["body"]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == "Sure, here are the filters."
    assert messages[3]["content"].startswith("Your reply was rejected: no JSON object")


@pytest.mark.parametrize("status", [429, 500, 503])
def test_the_sdk_retries_rate_limits_and_server_errors(openrouter, capsys, status):
    scripted.append((status, {"error": {"message": "try again", "code": status}}))
    assert main.main(["-q", OHIO_OVER_500]) == main.EXIT_OK
    assert json.loads(capsys.readouterr().out) == OHIO_OVER_500_OUTPUT
    assert len(received) == 7


def test_an_invalid_key_stops_the_run(openrouter, capsys):
    scripted.append((401, {"error": {"message": "No auth credentials found", "code": 401}}))
    assert main.main(["-q", OHIO_OVER_500]) == main.EXIT_ERROR
    err = capsys.readouterr().err
    assert "Error code: 401" in err
    assert "(check OPENROUTER_API_KEY)" in err
    assert len(received) == 1


def test_a_model_without_providers_stops_the_run(openrouter, capsys, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
    scripted.append((404, {"error": {"message": "No endpoints found", "code": 404}}))
    assert main.main(["-q", OHIO_OVER_500]) == main.EXIT_ERROR
    err = capsys.readouterr().err
    assert err.startswith("error: request to openai/gpt-oss-120b:free failed: Error code: 404")
    assert "(check the model id, or set OPENROUTER_MODEL)" in err
    assert received[0]["body"]["model"] == "openai/gpt-oss-120b:free"


def test_a_server_error_on_one_order_skips_it(openrouter, capsys, monkeypatch):
    # The API returns its orders unshuffled, so order 1001 is parsed first. Its request
    # and the SDK's 3 retries all fail.
    unshuffled = SimpleNamespace(sample=lambda orders, k: list(orders)[:k])
    monkeypatch.setattr(dummy_customer_api, "random", unshuffled)
    scripted.append((200, json.dumps(agent.OrderFilters(state="OH").model_dump())))
    scripted.extend([(502, {"error": {"message": "bad gateway", "code": 502}})] * 4)
    assert main.main(["-q", OHIO_OVER_500]) == main.EXIT_PARTIAL
    out, err = capsys.readouterr()
    assert [order["orderId"] for order in json.loads(out)["orders"]] == ["1003", "1005"]
    assert "warning: 1 order could not be parsed" in err
    assert len(received) == 1 + 4 + 4
