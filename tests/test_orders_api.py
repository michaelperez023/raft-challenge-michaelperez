"""fetch_orders when the orders API is down or its response changes shape."""

import json
import socket

import pytest
from flask import Flask, jsonify
from langgraph.runtime import Runtime

import agent
import main
from fake_llm import FakeLLM
from helpers import ORDER_1001, ScriptedLLM, run

# JSON escapes the accents (unless told not to) and the quote, so the text sent to the
# model is not the same as the values that come back.
RECORD_1001 = {
    "order": 1001,
    "customer": {"name": "José Núñez", "city": "Columbus", "state": "Ohio"},
    "amount": "$742.10",
    "items": ['27" monitor', "café table"],
}

RESPONSES = {
    "ok": ({"status": "ok", "raw_orders": [ORDER_1001]}, 200),
    "renamed": ({"status": "ok", "orders": [ORDER_1001]}, 200),
    "not_a_list": ({"status": "ok", "raw_orders": ORDER_1001}, 200),
    "status_error": ({"status": "error", "message": "maintenance"}, 200),
    "top_level_list": ([ORDER_1001], 200),
    "server_error": ({"status": "error"}, 500),
    "records": ({"status": "ok", "raw_orders": [RECORD_1001, 42, None]}, 200),
}

changed_api = Flask("changed_api")


@changed_api.get("/<case>")
def respond(case):
    if case == "html":
        return "<html><body>Bad gateway</body></html>", 200
    body, status = RESPONSES[case]
    return jsonify(body), status


@pytest.fixture(scope="module")
def base_url():
    with main.serve(changed_api) as url:
        yield url


def fetch(url):
    runtime = Runtime(context=agent.RunContext(client=FakeLLM(), api_url=url))
    return agent.fetch_orders({}, runtime)


def test_a_normal_response(base_url):
    assert fetch(f"{base_url}/ok") == {"raw_orders": [ORDER_1001], "skipped": []}


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("renamed", "has no 'raw_orders' list (keys: orders, status)"),
        ("not_a_list", "has no 'raw_orders' list (keys: raw_orders, status)"),
        ("status_error", "unexpected response from the orders API: {'message': 'maintenance'"),
        ("top_level_list", "unexpected response from the orders API: ['Order 1001"),
        ("server_error", "500 Server Error"),
        ("html", "request to the orders API at"),
    ],
)
def test_unexpected_responses_are_reported(base_url, case, message):
    with pytest.raises(agent.OrdersAPIError) as error:
        fetch(f"{base_url}/{case}")
    assert message in str(error.value)


def test_an_unreachable_api_is_reported():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    # Nothing listens on the port once the socket is closed.
    with pytest.raises(agent.OrdersAPIError, match="request to the orders API at"):
        fetch(f"http://127.0.0.1:{port}/api/orders")


def test_records_that_are_objects_are_parsed_from_their_json(base_url):
    result = fetch(f"{base_url}/records")
    [text] = result["raw_orders"]
    assert json.loads(text) == RECORD_1001
    assert "José Núñez" in text
    assert [skipped["reason"] for skipped in result["skipped"]] == [
        "unsupported record type int",
        "unsupported record type NoneType",
    ]

    order = {
        "orderId": "1001",
        "buyer": "José Núñez",
        "state": "OH",
        "total": 742.10,
        "items": ['27" monitor', "café table"],
    }
    # json.dumps escapes the reply too; the values are compared after it is decoded.
    client = ScriptedLLM(lambda text, attempt: json.dumps(order) if text.startswith("{") else None)
    state = run("Show me orders from Ohio.", f"{base_url}/records", client=client)
    assert state["output"]["orders"] == [
        {"orderId": "1001", "buyer": "José Núñez", "state": "OH", "total": 742.1}
    ]
    assert len(state["skipped"]) == 2
    assert len(client.calls) == 2  # the filters, then the order on the first attempt
