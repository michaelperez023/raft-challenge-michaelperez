"""The command line: python main.py [--offline] [--predict] [-q] "request"."""

import io
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import openai
import pytest

import agent
import main
from fake_llm import FakeLLM
from helpers import OHIO_OVER_500, OHIO_OVER_500_OUTPUT, ScriptedLLM, status_error

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def clients(monkeypatch):
    """Record the fake clients that main() creates."""
    created = []

    def make_fake():
        created.append(FakeLLM())
        return created[-1]

    monkeypatch.setattr(main, "FakeLLM", make_fake)
    return created


def test_offline_run_prints_the_briefs_json(capsys, clients):
    assert main.main(["--offline", OHIO_OVER_500]) == main.EXIT_OK
    out, err = capsys.readouterr()
    assert json.loads(out) == OHIO_OVER_500_OUTPUT
    assert "Offline mode" in err
    [client] = clients
    assert {call["model"] for call in client.calls} == {"openai/gpt-oss-120b:exacto"}


def test_the_request_may_be_unquoted(capsys):
    assert main.main(["--offline", "Find", "order", "1001"]) == main.EXIT_OK
    assert [o["orderId"] for o in json.loads(capsys.readouterr().out)["orders"]] == ["1001"]


def test_the_request_can_come_from_stdin(capsys, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("Find order 1003.\n"))
    assert main.main(["--offline"]) == main.EXIT_OK
    assert [o["orderId"] for o in json.loads(capsys.readouterr().out)["orders"]] == ["1003"]


def test_predict(capsys):
    assert main.main(["--offline", "--predict", OHIO_OVER_500]) == main.EXIT_OK
    output = json.loads(capsys.readouterr().out)
    assert set(output) == {"orders", "linear_regression"}
    assert set(output["orders"][0]) == {
        "orderId",
        "buyer",
        "state",
        "total",
        "num_items",
        "predicted_total",
        "prediction_error",
    }


def test_openrouter_model_overrides_the_default(monkeypatch, clients):
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-oss-120b")
    assert main.main(["--offline", "-q", OHIO_OVER_500]) == main.EXIT_OK
    assert {call["model"] for call in clients[0].calls} == {"openai/gpt-oss-120b"}


def test_a_missing_key_is_a_clear_error(capsys):
    assert main.main([OHIO_OVER_500]) == main.EXIT_USAGE
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("error: OPENROUTER_API_KEY is not set.")
    assert "--offline" in err


@pytest.mark.parametrize("args", [[], [" "], ["x" * (main.MAX_QUERY_CHARS + 1)]])
def test_empty_or_long_requests_are_rejected(capsys, monkeypatch, args):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert main.main(["--offline", *args]) == main.EXIT_USAGE
    assert capsys.readouterr().err.startswith("error: the request")


def test_an_unreachable_orders_api(capsys, monkeypatch):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    monkeypatch.setenv("ORDERS_API_URL", f"http://127.0.0.1:{port}/api/orders")
    assert main.main(["--offline", OHIO_OVER_500]) == main.EXIT_ERROR
    out, err = capsys.readouterr()
    assert out == ""
    assert f"error: request to the orders API at http://127.0.0.1:{port}/api/orders" in err


def test_orders_api_url_is_used(capsys, monkeypatch, api_url):
    monkeypatch.setenv("ORDERS_API_URL", api_url)
    assert main.main(["--offline", OHIO_OVER_500]) == main.EXIT_OK
    assert json.loads(capsys.readouterr().out) == OHIO_OVER_500_OUTPUT


def test_skipped_orders_give_exit_status_3(capsys, monkeypatch):
    def broken_1003(text, attempt):
        return "{}" if text.startswith("Order 1003") else None

    monkeypatch.setattr(main, "FakeLLM", lambda: ScriptedLLM(broken_1003))
    assert main.main(["--offline", OHIO_OVER_500]) == main.EXIT_PARTIAL
    out, err = capsys.readouterr()
    assert [o["orderId"] for o in json.loads(out)["orders"]] == ["1001", "1005"]
    assert "warning: 1 order could not be parsed" in err


@pytest.mark.parametrize(
    ("error", "hint"),
    [
        (status_error(openai.AuthenticationError, 401), "(check OPENROUTER_API_KEY)"),
        (status_error(openai.NotFoundError, 404), "(check the model id, or set OPENROUTER_MODEL)"),
        (status_error(openai.APIStatusError, 402), "Error code: 402"),
    ],
)
def test_model_errors_are_reported(capsys, monkeypatch, error, hint):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(agent, "make_client", lambda: ScriptedLLM(lambda text, attempt: error))
    assert main.main([OHIO_OVER_500]) == main.EXIT_ERROR
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("error: request to openai/gpt-oss-120b:exacto failed")
    assert hint in err


def test_a_request_the_model_cannot_parse(capsys, monkeypatch):
    monkeypatch.setattr(main, "FakeLLM", lambda: ScriptedLLM(lambda text, attempt: "no idea"))
    assert main.main(["--offline", OHIO_OVER_500]) == main.EXIT_ERROR
    assert "did not turn the request into valid filters" in capsys.readouterr().err


def test_one_command_in_a_new_process():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENROUTER_", "ORDERS_"))}
    result = subprocess.run(
        [sys.executable, "main.py", "--offline", OHIO_OVER_500],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == OHIO_OVER_500_OUTPUT
    assert "INFO: 3 of 5 orders match" in result.stderr
