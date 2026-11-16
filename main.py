"""Answer a natural-language request about customer orders with JSON, in one command.

    python main.py "Show me all orders from Ohio over 500."

Unless ORDERS_API_URL is set, Raft's dummy API (dummy_customer_api.py) is served on
127.0.0.1 for the length of the run.
"""

import argparse
import json
import logging
import os
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

import openai
from werkzeug.serving import make_server

import agent
from fake_llm import FakeLLM

MAX_QUERY_CHARS = 1000

EXIT_OK = 0
EXIT_ERROR = 1  # the orders API or the model failed
EXIT_USAGE = 2  # bad arguments, empty request or missing API key
EXIT_PARTIAL = 3  # results printed, but some orders could not be parsed


@contextmanager
def serve(app) -> Iterator[str]:
    """Serve a WSGI app on 127.0.0.1 with a free port, and yield its base URL."""
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


@contextmanager
def dummy_api() -> Iterator[str]:
    """Serve Raft's dummy API with the debugger off, and yield its orders URL.

    Running dummy_customer_api.py directly listens on every interface with debug=True.
    """
    from dummy_customer_api import app

    with serve(app) as base_url:
        yield f"{base_url}/api/orders"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer a natural-language request about customer orders with JSON.",
        epilog=(
            "environment: OPENROUTER_API_KEY (needed unless --offline), OPENROUTER_MODEL "
            f"(default {agent.DEFAULT_MODEL}), ORDERS_API_URL (default: serve "
            "dummy_customer_api.py on 127.0.0.1 during the run)"
        ),
    )
    parser.add_argument(
        "query", nargs="*", help="the request; read from standard input if it is not given"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the rule-based fake in fake_llm.py instead of a model (no API key needed)",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="add a linear-regression prediction of each order's total from its number of items",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="log warnings and errors only")
    return parser.parse_args(argv)


def read_query(words: list[str]) -> str:
    if words:
        return " ".join(words)
    if sys.stdin.isatty():
        print("Enter natural language request: ", end="", file=sys.stderr, flush=True)
        return sys.stdin.readline()
    return sys.stdin.read()


def error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    for name in ("httpx", "httpx2", "werkzeug"):
        logging.getLogger(name).setLevel(logging.WARNING)

    query = read_query(args.query).strip()
    if not query:
        error("the request is empty")
        return EXIT_USAGE
    if len(query) > MAX_QUERY_CHARS:
        error(f"the request has {len(query)} characters; the limit is {MAX_QUERY_CHARS}")
        return EXIT_USAGE

    if args.offline:
        client = FakeLLM()
        print(
            "Offline mode: replies come from fake_llm.py, a rule-based stand-in for the "
            "model. Nothing is sent to OpenRouter.",
            file=sys.stderr,
        )
    else:
        try:
            client = agent.make_client()
        except agent.ConfigError as exc:
            error(str(exc))
            return EXIT_USAGE
    model = agent.model_from_env()
    api_url = os.environ.get("ORDERS_API_URL", "").strip()
    orders_api = nullcontext(api_url) if api_url else dummy_api()

    try:
        with orders_api as url:
            state = agent.run_query(query, client, model=model, api_url=url, predict=args.predict)
    except agent.OrdersAPIError as exc:
        error(str(exc))
        return EXIT_ERROR
    except agent.ModelReplyError as exc:
        error(f"the model did not turn the request into valid filters: {exc}")
        return EXIT_ERROR
    except openai.APIError as exc:
        hint = ""
        if isinstance(exc, openai.AuthenticationError):
            hint = " (check OPENROUTER_API_KEY)"
        elif isinstance(exc, openai.NotFoundError):
            hint = " (check the model id, or set OPENROUTER_MODEL)"
        error(f"request to {model} failed: {exc}{hint}")
        return EXIT_ERROR

    print(json.dumps(state["output"], indent=2))
    skipped = state.get("skipped", [])
    if skipped:
        count = "1 order" if len(skipped) == 1 else f"{len(skipped)} orders"
        print(
            f"warning: {count} could not be parsed, so the results may be incomplete",
            file=sys.stderr,
        )
        return EXIT_PARTIAL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
