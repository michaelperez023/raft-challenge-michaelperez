"""The whole graph against Raft's dummy API, with the fake model."""

import json
from types import SimpleNamespace

import openai
import pytest
from langgraph.runtime import Runtime

import agent
import dummy_customer_api
from fake_llm import FakeLLM
from helpers import (
    OHIO_OVER_500,
    OHIO_OVER_500_OUTPUT,
    ORDER_1001,
    ScriptedLLM,
    connection_error,
    order_ids,
    run,
    status_error,
)

# The queries listed under "Test Cases Passed" in the submitted README.
SUBMISSION_QUERIES = {
    OHIO_OVER_500: ["1001", "1003", "1005"],
    "Show orders from Washington.": ["1004"],
    "Show me all orders where the buyer was located in Texas.": ["1002"],
    "Show me orders for John Davis.": ["1001"],
    "Find order 1001.": ["1001"],
    "Show me all orders from Florida.": [],
    "Show me all orders over 2000 dollars.": [],
    "Show me all orders from Ohio under 600 dollars.": ["1005"],
    "Get orders from OH above 700.": ["1001", "1003"],
}

# Order 1005 totals exactly $512.00 and order 1002 exactly $156.55.
BOUNDARY_QUERIES = {
    "Show me orders of at least 512 dollars.": ["1001", "1003", "1005"],
    "Show me orders over 512 dollars.": ["1001", "1003"],
    "Show me orders of $1,299.99 or more.": ["1003"],
    "Show me orders of at most 156.55 dollars.": ["1002", "1004"],
    "Show me orders under 156.55 dollars.": ["1004"],
    "Show me orders between 100 and 600 dollars.": ["1002", "1005"],
    "Find order #1003.": ["1003"],
    "Show me orders for Davis.": ["1001"],
}


@pytest.mark.parametrize(("query", "expected"), SUBMISSION_QUERIES.items())
def test_queries_from_the_submission(api_url, query, expected):
    state = run(query, api_url)
    assert order_ids(state) == expected
    assert state["skipped"] == []


@pytest.mark.parametrize(("query", "expected"), BOUNDARY_QUERIES.items())
def test_bounds_and_normalized_filters(api_url, query, expected):
    assert order_ids(run(query, api_url)) == expected


@pytest.mark.parametrize(
    "filters",
    [
        {"state": "oh", "orderId": None},
        {"state": "Ohio", "orderId": None},
        {"state": None, "orderId": 1003},
        {"state": None, "orderId": "Order #1003"},
    ],
)
def test_filter_values_are_normalized_before_matching(api_url, filters):
    """The submitted code matched nothing for these, or stopped on the numeric orderId."""
    query = "Show me the orders."
    reply = json.dumps({"buyer": None, "min_total": None, "max_total": None, **filters})
    client = ScriptedLLM(lambda text, attempt: reply if text == query else None)
    expected = ["1003"] if filters["orderId"] else ["1001", "1003", "1005"]
    assert order_ids(run(query, api_url, client=client)) == expected


def test_output_matches_the_brief_exactly(api_url):
    state = run(OHIO_OVER_500, api_url)
    assert state["output"] == OHIO_OVER_500_OUTPUT
    assert json.dumps(state["output"]) == json.dumps(OHIO_OVER_500_OUTPUT)


def test_results_do_not_depend_on_the_api_order(api_url, monkeypatch):
    # The dummy API shuffles its orders; force the reverse of the usual order.
    reverse = SimpleNamespace(sample=lambda orders, k: list(reversed(orders))[:k])
    monkeypatch.setattr(dummy_customer_api, "random", reverse)
    state = run(OHIO_OVER_500, api_url)
    assert state["raw_orders"][0].startswith("Order 1005")
    assert state["output"] == OHIO_OVER_500_OUTPUT


def test_one_model_call_per_order(api_url):
    client = FakeLLM()
    state = run(OHIO_OVER_500, api_url, client=client, model="example/model")
    assert len(client.calls) == 1 + len(state["raw_orders"]) == 6
    for call, text in zip(client.calls[1:], state["raw_orders"], strict=True):
        assert [message["content"] for message in call["messages"]] == [agent.ORDER_PROMPT, text]
    for call in client.calls:
        assert call["model"] == "example/model"
        assert call["temperature"] == 0
        assert call["response_format"]["type"] == "json_schema"


def test_an_order_with_invented_values_is_retried_then_skipped(api_url):
    def script(text, attempt):
        if text.startswith("Order 1003"):
            return json.dumps(
                {
                    "orderId": "1003",
                    "buyer": "Mike Turner",
                    "state": "OH",
                    "total": 999.0,
                    "items": ["gaming pc", "mouse"],
                }
            )
        return None

    client = ScriptedLLM(script)
    state = run(OHIO_OVER_500, api_url, client=client)

    assert order_ids(state) == ["1001", "1005"]
    [skipped] = state["skipped"]
    assert skipped["text"].startswith("Order 1003")
    assert "total 999.0 is not in the text" in skipped["reason"]
    assert len(client.calls) == 1 + 5 + 1  # one retry for order 1003


def test_an_order_is_kept_when_the_retry_fixes_it(api_url):
    def script(text, attempt):
        if text.startswith("Order 1001") and attempt == 1:
            return '{"orderId": "1001", "buyer": "John Davis", "state": "OH"}'
        return None

    state = run(OHIO_OVER_500, api_url, client=ScriptedLLM(script))
    assert state["output"] == OHIO_OVER_500_OUTPUT
    assert state["skipped"] == []


@pytest.mark.parametrize(
    "error",
    [
        connection_error(),
        status_error(openai.RateLimitError, 429),
        status_error(openai.InternalServerError, 503),
    ],
)
def test_a_transient_model_error_skips_one_order(api_url, error):
    client = ScriptedLLM(lambda text, attempt: error if text.startswith("Order 1003") else None)
    state = run(OHIO_OVER_500, api_url, client=client)
    assert order_ids(state) == ["1001", "1005"]
    [skipped] = state["skipped"]
    assert skipped["reason"].startswith("model request failed")


@pytest.mark.parametrize(
    "error",
    [
        status_error(openai.AuthenticationError, 401),
        status_error(openai.NotFoundError, 404),
        status_error(openai.APIStatusError, 402),
    ],
)
def test_a_fatal_model_error_stops_the_run(api_url, error):
    client = ScriptedLLM(lambda text, attempt: error if text.startswith("Order") else None)
    with pytest.raises(type(error)):
        run(OHIO_OVER_500, api_url, client=client)


def test_a_request_that_cannot_be_parsed_stops_the_run(api_url):
    client = ScriptedLLM(lambda text, attempt: "I can't help with that.")
    with pytest.raises(agent.ModelReplyError, match="no valid reply after 2 attempts"):
        run(OHIO_OVER_500, api_url, client=client)
    assert len(client.calls) == 2


def test_an_order_that_is_too_long_is_skipped_without_a_model_call():
    client = FakeLLM()
    long_order = ORDER_1001 + ", " + "x" * agent.MAX_ORDER_CHARS
    runtime = Runtime(context=agent.RunContext(client=client))
    result = agent.parse_orders({"raw_orders": [long_order, ORDER_1001]}, runtime)
    assert [order["orderId"] for order in result["parsed_orders"]] == ["1001"]
    [skipped] = result["skipped"]
    assert skipped["reason"].endswith(f"the limit is {agent.MAX_ORDER_CHARS}")
    assert len(client.calls) == 1


def test_predict_adds_regression_fields(api_url):
    state = run(OHIO_OVER_500, api_url, predict=True)
    output = state["output"]
    # Items per order: 1001, 1003 and 1005 have 2, 1002 and 1004 have 1. With one feature
    # that takes two values, the fit goes through the mean total of each group.
    mean_of_two = (742.10 + 1299.99 + 512.00) / 3
    mean_of_one = (156.55 + 89.50) / 2
    assert order_ids(state) == ["1001", "1003", "1005"]
    for order in output["orders"]:
        assert order["num_items"] == 2
        assert order["predicted_total"] == pytest.approx(mean_of_two, abs=0.01)
        assert order["prediction_error"] == pytest.approx(order["total"] - mean_of_two, abs=0.01)
    regression = output["linear_regression"]
    assert regression["fitted_on"] == 5
    assert regression["coefficient"] == pytest.approx(mean_of_two - mean_of_one, abs=0.01)
    assert regression["intercept"] == pytest.approx(2 * mean_of_one - mean_of_two, abs=0.01)


def test_predict_with_fewer_than_two_orders():
    order = {"orderId": "1", "buyer": "A B", "state": "OH", "total": 10.0, "items": ["x"]}
    state = {
        "parsed_orders": [order],
        "matches": [order],
        "output": {"orders": [{key: order[key] for key in agent.OUTPUT_FIELDS}]},
    }
    output = agent.predict_totals(state)["output"]
    assert output["linear_regression"] == {
        "message": "Not enough orders to fit linear regression model.",
        "fitted_on": 1,
        "coefficient": None,
        "intercept": None,
    }
    assert output["orders"] == [
        {
            "orderId": "1",
            "buyer": "A B",
            "state": "OH",
            "total": 10.0,
            "num_items": 1,
            "predicted_total": None,
            "prediction_error": None,
        }
    ]


def test_without_predict_the_graph_ends_after_filtering(api_url):
    assert set(run(OHIO_OVER_500, api_url)["output"]) == {"orders"}
