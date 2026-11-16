"""How the model is called and how its replies are parsed, validated and checked."""

import json
import re

import openai
import pytest
from pydantic import ValidationError

import agent
from helpers import ORDER_1001, connection_error, replies, status_error

CONTEXT = {"source": ORDER_1001}
GOOD_ORDER = {
    "orderId": "1001",
    "buyer": "John Davis",
    "state": "OH",
    "total": 742.1,
    "items": ["laptop", "hdmi cable"],
}
NO_FILTERS = json.dumps(agent.OrderFilters().model_dump())


def ask_for_filters(client):
    ctx = agent.RunContext(client=client, model="example/model")
    return agent.call_llm(ctx, agent.FILTERS_PROMPT, "Show me everything", agent.OrderFilters)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('Here you go: {"a": 1}. Anything else?', {"a": 1}),
        ('```json\n{"a": {"b": [1, 2]}}\n```', {"a": {"b": [1, 2]}}),
        ('{"a": 1} and also {"b": 2}', {"a": 1}),
        ('{not json} {"a": "}"}', {"a": "}"}),
    ],
)
def test_extract_json(text, expected):
    assert agent.extract_json(text) == expected


@pytest.mark.parametrize("text", ["", "no braces", "[1, 2]", "{'single': 'quotes'}", '{"a": '])
def test_extract_json_without_an_object(text):
    with pytest.raises(ValueError, match="no JSON object"):
        agent.extract_json(text)


def test_call_llm_sends_a_strict_json_schema():
    client = replies(NO_FILTERS)
    ask_for_filters(client)
    [call] = client.calls
    assert call["model"] == "example/model"
    assert call["temperature"] == 0
    response_format = call["response_format"]
    assert response_format["type"] == "json_schema"
    spec = response_format["json_schema"]
    assert spec["name"] == "OrderFilters"
    assert spec["strict"] is True
    schema = spec["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == list(agent.OrderFilters.model_fields)
    assert schema["properties"]["state"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    for keyword in ("title", "description", "default", "pattern"):
        assert f'"{keyword}"' not in json.dumps(schema)


def test_call_llm_retries_with_the_reply_and_the_error():
    client = replies("Sure! The filters are state OH.", NO_FILTERS)
    assert ask_for_filters(client) == agent.OrderFilters()
    first, second = client.calls
    assert [m["role"] for m in first["messages"]] == ["system", "user"]
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "user"]
    assert second["messages"][2]["content"] == "Sure! The filters are state OH."
    assert "no JSON object" in second["messages"][3]["content"]


def test_call_llm_retries_after_a_validation_error():
    invalid = json.dumps({**json.loads(NO_FILTERS), "state": "Ohio State"})
    client = replies(invalid, NO_FILTERS)
    assert ask_for_filters(client) == agent.OrderFilters()
    correction = client.calls[1]["messages"][-1]["content"]
    assert "state: String should match pattern" in correction


def test_call_llm_retries_after_an_empty_reply():
    client = replies("", NO_FILTERS)
    assert ask_for_filters(client) == agent.OrderFilters()
    assert [m["role"] for m in client.calls[1]["messages"]] == ["system", "user", "user"]


def test_call_llm_gives_up_after_the_retry():
    client = replies('{"orderId": [1]}', '{"city": "Columbus"}')
    with pytest.raises(agent.ModelReplyError, match=r"after 2 attempts \(city: Extra inputs"):
        ask_for_filters(client)
    assert len(client.calls) == 2


def test_call_llm_does_not_retry_api_errors():
    client = replies(connection_error())
    with pytest.raises(openai.APIConnectionError):
        ask_for_filters(client)
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"orderId": 1001}, {"orderId": "1001"}),
        ({"orderId": "#1001"}, {"orderId": "1001"}),
        ({"orderId": "Order 1001"}, {"orderId": "1001"}),
        ({"orderId": "Order #1001"}, {"orderId": "1001"}),
        ({"orderId": "ORDER-1001"}, {"orderId": "1001"}),
        ({"orderId": "Order: 1001"}, {"orderId": "1001"}),
        ({"orderId": "Order No. 1001"}, {"orderId": "1001"}),
        ({"orderId": "order number 1001"}, {"orderId": "1001"}),
        ({"orderId": "Order ID: A-7"}, {"orderId": "A-7"}),
        ({"orderId": "A-7"}, {"orderId": "A-7"}),
        ({"orderId": "ORD-1001"}, {"orderId": "ORD-1001"}),
        ({"orderId": "Order"}, {}),
        ({"buyer": "  John   Davis "}, {"buyer": "John Davis"}),
        ({"state": "oh"}, {"state": "OH"}),
        ({"state": " Ohio "}, {"state": "OH"}),
        ({"state": "new york"}, {"state": "NY"}),
        ({"min_total": "$1,299.99"}, {"min_total": 1299.99}),
        ({"orderId": "", "buyer": " ", "state": "", "max_total": ""}, {}),
    ],
)
def test_filters_are_cleaned_up(raw, expected):
    assert agent.OrderFilters.model_validate(raw).model_dump(exclude_defaults=True) == expected


@pytest.mark.parametrize(
    "raw", [{"state": "Ohio State"}, {"state": 39}, {"city": "Columbus"}, {"max_total": "lots"}]
)
def test_invalid_filters_are_rejected(raw):
    with pytest.raises(ValidationError):
        agent.OrderFilters.model_validate(raw)


def test_a_grounded_order_is_accepted():
    order = agent.Order.model_validate({**GOOD_ORDER, "total": "$742.10"}, context=CONTEXT)
    assert order.model_dump() == GOOD_ORDER


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"orderId": "1009"}, "orderId '1009' is not in the text"),
        ({"orderId": "100"}, "orderId '100' is not in the text"),
        ({"buyer": "Jane Davis"}, "buyer 'Jane Davis' is not in the text"),
        ({"state": "TX"}, "state 'TX' is not in the text"),
        ({"total": 742.0}, "total 742.0 is not in the text"),
        ({"items": ["laptop", "mouse"]}, "items ['mouse'] are not in the text"),
    ],
)
def test_invented_values_are_rejected(change, message):
    with pytest.raises(ValidationError, match=re.escape(message)):
        agent.Order.model_validate({**GOOD_ORDER, **change}, context=CONTEXT)


def test_grounding_accepts_other_formats():
    text = "#1001 | DAVIS, JOHN | Columbus, Ohio | 1 Laptop + 1 HDMI cable | USD 742.10"
    order = {**GOOD_ORDER, "items": ["Laptop", "HDMI cable"]}
    assert agent.ungrounded_values(agent.Order.model_validate(order), text) == []


def test_a_labelled_order_id_is_grounded():
    """The model copies "ORDER-1001" from the text, and "1001" is what is checked."""
    context = {"source": ORDER_1001.replace("Order 1001:", "ORDER-1001:")}
    order = agent.Order.model_validate({**GOOD_ORDER, "orderId": "ORDER-1001"}, context=context)
    assert order.orderId == "1001"


def test_grounding_reads_json_text():
    """A record that arrives as a JSON object is parsed from its JSON text."""
    values = {**GOOD_ORDER, "buyer": "José Núñez", "items": ['27" monitor']}
    text = json.dumps(values, ensure_ascii=False)
    assert '27\\" monitor' in text
    order = agent.Order.model_validate(values)
    assert agent.ungrounded_values(order, text) == []
    order = order.model_copy(update={"items": ["27 monitor"]})
    assert agent.ungrounded_values(order, text) == ["items ['27 monitor'] are not in the text"]


def test_without_a_source_only_the_schema_is_checked():
    assert agent.Order.model_validate({**GOOD_ORDER, "buyer": "Anyone"}).buyer == "Anyone"


def order(order_id="1001", buyer="John Davis", state="OH", total=742.1):
    return {"orderId": order_id, "buyer": buyer, "state": state, "total": total, "items": []}


@pytest.mark.parametrize(
    ("filters", "matches"),
    [
        ({}, True),
        ({"orderId": "1001"}, True),
        ({"orderId": "1002"}, False),
        ({"buyer": "davis"}, True),
        ({"buyer": "Davis John"}, True),
        ({"buyer": "Dav"}, False),
        ({"buyer": "John Smith"}, False),
        ({"state": "OH"}, True),
        ({"state": "TX"}, False),
        ({"min_total": 742.1}, False),
        ({"min_total": 742.1, "min_inclusive": True}, True),
        ({"min_total": 742.11, "min_inclusive": True}, False),
        ({"max_total": 742.1}, False),
        ({"max_total": 742.1, "max_inclusive": True}, True),
        ({"min_total": 700, "max_total": 800}, True),
    ],
)
def test_matches_filters(filters, matches):
    assert agent.matches_filters(order(), agent.OrderFilters(**filters)) is matches


def test_sort_key_orders_ids_numerically():
    orders = [order("A-7"), order("1001"), order("999"), order("12")]
    assert [o["orderId"] for o in sorted(orders, key=agent.sort_key)] == [
        "12",
        "999",
        "1001",
        "A-7",
    ]


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        (connection_error(), True),
        (openai.APITimeoutError(request=None), True),
        (status_error(openai.RateLimitError, 429), True),
        (status_error(openai.InternalServerError, 502), True),
        (status_error(openai.APIStatusError, 408), True),
        (status_error(openai.BadRequestError, 400), False),
        (status_error(openai.AuthenticationError, 401), False),
        (status_error(openai.APIStatusError, 402), False),
        (status_error(openai.NotFoundError, 404), False),
    ],
)
def test_is_transient(error, transient):
    assert agent.is_transient(error) is transient


@pytest.mark.parametrize("key", [None, "", "   "])
def test_make_client_needs_a_key(monkeypatch, key):
    if key is not None:
        monkeypatch.setenv("OPENROUTER_API_KEY", key)
    with pytest.raises(agent.ConfigError, match="OPENROUTER_API_KEY is not set"):
        agent.make_client()


def test_make_client_points_at_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    client = agent.make_client()
    assert str(client.base_url) == "https://openrouter.ai/api/v1/"
    assert client.api_key == "test-key"
    assert client.max_retries == 3


def test_default_model_is_the_one_the_brief_requires(monkeypatch):
    assert agent.model_from_env() == "openai/gpt-oss-120b:exacto"
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-oss-120b")
    assert agent.model_from_env() == "openai/gpt-oss-120b"
