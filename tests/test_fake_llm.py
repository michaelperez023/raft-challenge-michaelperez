"""The rule-based fake that stands in for the model offline."""

import json

import pytest

import agent
from fake_llm import FakeLLM, parse_order_text, parse_request
from helpers import ORDER_1001


@pytest.mark.parametrize(
    ("request_text", "expected"),
    [
        ("Show me everything.", {}),
        ("Orders from Ohio over 500", {"state": "OH", "min_total": 500.0}),
        ("orders from OH above $700", {"state": "OH", "min_total": 700.0}),
        ("orders from West Virginia", {"state": "WV"}),
        ("orders from new york", {"state": "NY"}),
        ("orders of at least 512", {"min_total": 512.0, "min_inclusive": True}),
        ("orders of 1,000 dollars or more", {"min_total": 1000.0, "min_inclusive": True}),
        ("orders under 600", {"max_total": 600.0}),
        ("orders of no more than 600", {"max_total": 600.0, "max_inclusive": True}),
        ("orders of 600 or less", {"max_total": 600.0, "max_inclusive": True}),
        (
            "orders between 100 and 600",
            {"min_total": 100.0, "min_inclusive": True, "max_total": 600.0, "max_inclusive": True},
        ),
        ("Find order 1001.", {"orderId": "1001"}),
        ("Find order number 1001.", {"orderId": "1001"}),
        ("Find #1001", {"orderId": "1001"}),
        ("Show me orders for John Davis.", {"buyer": "John Davis"}),
        ("For Chris Myers, show the orders", {"buyer": "Chris Myers"}),
        ("orders by Rachel Kim from WA", {"buyer": "Rachel Kim", "state": "WA"}),
        ("orders for New York", {"state": "NY"}),
    ],
)
def test_parse_request(request_text, expected):
    filters = agent.OrderFilters.model_validate(parse_request(request_text))
    assert filters.model_dump(exclude_defaults=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            ORDER_1001,
            {
                "orderId": "1001",
                "buyer": "John Davis",
                "state": "OH",
                "total": 742.1,
                "items": ["laptop", "hdmi cable"],
            },
        ),
        (
            "Order 77: Buyer=Ana Ruiz, Location=Miami, Florida, Total=$1,050.00, Items: tv",
            {"orderId": "77", "buyer": "Ana Ruiz", "state": "FL", "total": 1050.0, "items": ["tv"]},
        ),
    ],
)
def test_parse_order_text(text, expected):
    assert parse_order_text(text) == expected
    agent.Order.model_validate(parse_order_text(text), context={"source": text})


def test_unknown_order_formats_fail_validation():
    reply = parse_order_text("1001 | John Davis | Columbus, OH | $742.10")
    assert "buyer" not in reply
    with pytest.raises(ValueError, match="Field required"):
        agent.Order.model_validate(reply)


def test_fake_llm_has_the_client_interface():
    client = FakeLLM()
    messages = [
        {"role": "system", "content": agent.ORDER_PROMPT},
        {"role": "user", "content": ORDER_1001},
    ]
    response = client.chat.completions.create(model="m", messages=messages, temperature=0)
    assert json.loads(agent.reply_text(response))["orderId"] == "1001"
    assert client.calls == [{"model": "m", "messages": messages, "temperature": 0}]


def test_fake_llm_only_answers_the_two_prompts():
    client = FakeLLM()
    messages = [{"role": "system", "content": "Hi"}, {"role": "user", "content": "Hello"}]
    response = client.chat.completions.create(model="m", messages=messages)
    with pytest.raises(ValueError, match="no JSON object"):
        agent.extract_json(agent.reply_text(response))
