"""Shared test data and a scriptable version of the fake model."""

from types import SimpleNamespace

import openai

import agent
from fake_llm import FakeLLM

OHIO_OVER_500 = (
    "Show me all orders where the buyer was located in Ohio and total value was over 500."
)

OHIO_OVER_500_OUTPUT = {
    "orders": [
        {"orderId": "1001", "buyer": "John Davis", "state": "OH", "total": 742.1},
        {"orderId": "1003", "buyer": "Mike Turner", "state": "OH", "total": 1299.99},
        {"orderId": "1005", "buyer": "Chris Myers", "state": "OH", "total": 512.0},
    ]
}

ORDER_1001 = (
    "Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, Items: laptop, hdmi cable"
)


class ScriptedLLM(FakeLLM):
    """FakeLLM with replies chosen by a script.

    script(user_content, attempt) returns the reply, an exception to raise, or None to give
    FakeLLM's own reply. `attempt` is 1 for the first request of a call_llm() call and 2
    for its retry.
    """

    def __init__(self, script):
        super().__init__()
        self.script = script

    def reply(self, messages):
        attempt = sum(message["role"] == "user" for message in messages)
        result = self.script(messages[1]["content"], attempt)
        if isinstance(result, Exception):
            raise result
        return super().reply(messages) if result is None else result


def replies(*texts):
    """A ScriptedLLM that gives these replies in turn."""
    remaining = list(texts)
    return ScriptedLLM(lambda user, attempt: remaining.pop(0))


def status_error(cls, status):
    """An openai.APIStatusError subclass as the SDK raises it for an HTTP status."""
    response = SimpleNamespace(status_code=status, headers={}, request=None)
    return cls(f"Error code: {status}", response=response, body=None)


def connection_error():
    return openai.APIConnectionError(request=None)


def run(query, api_url, client=None, **kwargs):
    return agent.run_query(query, client or FakeLLM(), api_url=api_url, **kwargs)


def order_ids(state):
    return [order["orderId"] for order in state["output"]["orders"]]
