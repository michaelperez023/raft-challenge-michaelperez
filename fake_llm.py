"""A deterministic, rule-based stand-in for the model, used by --offline and the tests.

FakeLLM has the interface of the OpenAI client (client.chat.completions.create) and
answers the two prompts in agent.py the way a well-behaved model would, for simple
requests and for the dummy API's order format. It is not a language model: it knows a
few phrasings of states, amounts, order numbers and buyer names, and nothing else.
"""

import json
import re
from types import SimpleNamespace

from agent import FILTERS_PROMPT, ORDER_PROMPT
from us_states import STATE_CODES, STATE_NAMES, to_code

_AMOUNT = r"\$?\s*(\d[\d,]*(?:\.\d+)?)"

# (pattern, bounds it sets, inclusive). Checked in this order, and each match is removed
# from the request before the next pattern, so "no more than 5" is not also "more than 5".
_BOUNDS = [
    (rf"\bbetween\s+{_AMOUNT}\s+and\s+{_AMOUNT}", ("min", "max"), True),
    (rf"\b(?:at least|no less than|minimum of)\s+{_AMOUNT}", ("min",), True),
    (rf"{_AMOUNT}\s*(?:dollars\s*)?or more\b", ("min",), True),
    (rf"\b(?:at most|no more than|up to|maximum of)\s+{_AMOUNT}", ("max",), True),
    (rf"{_AMOUNT}\s*(?:dollars\s*)?or less\b", ("max",), True),
    (rf"\b(?:over|above|more than|greater than|exceeding)\s+{_AMOUNT}", ("min",), False),
    (rf"\b(?:under|below|less than)\s+{_AMOUNT}", ("max",), False),
]
_ORDER_ID = r"\border\s*(?:#|number\s*|no\.\s*)?\s*(\d+)\b|#(\d+)\b"
_STATE_NAME = "|".join(sorted(map(re.escape, STATE_CODES), key=len, reverse=True))
# A capitalized name after one of these words: "orders for John Davis".
_BUYER = r"\b(?i:for|by|from|buyer|customer|named)\s+([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*)"


def _amount(text: str) -> float:
    return float(text.replace(",", ""))


def parse_request(request: str) -> dict:
    """Filters for a request, in the JSON shape that FILTERS_PROMPT asks for."""
    filters = {
        "orderId": None,
        "buyer": None,
        "state": None,
        "min_total": None,
        "min_inclusive": False,
        "max_total": None,
        "max_inclusive": False,
    }
    text = request
    for pattern, bounds, inclusive in _BOUNDS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            for bound, value in zip(bounds, match.groups(), strict=True):
                filters[f"{bound}_total"] = _amount(value)
                filters[f"{bound}_inclusive"] = inclusive
            text = text[: match.start()] + " " + text[match.end() :]

    match = re.search(_ORDER_ID, text, re.IGNORECASE)
    if match:
        filters["orderId"] = match.group(1) or match.group(2)
        text = text[: match.start()] + " " + text[match.end() :]

    by_name = re.search(rf"(?<!\w)(?:{_STATE_NAME})(?!\w)", text, re.IGNORECASE)
    by_code = next(
        (m for m in re.finditer(r"\b[A-Z]{2}\b", text) if m.group() in STATE_NAMES), None
    )
    first = min((m for m in (by_name, by_code) if m), key=lambda m: m.start(), default=None)
    if first:
        filters["state"] = to_code(first.group())

    for match in re.finditer(_BUYER, text):
        name = match.group(1)
        if to_code(name) not in STATE_NAMES:  # "from Ohio" or "from OH" is a place
            filters["buyer"] = name
            break
    return filters


_ORDER_FIELDS = {
    "orderId": r"\border\s*#?\s*([A-Za-z0-9-]+)",
    "buyer": r"\bbuyer\s*[=:]\s*([^,\n]+)",
    "state": r"\blocation\s*[=:]\s*[^,\n]+,\s*([A-Za-z][A-Za-z ]*[A-Za-z])",
    "total": rf"\btotal\s*[=:]\s*{_AMOUNT}",
    "items": r"\bitems\s*[=:]\s*(.+?)\s*$",
}


def parse_order_text(text: str) -> dict:
    """The fields of one order in the dummy API's format, as ORDER_PROMPT asks for them.

    Fields that are not found are left out, so the reply fails validation.
    """
    order = {}
    for field, pattern in _ORDER_FIELDS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            order[field] = match.group(1).strip()
    if "state" in order:
        order["state"] = to_code(order["state"])
    if "total" in order:
        order["total"] = _amount(order["total"])
    if "items" in order:
        order["items"] = [item.strip() for item in order["items"].split(",") if item.strip()]
    return order


class FakeLLM:
    """Offline replacement for openai.OpenAI. Every request is recorded in `calls`."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, *, model: str, messages: list[dict], **params) -> SimpleNamespace:
        self.calls.append({"model": model, "messages": messages, **params})
        message = SimpleNamespace(content=self.reply(messages))
        choice = SimpleNamespace(index=0, finish_reason="stop", message=message)
        return SimpleNamespace(model=model, choices=[choice])

    def reply(self, messages: list[dict]) -> str:
        """The reply to a conversation. Retries get the same reply as the first attempt."""
        system, user = messages[0]["content"], messages[1]["content"]
        if system == FILTERS_PROMPT:
            return json.dumps(parse_request(user))
        if system == ORDER_PROMPT:
            return json.dumps(parse_order_text(user))
        return "This fake only answers the two prompts in agent.py."
