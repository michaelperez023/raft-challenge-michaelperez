from langgraph.graph import START, StateGraph, END
from typing_extensions import TypedDict
from typing import List, Optional
import logging
import os
from openai import OpenAI
import json
import requests
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)

API_URL = "http://localhost:5001/api/orders"
MODEL = "openai/gpt-oss-120b:free"

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

class State(TypedDict):
    query: str
    raw_orders: List[str]
    parsed_orders: List[dict]
    filters: dict
    output: dict

class OrderFilters(BaseModel):
    orderId: Optional[str] = None
    buyer: Optional[str] = None
    state: Optional[str] = None
    min_total: Optional[float] = None
    max_total: Optional[float] = None

class Order(BaseModel):
    orderId: str
    buyer: str
    state: str
    total: float

def extract_json(text: str) -> dict:
    text = text.strip()

    start = text.find("{")
    end = text.find("}")

    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in response: {text}")
    
    return json.loads(text[start:end + 1])

def call_llm(system_prompt, user_prompt, retries=1):
    last_error = None

    for attempt in range(retries + 1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if last_error:
            messages.append({
                "role": "user",
                "content": (
                    f"Your previous response was invalid JSON or failed validation. "
                    f"Error: {last_error}. Return ONLY corrected JSON."
                ),
            })

        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=messages
        )

        content = response.choices[0].message.content

        try:
            return extract_json(content)
        except Exception as e:
            last_error = str(e)
    
    raise ValueError(f"LLM failed to produce valid JSON: {last_error}")

def parse_filters(state: State) -> State:
    logging.info("Parsing query into filters")
    prompt = """
            Extract order search filters from the user's request.

            Return ONLY valid JSON with exactly these keys:
            {
            "orderId": string or null,
            "buyer": string or null,
            "state": two-letter US state abbreviation or null,
            "min_total": number or null,
            "max_total": number or null
            }
            """
    
    filters = call_llm(prompt, state["query"], retries=1)
    filters_validated = OrderFilters.model_validate(filters)
    #print(filters_validated.model_dump())

    return {"filters": filters_validated.model_dump()}

def fetch_orders(state: State) -> State:
    logging.info("Fetching orders from API")

    try:
        response = requests.get(API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logging.exception("Failed to fetch orders")
        raise RuntimeError(f"API request failed: {e}")
    
    raw_orders = data.get("raw_orders")
    if raw_orders is None or not isinstance(raw_orders, list):
        raise ValueError("Expected the API response to include a list of raw orders.")
    
    return {"raw_orders": raw_orders}

def parse_orders(state: State) -> State:
    logging.info("Parsing raw orders into structured JSON")

    parsed_orders = []

    prompt = """
            You extract structured order data from messy customer order text.

            Return ONLY valid JSON with exactly these keys:
            {
            "orderId": string,
            "buyer": string,
            "state": two-letter US state abbreviation,
            "total": number
            }
    """

    for raw_order in state["raw_orders"]:
        try:
            order = call_llm(prompt, raw_order, retries=1)
            order_validated = Order.model_validate(order)
            parsed_orders.append(order_validated.model_dump())
        except (ValueError) as e:
            logging.exception(f"Failed to parse order. Error: {e}")
    #print(parsed_orders)

    return {"parsed_orders": parsed_orders}

def filter_orders(state: State) -> State:
    logging.info("Filtering orders")

    filters = state["filters"]
    filtered_orders = []

    order_id = filters.get("orderId")
    buyer = filters.get("buyer")
    state_filter = filters.get("state")
    min_total = filters.get("min_total")
    max_total = filters.get("max_total")

    for order in state["parsed_orders"]:
        if order_id and order.get("orderId") != order_id:
            continue

        if buyer and buyer.lower() not in order.get("buyer", "").lower():
            continue

        if state_filter and order.get("state") != state_filter:
            continue

        total = order.get("total", 0)

        if min_total is not None and total <= min_total:
            continue

        if max_total is not None and total >= max_total:
            continue

        filtered_orders.append(order)

    return {"output": {"orders": filtered_orders}}

def build_graph():
    logging.info("Building graph")

    graph = StateGraph(State)

    graph.add_node("parse_filters", parse_filters)
    graph.add_node("fetch_orders", fetch_orders)
    graph.add_node("parse_orders", parse_orders)
    graph.add_node("filter_orders", filter_orders)

    graph.add_edge(START, "parse_filters")
    graph.add_edge("parse_filters", "fetch_orders")
    graph.add_edge("fetch_orders", "parse_orders")
    graph.add_edge("parse_orders", "filter_orders")
    graph.add_edge("filter_orders", END)

    return graph.compile()

def main():
    query = input("Enter natural language request: ")

    if not query:
        raise ValueError("Query cannot be empty")
    
    app = build_graph()
    result = app.invoke({"query": query})
    
    print(json.dumps(result["output"]))

if __name__ == "__main__":
    main()