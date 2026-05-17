from langgraph.graph import START, StateGraph, END
from typing_extensions import TypedDict
from typing import List
import logging
import os
from openai import OpenAI
import json

logging.basicConfig(level=logging.INFO)

API_URL = "https://localhost:5001/api/orders"
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

            Return ONLY valid JSOn with exactly these keys:
            {
            "orderId": string or null,
            "buyer": string or null,
            "state": two-letter US state abbreviation or null,
            "min_total": number or null,
            "max_total": number or null
            }
            """
    
    filters = call_llm(prompt, state["query"], retries=1)

    breakpoint()
    return {}

def fetch_orders(state: State) -> State:
    return {}

def parse_orders(state: State) -> State:
    return {}

def filter_orders(state: State) -> State:
    return {}

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
    #query = input("Enter natural language request: ")
    query = "Show me all orders where the buyer was located in Ohio and total value was over 500."

    if not query:
        raise ValueError("Query cannot be empty")
    
    app = build_graph()
    result = app.invoke({"query": query})
    
    print(str(result["output"]))

if __name__ == "__main__":
    main()