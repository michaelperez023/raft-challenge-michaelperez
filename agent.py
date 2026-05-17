from langgraph.graph import START, StateGraph, END
from typing_extensions import TypedDict
from typing import List
import logging

logging.basicConfig(level=logging.INFO)

class State(TypedDict):
    query: str
    raw_orders: List[str]
    parsed_orders: List[dict]
    filters: dict
    output: dict

def parse_filters(state: State) -> State:
    logging.info("Parsing query into filters")


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