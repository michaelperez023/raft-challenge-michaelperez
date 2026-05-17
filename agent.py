from langgraph.graph import START, StateGraph, END

def app(query):
    return -1

def main():
    query = input("Enter natural language request: ")

    if not query:
        raise ValueError("Query cannot be empty")
    
    result = app(query)
    print(str(result))

if __name__ == "__main__":
    main()