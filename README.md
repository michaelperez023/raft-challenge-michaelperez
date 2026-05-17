# raft-challenge-michaelperez
My code for the Raft AI Engineer Coding Challenge

# run instructions:
On Mac:

Open terminal 1:

git clone https://github.com/michaelperez023/raft-challenge-michaelperez.git

cd raft_challenge

python -m venv ./.venv

source ./.venv/bin/activate

pip install -r requirements.txt

python dummy_customer_api.py

Open terminal 2:

cd raft_challenge

python -m venv ./.venv

source ./.venv/bin/activate

pip install -r requirements.txt

python agent.py


# Architecture Diagram

user query -> parse filters -> fetch orders -> parse orders -> filter orders -> predict totals (optional linear regression) -> returned orders


____________________________
User Query:

Example: Show me all orders where the buyer was located in Ohio and total value was over 500. ->

->

parse_filters: 

Uses the LLM to convert the natural language request into structured filters

Example: state = "OH", min_total = 500

->

fetch_orders:

Calls the customer orders API and retrieves raw order strings.

Example:
"Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, Items: laptop, hdmi cable", \
"Order 1002: Buyer=Sarah Liu, Location=Austin, TX, Total=$156.55, Items: headphones", \
"Order 1003: Buyer=Mike Turner, Location=Cleveland, OH, Total=$1299.99, Items: gaming pc, mouse", \
"Order 1004: Buyer=Rachel Kim, Location=Seattle, WA, Total=$89.50, Items: coffee maker", \
"Order 1005: Buyer=Chris Myers, Location=Cincinnati, OH, Total=$512.00, Items: monitor, desk lamp"

->

parse_orders:

Uses the LLM to convert each raw order string into structured JSON \
Example: { "orderId": "1001", "buyer": "John Davis", "state": "OH", "total": 742.10 }, ...

->

filter_orders:

Applies the extracted filters to the parsed orders

-> 

predict_totals:\
Fit a linear regression model over all orders, to predict an order's total from the number of items.\
Uses SciKit-learn LinearRegression\

->

Final JSON Output
{"orders":\
[\
    {"orderId": "1003", "buyer": "Mike Turner", "state": "OH", "total": 1299.99},\
    {"orderId": "1005", "buyer": "Chris Myers", "state": "OH", "total": 512.0},\
    {"orderId": "1001", "buyer": "John Davis", "state": "OH", "total": 742.1}\
]}


# Edge Cases and how they are handled
context window overflow
In the case of querying a very large database of 100s of orders, the LLM's context window does not overflow because each order is sent to the LLM individually, keeping each LLM call small.

model hallucination
1) prompting the model to only return valid JSON with exact keys.
2) I validate model outputs with pydantic
3) call_llm() has a retry path to ask for corrections if errors occur
4) using temperature = 0 when calling the LLM to make output deterministic

unpredictable API schema changes
1) I validate the raw_orders field immediately after the API call for fetching orders. If the field is missing or the wrong type, an error is raised. 


# Test Cases Passed

Show me all orders where the buyer was located in Ohio and total value was over 500.
Show orders from Washington.
Show me all orders where the buyer was located in Texas.
Show me orders for John Davis.
Find order 1001.
Show me all orders from Florida.
Show me all orders over 2000 dollars.
Show me all orders from Ohio under 600 dollars.
Get orders from OH above 700.