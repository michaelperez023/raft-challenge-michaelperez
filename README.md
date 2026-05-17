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

user query -> parse filters -> fetch orders -> parse orders -> filter orders -> returned orders


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

-> Final JSON Output
