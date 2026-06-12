#!/bin/bash
cd '/home/sandeep/agent first IDE'

OBJ=$(cat << 'EOF'
Navigate to https://www.flipkart.com, search for "Logitech G502 Hero Gaming Mouse". From the search results, click on the first relevant product, scroll down if necessary, and click the "Add to Cart" button. Verify that the product was successfully added to the cart.
EOF
)

.venv/bin/python run_v16.py run "$OBJ"
