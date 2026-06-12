#!/bin/bash
cd '/home/sandeep/agent first IDE'

OBJ=$(cat << 'EOF'
Navigate to https://www.meesho.com, click on the search bar, and search for "Water Bottle 1 Litre". From the search results, click on the first relevant product, scroll down to bring the product options into center view, and click the "Add to Cart" button. Verify that the cart count changes or the success message appears without prompting for a login.
EOF
)

.venv/bin/python run_v16.py run "$OBJ"
