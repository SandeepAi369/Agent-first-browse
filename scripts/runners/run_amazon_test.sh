#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1  # repo root
.venv/bin/python src/run_v16.py run "Navigate to Amazon (https://www.amazon.in). Search for 'realme buds wireless 3 neckband earphones' and press enter. DO NOT navigate back to the homepage after searching. Click on the first relevant product to go to its product page. Click the 'Add to Cart' button. Once added, verify it is in the cart and mark the task as complete immediately. Do not click 'Add to Cart' on any other recommended items."
