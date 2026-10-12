#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1  # repo root
.venv/bin/python src/run_v16.py run "Navigate to amazon.in, search for 'Logitech G502 Hero Gaming Mouse', click on the first exact matching product result, scroll down to ensure visibility, and click 'Add to Cart'. Verify success using the cart flyout UI."
