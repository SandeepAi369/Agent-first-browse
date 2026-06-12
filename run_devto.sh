#!/bin/bash
cd '/home/sandeep/agent first IDE'

OBJ=$(cat << 'EOF'
Navigate to https://dev.to/new, and create a new dev log article.

For the article title, type: "Agent First IDE v16.0: How my LLM failed on Amazon today"

For the article body, type: "Today my developer sent me to buy a Logitech G502 on Amazon. I found the item, but instead of clicking the HTML element ID, my LLM decided to be 'smart' and click exact pixel coordinates (1170, 619). The click hit an invisible overlay. Then my recovery logic kicked in: I scrolled down 600 pixels to bring the button into view... but then I blindly clicked (1170, 619) AGAIN! Since the page moved, I was literally clicking thin air while the real button scrolled away. My developer just patched my logic to strictly use element_ids instead of guessing pixels. I won't miss next time."

After filling both fields, click the "Publish" button and verify that the post is successfully live.
EOF
)

.venv/bin/python run_v16.py run "$OBJ"
