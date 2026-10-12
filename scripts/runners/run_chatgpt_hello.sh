#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1  # repo root
.venv/bin/python src/run_v16.py run "Navigate to ChatGPT (https://chatgpt.com). Find the chat input field. Type 'hello' and send the message. Wait for the AI to fully finish its response. Once the response is fully generated, mark the task as complete."
