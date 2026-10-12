#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1  # repo root
.venv/bin/python src/run_v16.py run "Navigate to Groq Console (https://console.groq.com/keys). Check if you are logged in. If you see a login screen, click 'Continue with Google' or 'Sign In' and select the existing Gmail account. Once on the API Keys page, click 'Create API Key'. If it asks for a name, enter 'Sandeep'. Finally, click Submit/Create, copy the generated API key, and mark the task as complete."
