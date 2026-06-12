#!/bin/bash
cd '/home/sandeep/agent first IDE'

OBJ=$(cat << 'EOF'
Navigate to https://github.com/new, and create a new public repository. For the repository name, type "agent-first-ide". You can add a short description if you want. Then click the "Create repository" button and verify that the repository was successfully created.
EOF
)

.venv/bin/python run_v16.py run "$OBJ"
