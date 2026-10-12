#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1  # repo root

OBJ=$(cat << 'EOF'
Navigate to https://github.com/new, and create a new public repository. For the repository name, type "agent-first-ide". You can add a short description if you want. Then click the "Create repository" button and verify that the repository was successfully created.
EOF
)

.venv/bin/python src/run_v16.py run "$OBJ"
