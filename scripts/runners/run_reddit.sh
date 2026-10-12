#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1  # repo root
mkdir -p logs
source .venv/bin/activate
python3 src/advanced_agent.py run "$(cat examples/objectives/reddit_objective.txt)" > logs/reddit_ucrf.log 2>&1
