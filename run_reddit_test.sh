#!/bin/bash
cd "/home/sandeep/agent first IDE"
source .venv/bin/activate
python3 advanced_agent.py run "$(cat reddit_objective.txt)" > reddit_v13_test.log 2>&1
