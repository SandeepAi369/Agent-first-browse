#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1  # repo root
.venv/bin/python src/run_v16.py run "Go to https://github.com/SandeepAi369/xel-studio. Your only goal is to Star this repository. The Star button may be hidden from the accessibility tree (shadow DOM). Use vision coordinate fallback if the Star button is not in the element map. Verify the repository is starred before completing."
