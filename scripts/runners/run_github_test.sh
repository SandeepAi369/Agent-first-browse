#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1  # repo root
.venv/bin/python src/run_v16.py run "Go to https://github.com/. Use the search bar to find the repository 'xel studio'. Click on the correct repository from the search results. Once on the repository page, click the 'Star' button to star the project, and then click the 'Fork' button to fork the repository. Complete the forking process if a confirmation is required."
