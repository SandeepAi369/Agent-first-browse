#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1  # repo root
.venv/bin/python src/run_v16.py run "Navigate to YouTube (https://www.youtube.com). Search for 'BR Chopra mahabharat episode 2' using the search bar and press enter. Click on the correct video from the search results to start playing it. Wait to verify the video has successfully started playing, and then mark the task as complete (this will automatically close the browser)."
