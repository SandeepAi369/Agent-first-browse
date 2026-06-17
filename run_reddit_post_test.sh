#!/bin/bash
cd "/home/sandeep/agent first IDE"
ARTICLE=$(cat article.txt)
.venv/bin/python run_v16.py run "Navigate to Reddit and go to the 'r/test' community. Find the Create Post button. Enter the title: 'Anthropic Fable 5 Review & Agent First IDE Release Update'. Enter the body with the following text: '$ARTICLE'. Do not submit until both title and body are filled. Once filled, click Post/Submit and verify the post is successfully published and live."
