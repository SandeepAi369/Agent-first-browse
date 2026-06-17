#!/bin/bash
cd "/home/sandeep/agent first IDE"
ARTICLE=$(cat article.txt)
.venv/bin/python run_v16.py run "Navigate to Hashnode (https://hashnode.com). If you are not logged in, click 'Log in' and select 'Continue with Google' to log in using the saved Google profile. Once logged in, navigate to create a new draft/post (Write). Enter the title: 'Anthropic Fable 5 Review & Agent First IDE Release Update'. Enter the body with the following text: '$ARTICLE'. Do not submit until both title and body are filled. Once filled, click Publish to publish the post, and verify it is live."
