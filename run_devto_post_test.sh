#!/bin/bash
cd "/home/sandeep/agent first IDE"
ARTICLE=$(cat article.txt)
.venv/bin/python run_v16.py run "Navigate to dev.to (https://dev.to). If you are not logged in, click 'Log in' and select the appropriate option (like Google or GitHub) to log in using a saved profile. Once logged in, navigate to create a new post (click 'Create Post'). Enter the title: 'Anthropic Fable 5 Review & Agent First IDE Release Update'. Enter the body with the following text: '$ARTICLE'. Do not submit until both title and body are filled. Once filled, click Publish to publish the post, and verify it is successfully published and live on the site."
