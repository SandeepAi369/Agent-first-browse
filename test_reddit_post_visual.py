import sys
import asyncio
import os

# Add orchestrator to path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), "python-orchestrator"))

from advanced_agent import run_agent

objective = """
MISSION: TEST REDDIT POST (VISUAL CURSOR VERIFICATION)

TARGET PLATFORM: reddit.com/r/test
OBJECTIVE: Make a simple, plain text post in /r/test to verify the new visual mouse cursor and GhostCursor clicking animation.

CRITICAL RULES FOR THIS RUN:
1. STRICTLY NO MARKETING: Do not use any marketing terms.
2. ZERO LINKS: Do not include any links.
3. SIMPLE TEXT: Just write a short sentence like "Testing out SearchWala local deployment pipeline" or similar.
4. OBSERVE MOUSE: The primary goal is to use GhostCursor to click "Create Post", type the title/body, and click "Post" so the observer can see the visual cursor animations.

EXECUTION STEPS:
1. NAVIGATE: Go directly to https://www.reddit.com/r/test/submit
2. DRAFT TITLE: "SearchWala local testing"
3. DRAFT BODY: "Just running a quick local test of the deployment pipeline. Nothing to see here."
4. PUBLISH: Click Post and output action_type="done".
"""

if __name__ == "__main__":
    print("Starting Visual Cursor Test on Reddit...")
    asyncio.run(run_agent(objective))
