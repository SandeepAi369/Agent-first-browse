import sys
import asyncio
import os

# Add orchestrator to path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), "python-orchestrator"))

from advanced_agent import run_agent

objective = """
MISSION: REDDIT ORGANIC STORYTELLING (STRICTLY NON-MARKETING)

TARGET PLATFORM: reddit.com/r/SideProject
OBJECTIVE: Share a genuine, human-like developer story about building "SearchWala". 

CRITICAL RULES FOR THIS RUN:
1. NO MARKETING MINDSET: You are NOT a marketer. You are a solo 22-year-old developer who just spent weeks building a complex project and wants to share the technical journey.
2. ZERO LINKS: DO NOT include any GitHub links, website links, or URLs in the post. If people want it, they will ask in the comments. 
3. ZERO AI JARGON: The Content Critic MUST ruthlessly block words like "leverage, revolutionize, cutting-edge, ultimate, game-changer." 
4. IMPERFECT TONE: Write like a real human on Reddit. Use lowercase sometimes, keep sentences conversational, and talk about the "struggle" (e.g., "managing 90+ concurrent search engines in Rust async was a nightmare").

EXECUTION STEPS:
1. AUTH: Navigate to https://reddit.com. The Smart Cookie Vault will inject existing Reddit session cookies.
2. NAVIGATE: Go to the https://reddit.com/r/SideProject subreddit and click "Create Post" or navigate directly to https://reddit.com/r/SideProject/submit
3. DRAFT TITLE: Something casual. Example: "I got frustrated with Google, so I built my own meta-search engine in Rust that queries 90 sites at once."
4. DRAFT BODY: Pass it through the Content Critic. Ensure it sounds like a raw developer diary.
5. STEALTH: Use GhostCursor to navigate and human-typing (with random micro-pauses) to fill the title and body.
6. PUBLISH: Click Post and record the success in CampaignMemory.
"""

if __name__ == "__main__":
    print("Starting Live Enterprise Deployment Test on Reddit...")
    asyncio.run(run_agent(objective))
