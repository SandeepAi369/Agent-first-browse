import asyncio
from advanced_agent import run_agent

if __name__ == "__main__":
    prompt = """Go to https://news.ycombinator.com/submit and submit this post.

Title: XeL Studio: The Power of an Autonomous Agentic Workflow

If you've been following my updates, you might already be familiar with XeL Studio. If you haven't seen it yet, a quick search on Google, Bing, Yahoo, or DuckDuckGo will bring it right up. Just look for the first Vercel-hosted result focused on "AI research and cybersecurity."

Today, let's dive into what makes XeL Studio truly unique: its fully autonomous, agentic workflow.

**A Self-Driving Newsroom**
The standout feature of the platform is the "AI Tech News" section, which operates entirely on autopilot. Every 30 minutes, the system wakes up, scours the internet for the latest tech news, writes an article, generates a relevant accompanying image, and publishes the complete post. 

More importantly, it is a self-managing system. The AI automatically cleans up its database every single day, deleting outdated news to keep the feed fresh. I haven't manually updated the code in two months, yet the platform continues to run flawlessly day in and day out with absolutely zero human intervention. 

**Interactive AI Labs**
Beyond the automated newsfeed, I’ve also added our Rising Chat AI into the platform's AI Labs section. This makes it incredibly easy for anyone to jump in and start chatting with the bot directly on Telegram.

**Under the Hood: A Robust Architecture**
Managing this level of automation requires a solid technical foundation. But why use three different databases? Since the project relies entirely on free tiers, storage space is strictly limited. By weaving together three distinct systems, I engineered a unified, high-capacity storage network that bypasses individual limits while keeping operations fast and cost-free.
*   **Supabase (PostgreSQL):** Acts as the primary brain for the news section, autonomously running queries and deleting old entries. 
*   **Firebase (NoSQL):** Powers the frontend, ensuring blazing-fast load times and handling local storage. 
*   **JSON-Configured Database:** A dedicated setup specifically tuned to manage the automated news pipeline without bloating the main databases.

**The Tech Stack & Infrastructure**
The workload is split across two main environments. Why not just run everything on Vercel? While Vercel is fantastic for hosting frontends, serverless functions have strict timeout limits that kill long-running background tasks. To bypass this, GitHub Actions acts as a dedicated, secondary backend server to handle the heavy lifting (like news scraping and AI generation), while Vercel handles serving the dynamic frontend.

Built from scratch with a unique concept, the platform leverages a modern stack. Why use multiple languages? Because each is perfectly suited for its specific job:
*   **Python:** The absolute powerhouse behind the scenes, handling data scraping, automation logic, and AI processing.
*   **Next.js, React, & TypeScript:** These dominate the frontend, working together to deliver a blazing-fast, highly interactive, and structurally secure user interface.

It’s a substantial codebase with over 50+ files. For an added touch, I’ve also integrated Microsoft’s text-to-speech voice to enhance accessibility.

**100% Free, Forever**
Despite the advanced AI integrations and heavy backend automation, the best thing about XeL Studio remains the same: it is completely free. Even as I plan to roll out more features and expand its capabilities, it will always remain at zero cost to the users. 

Search for it today and experience a truly agentic website in action!

GitHub Link: https://github.com/SandeepAi369/xel-studio"""
    asyncio.run(run_agent(prompt))
