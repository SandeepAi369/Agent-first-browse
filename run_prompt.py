import asyncio
import sys
from advanced_agent import run_agent

if __name__ == "__main__":
    with open("prompt.txt", "r") as f:
        prompt = f.read()
    asyncio.run(run_agent(prompt))
