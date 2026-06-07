import asyncio
from advanced_agent import run_agent

if __name__ == '__main__':
    with open('objective.txt', 'r', encoding='utf-8') as f:
        prompt = f.read()
    asyncio.run(run_agent(prompt))
