import asyncio
from advanced_agent import run_agent
from paths import EXAMPLES_DIR

if __name__ == '__main__':
    with open(EXAMPLES_DIR / 'objectives' / 'objective.txt', 'r', encoding='utf-8') as f:
        prompt = f.read()
    asyncio.run(run_agent(prompt))
