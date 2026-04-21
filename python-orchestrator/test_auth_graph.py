import asyncio
from app.browser_promoter.google_stealth_auth_graph import run_google_stealth_login

async def main():
    try:
        await run_google_stealth_login('deepsearch123@gmail.com', '8688566123Sa', 'profile_deepsearch_test_01')
    except Exception as e:
        print(f'Error during stealth login: {e}')

if __name__ == '__main__':
    asyncio.run(main())
