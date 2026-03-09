import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("https://bess-value-architect-fa6kvc5fvibgsez8z5tsoj.streamlit.app/")
        # Wait for the app to load
        await page.wait_for_timeout(5000) 
        await browser.close()

asyncio.run(run())
