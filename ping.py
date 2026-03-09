import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("[Your App Link]")
        # Wait for the app to load
        await page.wait_for_timeout(5000) 
        await browser.close()

asyncio.run(run())
