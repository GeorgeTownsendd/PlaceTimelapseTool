import aiohttp
import aiohttp
import asyncio
import os
import logging
from datetime import datetime


def get_tile_urls():
    return [['http://exampleimage.com', (0, 0)]]


# Set up logging
logging.basicConfig(filename='error.log', level=logging.ERROR)


async def fetch_tile(session, url, path, fetch_interval):
    delay = fetch_interval
    while True:
        try:
            async with session.get(url) as response:
                with open(path, 'wb') as f:
                    f.write(await response.read())
            return
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
            delay = min(delay * 2, 600)  # Double the delay, up to a maximum of 600 seconds
            await asyncio.sleep(delay)


async def fetch_all_tiles(fetch_interval, event_name):
    base_dir = os.path.join('canvas_data', event_name)
    async with aiohttp.ClientSession() as session:
        while True:
            tasks = []
            for url, coordinates in get_tile_urls():
                quadrant_dir = os.path.join(base_dir, f'quadrant_{coordinates[0]}_{coordinates[1]}')
                os.makedirs(quadrant_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                img_path = os.path.join(quadrant_dir, f'image_{timestamp}.png')
                task = fetch_tile(session, url, img_path, fetch_interval)
                tasks.append(task)
            await asyncio.gather(*tasks)
            await asyncio.sleep(fetch_interval)

# Start the main event loop
loop = asyncio.get_event_loop()
loop.run_until_complete(fetch_all_tiles(fetch_interval=60, event_name='place_2023'))
