import sys
import os
import json
import asyncio
import aiohttp
import time
from client import FramePackClient

async def process_batch(json_path, base_url):
    with open(json_path, 'r') as f:
        items = json.load(f)
    if not isinstance(items, list):
        print("Input JSON must be an array of objects with 'image' and 'prompt'.")
        return
    base_dir = os.path.dirname(os.path.abspath(json_path))
    for entry in items:
        image_path = entry.get('image')
        prompt = entry.get('prompt', '')
        if image_path and not image_path.startswith('/'):
            image_path = os.path.join(base_dir, image_path)
        if not image_path or not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        mp4_path = os.path.splitext(image_path)[0] + '.mp4'
        if os.path.exists(mp4_path):
            print(f"MP4 already exists, skipping: {mp4_path}")
            continue
        print(f"Processing: {image_path} | Prompt: {prompt}")
        retry_count = 0
        max_retries = 3
        while retry_count <= max_retries:
            try:
                async with FramePackClient(base_url) as client:
                    async for update in client.generate_video(
                        image_path=image_path,
                        prompt=prompt,
                        output_dir=os.path.dirname(image_path),
                        save_previews=False
                    ):
                        if update.get('status') == 'completed':
                            local_mp4 = update.get('local_output_file')
                            if local_mp4 and os.path.exists(local_mp4):
                                # Move/rename to match the image's directory and name
                                if os.path.abspath(local_mp4) != os.path.abspath(mp4_path):
                                    os.replace(local_mp4, mp4_path)
                                print(f"Saved: {mp4_path}")
                            else:
                                print(f"Failed to save mp4 for {image_path}")
                            break
                        elif update.get('status') == 'error':
                            print(f"Error: {update.get('message')}")
                            break
                        else:
                            print(update)
                break  # Success, break retry loop
            except aiohttp.client_exceptions.ClientOSError as e:
                retry_count += 1
                print(f"ClientOSError encountered, retrying {retry_count}/{max_retries}...")
                time.sleep(2 * retry_count)
                if retry_count > max_retries:
                    print(f"Failed after {max_retries} retries: {e}")
                    break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch process images and prompts for FramePack-F1")
    parser.add_argument("json_file", help="Path to batch JSON file")
    parser.add_argument("--base-url", default="http://localhost:8000", help="FramePack API base URL")
    args = parser.parse_args()

    async def main():
        await process_batch(args.json_file, args.base_url)

    asyncio.run(main())
