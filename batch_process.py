import sys
import os
import json
import asyncio
from client import FramePackClient

async def process_batch(json_path, client):
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
        print(f"Processing: {image_path} | Prompt: {prompt}")
        mp4_path = os.path.splitext(image_path)[0] + '.mp4'
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch process images and prompts for FramePack-F1")
    parser.add_argument("json_file", help="Path to batch JSON file")
    parser.add_argument("--base-url", default="http://localhost:8000", help="FramePack API base URL")
    args = parser.parse_args()

    async def main():
        async with FramePackClient(args.base_url) as client:
            await process_batch(args.json_file, client)

    asyncio.run(main())
