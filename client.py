import asyncio
import aiohttp
import argparse
import base64
import json
import os
import sys
from typing import Optional, AsyncGenerator, Dict, Any
from pathlib import Path
import time

from PIL import Image
import io


class FramePackClient:
    """Client for FramePack-F1 Video Generation API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image file to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                
            # Detect image format from file extension
            ext = Path(image_path).suffix.lower()
            if ext in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            elif ext == '.png':
                mime_type = 'image/png'
            elif ext == '.webp':
                mime_type = 'image/webp'
            else:
                # Default to jpeg
                mime_type = 'image/jpeg'
            
            return f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            raise ValueError(f"Failed to encode image {image_path}: {str(e)}")
    
    def _decode_base64_image(self, base64_string: str) -> Image.Image:
        """
        Decode base64 string to PIL Image.
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            PIL Image object
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            return image
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check API health status.
        
        Returns:
            Health status information
        """
        if not self.session:
            raise RuntimeError("Client session not initialized. Use async context manager.")
        
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"Health check failed with status {response.status}"
                    )
        except Exception as e:
            raise RuntimeError(f"Health check failed: {str(e)}")
    
    async def generate_video(
        self,
        image_path: str,
        prompt: str,
        output_dir: str = "./outputs",
        n_prompt: str = "",
        seed: int = 31337,
        total_second_length: float = 5.0,
        latent_window_size: int = 9,
        steps: int = 25,
        cfg: float = 1.0,
        gs: float = 10.0,
        rs: float = 0.0,
        gpu_memory_preservation: float = 6.0,
        use_teacache: bool = True,
        mp4_crf: int = 16,
        save_previews: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate video from image and prompt.
        
        Args:
            image_path: Path to input image
            prompt: Text prompt for video generation
            output_dir: Directory to save outputs
            n_prompt: Negative prompt
            seed: Random seed
            total_second_length: Total video length in seconds
            latent_window_size: Latent window size
            steps: Number of sampling steps
            cfg: CFG scale
            gs: Distilled CFG scale
            rs: CFG rescale
            gpu_memory_preservation: GPU memory preservation in GB
            use_teacache: Whether to use TeaCache
            mp4_crf: MP4 compression quality
            save_previews: Whether to save preview images
            
        Yields:
            Progress updates and results
        """
        if not self.session:
            raise RuntimeError("Client session not initialized. Use async context manager.")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Encode input image
        try:
            input_image_b64 = self._encode_image_to_base64(image_path)
        except Exception as e:
            yield {"status": "error", "message": f"Failed to load input image: {str(e)}"}
            return
        
        # Prepare request payload
        payload = {
            "input_image": input_image_b64,
            "prompt": prompt,
            "n_prompt": n_prompt,
            "seed": seed,
            "total_second_length": total_second_length,
            "latent_window_size": latent_window_size,
            "steps": steps,
            "cfg": cfg,
            "gs": gs,
            "rs": rs,
            "gpu_memory_preservation": gpu_memory_preservation,
            "use_teacache": use_teacache,
            "mp4_crf": mp4_crf
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    yield {"status": "error", "message": f"API request failed: {error_text}"}
                    return
                
                preview_count = 0
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Save preview images if available
                        if save_previews and "preview_image" in data:
                            try:
                                preview_image = self._decode_base64_image(data["preview_image"])
                                preview_filename = os.path.join(
                                    output_dir, 
                                    f"{data.get('job_id', 'preview')}_{preview_count:04d}.png"
                                )
                                preview_image.save(preview_filename)
                                data["preview_saved"] = preview_filename
                                preview_count += 1
                            except Exception as e:
                                print(f"Warning: Failed to save preview image: {e}")

                        # Download output file if provided and status is completed
                        if data.get("status") == "completed" and "output_file" in data:
                            output_file_name = os.path.basename(data["output_file"])
                            download_url = f"{self.base_url}/download?file={output_file_name}"
                            local_filename = os.path.join(output_dir, output_file_name)
                            async with self.session.get(download_url) as dl_resp:
                                if dl_resp.status == 200:
                                    with open(local_filename, "wb") as f:
                                        f.write(await dl_resp.read())
                                    data["local_output_file"] = local_filename
                                else:
                                    data["local_output_file"] = None
                                    data["download_error"] = f"Failed to download: {dl_resp.status}"
                        
                        yield data
                        
                    except json.JSONDecodeError as e:
                        yield {"status": "error", "message": f"Failed to parse response: {str(e)}"}
                        continue
                
        except Exception as e:
            import traceback; traceback.print_exc()
            yield {"status": "error", "message": f"Request failed: {str(e)}"}


async def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="FramePack-F1 Video Generation Client")
    
    # Required arguments
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("prompt", help="Text prompt for video generation")
    
    # Optional arguments
    parser.add_argument("--api-url", default="http://localhost:8000", help="API server URL")
    parser.add_argument("--output-dir", default="./client_outputs", help="Output directory")
    parser.add_argument("--n-prompt", default="", help="Negative prompt")
    parser.add_argument("--seed", type=int, default=31337, help="Random seed")
    parser.add_argument("--length", type=float, default=5.0, help="Video length in seconds")
    parser.add_argument("--steps", type=int, default=25, help="Number of sampling steps")
    parser.add_argument("--gs", type=float, default=10.0, help="Distilled CFG scale")
    parser.add_argument("--gpu-memory", type=float, default=6.0, help="GPU memory preservation (GB)")
    parser.add_argument("--no-teacache", action="store_true", help="Disable TeaCache")
    parser.add_argument("--mp4-crf", type=int, default=16, help="MP4 compression quality")
    parser.add_argument("--no-previews", action="store_true", help="Don't save preview images")
    parser.add_argument("--health-check", action="store_true", help="Only perform health check")
    
    args = parser.parse_args()
    
    # Create client
    async with FramePackClient(args.api_url) as client:
        
        # Health check mode
        if args.health_check:
            try:
                health = await client.health_check()
                print("API Health Status:")
                print(json.dumps(health, indent=2))
                return
            except Exception as e:
                print(f"Health check failed: {e}")
                sys.exit(1)
        
        # Validate input image
        if not os.path.exists(args.image):
            print(f"Error: Input image '{args.image}' not found")
            sys.exit(1)
        
        print(f"Starting video generation...")
        print(f"Input image: {args.image}")
        print(f"Prompt: {args.prompt}")
        print(f"Output directory: {args.output_dir}")
        print(f"API URL: {args.api_url}")
        print("-" * 50)
        
        # Generate video
        start_time = time.time()
        final_output = None
        
        try:
            async for update in client.generate_video(
                image_path=args.image,
                prompt=args.prompt,
                output_dir=args.output_dir,
                n_prompt=args.n_prompt,
                seed=args.seed,
                total_second_length=args.length,
                steps=args.steps,
                gs=args.gs,
                gpu_memory_preservation=args.gpu_memory,
                use_teacache=not args.no_teacache,
                mp4_crf=args.mp4_crf,
                save_previews=not args.no_previews
            ):
                
                status = update.get("status", "unknown")
                progress = update.get("progress", 0)
                
                if status == "error":
                    print(f"Error: {update.get('message', 'Unknown error')}")
                    sys.exit(1)
                
                elif status == "completed":
                    final_output = update.get("local_output_file")
                    elapsed_time = time.time() - start_time
                    print(f"\n✅ Generation completed in {elapsed_time:.1f} seconds!")
                    
                    if final_output:
                        print(f"📹 Video saved: {final_output}")
                    else:
                        print("⚠️  No output file generated: {update.get('download_error', 'Unknown error')}")
                    
                    if "current_frames" in update:
                        print(f"🎬 Total frames: {update['current_frames']}")
                        print(f"⏱️  Video length: {update.get('video_length', 0):.2f} seconds")
                
                else:
                    # Progress update
                    progress_bar = "█" * (progress // 5) + "░" * (20 - progress // 5)
                    print(f"\r[{progress_bar}] {progress}% - {status.replace('_', ' ').title()}", end="", flush=True)
                    
                    if "preview_saved" in update:
                        print(f"\n💾 Preview saved: {update['preview_saved']}")
                    
                    if "current_frames" in update and update["current_frames"] > 0:
                        print(f" | Frames: {update['current_frames']} | Length: {update.get('video_length', 0):.1f}s", end="")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Generation interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n❌ Generation failed: {e}")
            sys.exit(1)
        
        if final_output and os.path.exists(final_output):
            file_size = os.path.getsize(final_output) / (1024 * 1024)  # MB
            print(f"📊 File size: {file_size:.1f} MB")
        else:
            print("⚠️  No output file generated")


if __name__ == "__main__":
    asyncio.run(main())
