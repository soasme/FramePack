import os
import base64
import io
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, default=7680)
parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (auto-detected based on GPUs if not specified)")
args = parser.parse_args()

print(args)

# Auto-detect GPU count and set workers
if args.workers is None:
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    args.workers = max(1, gpu_count)
    print(f"Auto-detected {gpu_count} GPUs, setting workers to {args.workers}")

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

# Initialize models
text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePack_F1_I2V_HY_20250503', torch_dtype=torch.bfloat16).cpu()

# Set models to evaluation mode
vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

# Set dtypes
transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

# Disable gradients
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

# Setup memory management
if not high_vram:
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

# FastAPI app
app = FastAPI(title="FramePack-F1 Video Generation API", version="1.0.0")

# Thread pool for async processing
thread_pool = ThreadPoolExecutor(max_workers=args.workers)

class VideoGenerationRequest(BaseModel):
    input_image: str  # base64 encoded image
    prompt: str = ""
    n_prompt: str = ""
    seed: int = 31337
    total_second_length: float = 5.0
    latent_window_size: int = 9
    steps: int = 25
    cfg: float = 1.0
    gs: float = 10.0
    rs: float = 0.0
    gpu_memory_preservation: float = 6.0
    use_teacache: bool = True
    mp4_crf: int = 16


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array image."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")


@torch.no_grad()
def video_generation_worker(request: VideoGenerationRequest):
    """Main video generation worker function."""
    
    # Decode input image
    input_image = decode_base64_image(request.input_image)
    
    total_latent_sections = (request.total_second_length * 30) / (request.latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()

    def yield_progress(status: str, progress: int = 0, preview_image: Optional[np.ndarray] = None, 
                      current_frames: int = 0, video_length: float = 0.0, file_path: Optional[str] = None):
        """Yield progress updates as JSON."""
        response_data = {
            "status": status,
            "progress": progress,
            "job_id": job_id,
            "current_frames": current_frames,
            "video_length": video_length
        }
        
        if preview_image is not None:
            # Convert preview to base64
            preview_pil = Image.fromarray(preview_image)
            buffer = io.BytesIO()
            preview_pil.save(buffer, format='PNG')
            preview_base64 = base64.b64encode(buffer.getvalue()).decode()
            response_data["preview_image"] = f"data:image/png;base64,{preview_base64}"
        
        if file_path is not None:
            response_data["output_file"] = file_path
            
        return json.dumps(response_data) + "\n"

    try:
        yield yield_progress("starting", 0)

        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding
        yield yield_progress("text_encoding", 5)

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(request.prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if request.cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(request.n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image
        yield yield_progress("image_processing", 10)

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        yield yield_progress("vae_encoding", 15)

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision
        yield yield_progress("clip_vision_encoding", 20)

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype conversion
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling
        yield yield_progress("sampling", 25)

        rnd = torch.Generator("cpu").manual_seed(request.seed)

        history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None

        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        total_generated_latent_frames = 1

        for section_index in range(total_latent_sections):
            print(f'section_index = {section_index}, total_latent_sections = {total_latent_sections}')

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=request.gpu_memory_preservation)

            if request.use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=request.steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                current_step = d['i'] + 1
                percentage = int(25 + (70 * current_step / request.steps))  # Progress from 25% to 95%
                current_frames = int(max(0, total_generated_latent_frames * 4 - 3))
                video_length = max(0, (total_generated_latent_frames * 4 - 3) / 30)
                
                return yield_progress("sampling", percentage, preview, current_frames, video_length)

            indices = torch.arange(0, sum([1, 16, 2, 1, request.latent_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, request.latent_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=request.latent_window_size * 4 - 3,
                real_guidance_scale=request.cfg,
                distilled_guidance_scale=request.gs,
                guidance_rescale=request.rs,
                num_inference_steps=request.steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=lambda d: None,  # Callback handled internally
            )

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = request.latent_window_size * 2
                overlapped_frames = request.latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=request.mp4_crf)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            current_frames = int(max(0, total_generated_latent_frames * 4 - 3))
            video_length = max(0, (total_generated_latent_frames * 4 - 3) / 30)
            yield yield_progress("section_complete", 95, None, current_frames, video_length, output_filename)

        yield yield_progress("completed", 100, None, current_frames, video_length, output_filename)

    except Exception as e:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        yield yield_progress("error", 0, None, 0, 0.0)
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")


@app.post("/")
async def generate_video(request: VideoGenerationRequest):
    """
    Generate video from input image and prompt.
    
    Returns a streaming response with JSON progress updates.
    """
    try:
        # Validate input image
        if not request.input_image:
            raise HTTPException(status_code=400, detail="input_image is required")
        
        # Run the video generation in thread pool
        loop = asyncio.get_event_loop()
        
        def run_generation():
            for progress_update in video_generation_worker(request):
                yield progress_update
        
        generator = await loop.run_in_executor(thread_pool, lambda: run_generation())
        
        return StreamingResponse(
            generator,
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "application/x-ndjson",
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "free_vram_gb": free_mem_gb,
        "high_vram_mode": high_vram
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "FramePack-F1 Video Generation API",
        "version": "1.0.0",
        "description": "Generate videos from images using FramePack-F1",
        "endpoints": {
            "POST /": "Generate video (streaming response)",
            "GET /health": "Health check",
            "GET /": "API information"
        }
    }


if __name__ == "__main__":
    print(f"Starting FramePack-F1 API server on {args.host}:{args.port}")
    print(f"Workers: {args.workers}")
    print(f"GPU Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,  # Use 1 worker since we handle concurrency with ThreadPoolExecutor
        log_level="info"
    )
