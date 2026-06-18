"""
LTX-2 FastAPI Service
RESTful API for video generation with preset support and real-time SSE streaming.

Features:
- Generate videos using presets or custom settings
- Real-time progress streaming via Server-Sent Events (SSE)
- Preset management (CRUD operations)
- Health check and status endpoints

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000
    
    # Or with auto-reload for development
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import time
import uuid
import base64
import asyncio
import threading
import json
import io
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download, snapshot_download

from presets import (
    get_preset_manager,
    PresetManager,
    GenerationPreset,
    DEFAULT_PRESET_NAME,
)
from ltx2_compat import (
    CHECKPOINTS,
    DEFAULT_CHECKPOINT_CANDIDATES,
    DEFAULT_CHECKPOINT_KEY,
    DEFAULT_DISTILLED_LORA_KEY,
    DEFAULT_IC_LORA_KEY,
    DEFAULT_INGREDIENTS_IC_LORA_KEY,
    DEFAULT_UPSAMPLER_CANDIDATES,
    DEFAULT_UPSAMPLER_KEY,
    GEMMA_REPO_ID,
    GUIDANCELESS_PIPELINES,
    HF_REPO_ID,
    LTX23_SPECIALIZED_PIPELINE_TYPES,
    ONE_STAGE_PIPELINES,
    PIPELINES_REQUIRING_DISTILLED_LORA,
    PIPELINES_REQUIRING_UPSAMPLER,
    SUPPORTED_PIPELINE_TYPES,
    build_guidance_kwargs,
    create_pipeline,
    encode_video_compat,
    is_ltx23_model_path,
    missing_gemma_files,
    make_image_conditioning,
    model_repo_id,
    release_pipeline_models,
    validate_gemma_root,
)

# Constants
MODELS_DIR = Path("./models")
OUTPUTS_DIR = Path("./outputs")
TEMPLATES_DIR = Path("./templates")
AUDIO_VIDEO_RESOLUTION_MULTIPLE = 128

# ============================================================================
# Progress Callback System for SSE Streaming
# ============================================================================

# Thread-local storage for progress callbacks
_progress_callbacks = threading.local()

def get_progress_callback():
    """Get the current thread's progress callback."""
    return getattr(_progress_callbacks, 'callback', None)

def set_progress_callback(callback):
    """Set the current thread's progress callback."""
    _progress_callbacks.callback = callback

import re
import sys


class OutputCapture:
    """Capture stdout/stderr and parse tqdm progress for SSE streaming.
    
    Stage detection is based on step count patterns, NOT sequence position:
    - Distilled: Stage 1 (8 steps), Stage 2 (3 steps), Encoding (variable)
    - Two-stage: Stage 1 (num_inference_steps), Stage 2 (3 steps), Encoding (variable)
    - One-stage: Stage 1 (num_inference_steps), Encoding (variable)
    
    Loading bars from safetensors may or may not appear depending on model caching.
    """
    
    # Shared state across stdout/stderr captures
    _shared_state = {}
    
    # Known step counts for specific stages
    DISTILLED_STAGE1_STEPS = 8  # DISTILLED_SIGMA_VALUES has 9 values, tqdm uses sigmas[:-1]
    STAGE2_REFINEMENT_STEPS = 3  # STAGE_2_DISTILLED_SIGMA_VALUES has 4 values, tqdm uses sigmas[:-1]
    
    def __init__(self, progress_callback, original_stream, capture_id, pipeline_type=None, num_inference_steps=None):
        self.progress_callback = progress_callback
        self.original_stream = original_stream
        self.capture_id = capture_id
        
        # Initialize shared state for this capture session
        if capture_id not in OutputCapture._shared_state:
            OutputCapture._shared_state[capture_id] = {
                "pipeline_type": pipeline_type,
                "num_inference_steps": num_inference_steps,
                # Track completed stages by step count (to distinguish loading vs stage 2)
                "stage1_completed": False,  # Have we completed Stage 1?
                "stage2_completed": False,  # Have we completed Stage 2?
                "current_stage_total": None,  # Current bar's step count
                "current_stage_name": None,  # Current stage name
                "current_bar_completed": False,  # Has current bar hit 100%?
            }
        
        # tqdm patterns - matches progress like "  5%|███  | 1/8 [00:01<00:05, 1.25it/s]"
        # Extended pattern to capture time info: [elapsed<remaining, rate]
        self.tqdm_pattern = re.compile(r'^\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]')
        # Fallback without time info
        self.tqdm_pattern_simple = re.compile(r'^\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)')
        # Alternative pattern for simpler tqdm output
        self.tqdm_simple = re.compile(r'(\d+)/(\d+)\s+\[([^\]]+)\]')
    
    @property
    def state(self):
        return OutputCapture._shared_state[self.capture_id]
    
    def _determine_stage(self, total, percentage):
        """Determine stage name based on step count patterns and pipeline type.
        
        Detection logic:
        1. For fixed-sigma pipelines: Stage 1 = 8 steps, Stage 2 = 3 steps
        2. For CFG pipelines: Stage 1 = num_inference_steps, Stage 2 = 3 steps
        3. 3 steps BEFORE Stage 1 completes = Loading Models
        4. 3 steps AFTER Stage 1 completes = Stage 2 Refinement (for two-stage)
        5. Any steps after Stage 2 (or Stage 1 for one-stage) = Video Encoding
        """
        pipeline_type = self.state.get("pipeline_type")
        num_inference_steps = self.state.get("num_inference_steps")
        stage1_completed = self.state.get("stage1_completed", False)
        stage2_completed = self.state.get("stage2_completed", False)
        current_stage_total = self.state.get("current_stage_total")
        current_stage_name = self.state.get("current_stage_name")
        current_bar_completed = self.state.get("current_bar_completed", False)
        
        is_one_stage = pipeline_type in ONE_STAGE_PIPELINES
        is_distilled = pipeline_type in GUIDANCELESS_PIPELINES
        
        # If still in the same bar (same total and bar not completed), return cached name
        if current_stage_total == total and current_stage_name and not current_bar_completed:
            # Check if bar just completed
            if percentage >= 100:
                self.state["current_bar_completed"] = True
                self._mark_stage_completed(current_stage_name)
            return current_stage_name
        
        # New bar detected (different total or previous bar completed) - determine stage
        stage_name = self._identify_stage_by_steps(
            total, stage1_completed, stage2_completed, 
            is_one_stage, is_distilled, num_inference_steps
        )
        
        # Update state for new bar
        self.state["current_stage_total"] = total
        self.state["current_stage_name"] = stage_name
        self.state["current_bar_completed"] = (percentage >= 100)
        
        if percentage >= 100:
            self._mark_stage_completed(stage_name)
        
        return stage_name
    
    def _identify_stage_by_steps(self, total, stage1_completed, stage2_completed, 
                                   is_one_stage, is_distilled, num_inference_steps):
        """Identify stage based on step count and pipeline state."""
        
        # Determine expected Stage 1 step count
        expected_stage1_steps = self.DISTILLED_STAGE1_STEPS if is_distilled else num_inference_steps
        
        # Stage 1 detection: matches expected step count
        if total == expected_stage1_steps and not stage1_completed:
            return "Stage 1: Denoising"
        
        # Stage 2 detection (3 steps, only for two-stage pipelines)
        if total == self.STAGE2_REFINEMENT_STEPS:
            if not stage1_completed:
                # 3 steps before Stage 1 = Loading (from safetensors)
                return "Loading Models"
            elif not stage2_completed and not is_one_stage:
                # 3 steps after Stage 1 = Stage 2 Refinement
                return "Stage 2: Refinement"
        
        # After main stages complete, it's encoding
        if is_one_stage:
            if stage1_completed:
                return "Video Encoding"
        else:
            if stage1_completed and stage2_completed:
                return "Video Encoding"
            elif stage1_completed:
                # Still waiting for Stage 2, but got unexpected step count
                # Could be encoding if Stage 2 was skipped or has different steps
                return "Video Encoding"
        
        # Fallback for unexpected step counts
        if not stage1_completed:
            return "Loading Models"
        return "Video Encoding"
    
    def _mark_stage_completed(self, stage_name):
        """Mark a stage as completed based on its name."""
        if "Stage 1" in stage_name:
            self.state["stage1_completed"] = True
        elif "Stage 2" in stage_name:
            self.state["stage2_completed"] = True
    
    def write(self, text):
        # Also write to original stream
        self.original_stream.write(text)
        
        # Try to parse tqdm progress
        if text.strip():
            self._parse_progress(text)
    
    def _parse_time_info(self, time_str):
        """Parse tqdm time info like '00:08<00:00, 1.04s/it' into structured data."""
        time_info = {"elapsed": None, "remaining": None, "rate": None}
        try:
            # Split by comma to get elapsed<remaining and rate
            parts = time_str.split(',')
            if parts:
                # Parse elapsed<remaining
                time_part = parts[0].strip()
                if '<' in time_part:
                    elapsed, remaining = time_part.split('<')
                    time_info["elapsed"] = elapsed.strip()
                    time_info["remaining"] = remaining.strip()
                # Parse rate (e.g., "1.04s/it" or "1.25it/s")
                if len(parts) > 1:
                    time_info["rate"] = parts[1].strip()
        except Exception:
            pass
        return time_info
    
    def _parse_progress(self, text):
        # Check for tqdm progress pattern with time info
        match = self.tqdm_pattern.search(text)
        if match:
            percentage = int(match.group(1))
            current = int(match.group(2))
            total = int(match.group(3))
            time_str = match.group(4)
            time_info = self._parse_time_info(time_str)
            
            stage = self._determine_stage(total, percentage)
            
            self.progress_callback({
                "type": "step",
                "stage": stage,
                "step": current,
                "total": total,
                "percentage": percentage,
                "elapsed": time_info.get("elapsed"),
                "remaining": time_info.get("remaining"),
                "rate": time_info.get("rate")
            })
            return
        
        # Try fallback pattern without time
        match = self.tqdm_pattern_simple.search(text)
        if match:
            percentage = int(match.group(1))
            current = int(match.group(2))
            total = int(match.group(3))
            
            stage = self._determine_stage(total, percentage)
            
            self.progress_callback({
                "type": "step",
                "stage": stage,
                "step": current,
                "total": total,
                "percentage": percentage
            })
            return
        
        # Try simple pattern with time
        match = self.tqdm_simple.search(text)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            time_str = match.group(3)
            time_info = self._parse_time_info(time_str)
            percentage = round(current / total * 100, 1) if total > 0 else 0
            
            stage = self._determine_stage(total, percentage)
            
            self.progress_callback({
                "type": "step",
                "stage": stage,
                "step": current,
                "total": total,
                "percentage": percentage,
                "elapsed": time_info.get("elapsed"),
                "remaining": time_info.get("remaining"),
                "rate": time_info.get("rate")
            })
    
    def flush(self):
        self.original_stream.flush()
    
    def isatty(self):
        return self.original_stream.isatty()


class OutputCaptureContext:
    """Context manager to capture stdout/stderr during generation."""
    
    def __init__(self, progress_callback, pipeline_type=None, num_inference_steps=None):
        self.progress_callback = progress_callback
        self.original_stdout = None
        self.original_stderr = None
        # Unique capture ID for this context (shared between stdout/stderr)
        self.capture_id = id(self)
        self.pipeline_type = pipeline_type
        self.num_inference_steps = num_inference_steps
    
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = OutputCapture(self.progress_callback, self.original_stdout, self.capture_id, self.pipeline_type, self.num_inference_steps)
        sys.stderr = OutputCapture(self.progress_callback, self.original_stderr, self.capture_id, self.pipeline_type, self.num_inference_steps)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        # Clean up shared state
        if self.capture_id in OutputCapture._shared_state:
            del OutputCapture._shared_state[self.capture_id]
        return False

def ensure_directories():
    """Create necessary directories."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "checkpoints").mkdir(exist_ok=True)
    (MODELS_DIR / "loras").mkdir(exist_ok=True)
    (MODELS_DIR / "upsamplers").mkdir(exist_ok=True)
    (MODELS_DIR / "gemma").mkdir(exist_ok=True)
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


def get_model_path(model_key: str) -> Optional[Path]:
    """Get local path for a model if it exists."""
    if model_key not in CHECKPOINTS:
        return None
    
    model_info = CHECKPOINTS[model_key]
    model_type = model_info["type"]
    
    if model_type == "checkpoint":
        path = MODELS_DIR / "checkpoints" / model_info["filename"]
    elif model_type == "lora":
        path = MODELS_DIR / "loras" / model_info["filename"]
    elif model_type == "upscaler":
        path = MODELS_DIR / "upsamplers" / model_info["filename"]
    else:
        path = MODELS_DIR / model_info["filename"]
    
    return path if path.exists() else None


def check_model_status(model_key: str) -> tuple:
    """Check if a model is downloaded. Returns (status, path)."""
    path = get_model_path(model_key)
    if path and path.exists():
        # Always use forward slashes for cross-platform consistency
        return "ready", str(path).replace("\\", "/")
    return "missing", None


def download_model_file(model_key: str) -> str:
    """Download a model from HuggingFace. Returns the path."""
    if model_key not in CHECKPOINTS:
        raise ValueError(f"Unknown model: {model_key}")
    
    model_info = CHECKPOINTS[model_key]
    model_type = model_info["type"]
    filename = model_info["filename"]
    
    # Determine target directory
    if model_type == "checkpoint":
        target_dir = MODELS_DIR / "checkpoints"
    elif model_type == "lora":
        target_dir = MODELS_DIR / "loras"
    elif model_type == "upscaler":
        target_dir = MODELS_DIR / "upsamplers"
    else:
        target_dir = MODELS_DIR
    
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    
    if target_path.exists():
        return str(target_path)
    
    downloaded_path = hf_hub_download(
        repo_id=model_repo_id(model_key),
        filename=filename,
        local_dir=target_dir,
    )
    
    return downloaded_path


def download_gemma_model(gemma_path: str = "./models/gemma") -> str:
    """Download the Gemma text encoder snapshot expected by LTX-2.3."""
    target_dir = Path(gemma_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=GEMMA_REPO_ID,
        local_dir=target_dir,
    )


def get_default_checkpoint() -> Optional[str]:
    """Get the default checkpoint path, preferring LTX-2.3 distilled."""
    checkpoint_dir = MODELS_DIR / "checkpoints"
    if checkpoint_dir.exists():
        for name in DEFAULT_CHECKPOINT_CANDIDATES:
            path = checkpoint_dir / name
            if path.exists():
                return str(path)
        checkpoints = [path for path in checkpoint_dir.glob("*.safetensors") if is_ltx23_model_path(path)]
        if checkpoints:
            return str(checkpoints[0])
    return None


def get_default_upsampler() -> Optional[str]:
    """Get the default spatial upsampler path, preferring LTX-2.3."""
    upsampler_dir = MODELS_DIR / "upsamplers"
    if upsampler_dir.exists():
        for name in DEFAULT_UPSAMPLER_CANDIDATES:
            path = upsampler_dir / name
            if path.exists():
                return str(path)
        upsamplers = [path for path in upsampler_dir.glob("*.safetensors") if is_ltx23_model_path(path)]
        if upsamplers:
            return str(upsamplers[0])
    return None


def get_default_lora() -> Optional[str]:
    """Get the default distilled LoRA path if one is already downloaded."""
    lora_dir = MODELS_DIR / "loras"
    if lora_dir.exists():
        default_name = CHECKPOINTS[DEFAULT_DISTILLED_LORA_KEY]["filename"]
        default_path = lora_dir / default_name
        if default_path.exists():
            return str(default_path)
    return None


def _create_lipdub_reference_video(
    image_path: str,
    audio_path: str,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: float,
    temp_dir: Path,
    audio_start_time: float = 0.0,
    audio_max_duration: float | None = None,
) -> str:
    """Build a static reference video from a portrait image plus speech audio."""
    duration = max(0.1, float(audio_max_duration or (float(num_frames) / max(float(frame_rate), 1.0))))
    output_path = temp_dir / f"lipdub_ref_{uuid.uuid4().hex[:8]}.mp4"
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg = get_ffmpeg_exe()
    except Exception:
        ffmpeg = "ffmpeg"

    vf = (
        f"scale={int(width)}:{int(height)}:force_original_aspect_ratio=increase,"
        f"crop={int(width)}:{int(height)},format=yuv420p"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-loop",
        "1",
        "-framerate",
        str(float(frame_rate)),
        "-i",
        image_path,
    ]
    if audio_start_time > 0:
        cmd.extend(["-ss", f"{float(audio_start_time):.6f}"])
    cmd.extend([
        "-i",
        audio_path,
        "-t",
        f"{duration:.6f}",
        "-vf",
        vf,
        "-r",
        str(float(frame_rate)),
        "-frames:v",
        str(int(num_frames)),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-af",
        "apad",
        "-c:a",
        "aac",
        str(output_path),
    ])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create lipdub reference video: {result.stderr[-1200:]}")
    return str(output_path)


def _create_static_reference_video(
    image_path: str,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: float,
    temp_dir: Path,
) -> str:
    """Loop a still reference image into the static video expected by IC-LoRA."""
    duration = max(0.1, float(num_frames) / max(float(frame_rate), 1.0))
    output_path = temp_dir / f"ic_lora_ref_{uuid.uuid4().hex[:8]}.mp4"
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg = get_ffmpeg_exe()
    except Exception:
        ffmpeg = "ffmpeg"

    vf = (
        f"scale={int(width)}:{int(height)}:force_original_aspect_ratio=increase,"
        f"crop={int(width)}:{int(height)},format=yuv420p"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-loop",
        "1",
        "-framerate",
        str(float(frame_rate)),
        "-i",
        image_path,
        "-t",
        f"{duration:.6f}",
        "-vf",
        vf,
        "-r",
        str(float(frame_rate)),
        "-frames:v",
        str(int(num_frames)),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create IC-LoRA reference video: {result.stderr[-1200:]}")
    return str(output_path)


def _validate_audio_video_shape(width: int, height: int, num_frames: int) -> None:
    if width % AUDIO_VIDEO_RESOLUTION_MULTIPLE != 0 or height % AUDIO_VIDEO_RESOLUTION_MULTIPLE != 0:
        raise ValueError(
            "Audio-guided pipelines require width and height to be multiples of 128. "
            f"Got {width}x{height}."
        )
    if (int(num_frames) - 9) % 16 != 0:
        raise ValueError(
            "Audio-guided pipelines require num_frames to follow 16n+9 "
            f"(for example 73, 121, 169). Got {num_frames}."
        )


def _conform_audio_latent_length(latent: torch.Tensor, expected_frames: int) -> torch.Tensor:
    actual_frames = int(latent.shape[2])
    if actual_frames > expected_frames:
        return latent[:, :, :expected_frames]
    if actual_frames < expected_frames:
        pad_shape = list(latent.shape)
        pad_shape[2] = expected_frames - actual_frames
        padding = torch.zeros(pad_shape, device=latent.device, dtype=latent.dtype)
        return torch.cat([latent, padding], dim=2)
    return latent


def _ensure_distilled_audio_conditioner(pipeline: Any):
    audio_conditioner = getattr(pipeline, "audio_conditioner", None)
    if audio_conditioner is not None:
        return audio_conditioner

    from ltx_pipelines.utils.blocks import AudioConditioner

    checkpoint_path = getattr(getattr(pipeline, "stage", None), "_checkpoint_path", None)
    if not checkpoint_path:
        raise ValueError("The cached distilled pipeline does not expose its checkpoint path.")
    audio_conditioner = AudioConditioner(checkpoint_path, pipeline.dtype, pipeline.device)
    setattr(pipeline, "audio_conditioner", audio_conditioner)
    return audio_conditioner


def _run_distilled_portrait_lipsync(
    pipeline: Any,
    *,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    images: List[Any],
    audio_path: str,
    audio_start_time: float,
    audio_max_duration: float,
    tiling_config: Any,
):
    """Fast portrait lipsync path that reuses the cached distilled pipeline modules."""
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
    from ltx_core.types import Audio, AudioLatentShape
    from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
    from ltx_pipelines.utils.denoisers import SimpleDenoiser
    from ltx_pipelines.utils.helpers import assert_resolution, combined_image_conditionings
    from ltx_pipelines.utils.media_io import decode_audio_from_file
    from ltx_pipelines.utils.types import ModalitySpec

    assert_resolution(height=height, width=width, is_two_stage=True)
    if not images:
        raise ValueError("A portrait image is required for distilled portrait lipsync.")

    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)
    dtype = torch.bfloat16

    portrait = images[0]._replace(frame_idx=0, strength=0.8)

    (ctx_p,) = pipeline.prompt_encoder(
        [prompt],
        enhance_first_prompt=False,
        enhance_prompt_image=portrait.path,
    )
    video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

    decoded_audio = decode_audio_from_file(
        audio_path,
        pipeline.device,
        audio_start_time,
        audio_max_duration,
    )
    if decoded_audio is None:
        raise ValueError(f"Failed to decode audio from {audio_path}. Please check the file and try again.")

    audio_conditioner = _ensure_distilled_audio_conditioner(pipeline)
    encoded_audio_latent = audio_conditioner(lambda enc: vae_encode_audio(decoded_audio, enc, None))
    audio_shape = AudioLatentShape.from_duration(
        batch=1,
        duration=float(num_frames) / float(frame_rate),
        channels=8,
        mel_bins=16,
    )
    encoded_audio_latent = _conform_audio_latent_length(encoded_audio_latent, audio_shape.frames)

    stage_1_sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=pipeline.device)
    stage_1_w, stage_1_h = width // 2, height // 2
    stage_1_conditionings = pipeline.image_conditioner(
        lambda enc: combined_image_conditionings(
            images=[portrait],
            height=stage_1_h,
            width=stage_1_w,
            video_encoder=enc,
            dtype=dtype,
            device=pipeline.device,
        )
    )

    video_state, _audio_state = pipeline.stage(
        denoiser=SimpleDenoiser(video_context, audio_context),
        sigmas=stage_1_sigmas,
        noiser=noiser,
        width=stage_1_w,
        height=stage_1_h,
        frames=num_frames,
        fps=frame_rate,
        video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
        audio=ModalitySpec(
            context=audio_context,
            frozen=True,
            noise_scale=0.0,
            initial_latent=encoded_audio_latent,
        ),
    )

    upscaled_video_latent = pipeline.upsampler(video_state.latent[:1])
    stage_2_sigmas = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=pipeline.device)
    stage_2_conditionings = pipeline.image_conditioner(
        lambda enc: combined_image_conditionings(
            images=[portrait],
            height=height,
            width=width,
            video_encoder=enc,
            dtype=dtype,
            device=pipeline.device,
        )
    )

    video_state, _audio_state = pipeline.stage(
        denoiser=SimpleDenoiser(video_context, audio_context),
        sigmas=stage_2_sigmas,
        noiser=noiser,
        width=width,
        height=height,
        frames=num_frames,
        fps=frame_rate,
        video=ModalitySpec(
            context=video_context,
            conditionings=stage_2_conditionings,
            noise_scale=stage_2_sigmas[0].item(),
            initial_latent=upscaled_video_latent,
        ),
        audio=ModalitySpec(
            context=audio_context,
            frozen=True,
            noise_scale=0.0,
            initial_latent=encoded_audio_latent,
        ),
    )

    decoded_video = pipeline.video_decoder(video_state.latent, tiling_config, generator)
    original_audio = Audio(waveform=decoded_audio.waveform.squeeze(0), sampling_rate=decoded_audio.sampling_rate)
    return decoded_video, original_audio


def auto_download_required_models(preset, ic_lora_model_key: str | None = None) -> tuple:
    """
    Auto-download required models if not present.
    Returns (checkpoint_path, upsampler_path) or raises error.
    """
    checkpoint_path = preset.checkpoint_path
    upsampler_path = preset.spatial_upsampler_path

    if preset.pipeline_type == "a2vid_two_stage":
        dev_key = "ltx-2.3-22b-dev"
        existing_dev = get_model_path(dev_key)
        if existing_dev:
            checkpoint_path = str(existing_dev)
        else:
            print(f"Auto-downloading audio-video checkpoint: {dev_key}")
            checkpoint_path = download_model_file(dev_key)
    elif preset.pipeline_type in {"distilled", "ic_lora", "lipdub"}:
        dev_filename = CHECKPOINTS["ltx-2.3-22b-dev"]["filename"]
        if checkpoint_path and Path(checkpoint_path).name == dev_filename:
            existing_distilled = get_model_path(DEFAULT_CHECKPOINT_KEY)
            if existing_distilled:
                checkpoint_path = str(existing_distilled)
            else:
                print(f"Auto-downloading default checkpoint: {DEFAULT_CHECKPOINT_KEY}")
                checkpoint_path = download_model_file(DEFAULT_CHECKPOINT_KEY)
    
    # Check and download checkpoint
    if not checkpoint_path or not Path(checkpoint_path).exists():
        # Try to find existing checkpoint
        existing = get_default_checkpoint()
        if existing:
            checkpoint_path = existing
        else:
            print(f"Auto-downloading default checkpoint: {DEFAULT_CHECKPOINT_KEY}")
            checkpoint_path = download_model_file(DEFAULT_CHECKPOINT_KEY)
    
    # Check and download upsampler for pipelines that need it
    if preset.pipeline_type in PIPELINES_REQUIRING_UPSAMPLER:
        if not upsampler_path or not Path(upsampler_path).exists():
            existing = get_default_upsampler()
            if existing:
                upsampler_path = existing
            else:
                # Download spatial upscaler
                print("Auto-downloading spatial upscaler")
                upsampler_path = download_model_file(DEFAULT_UPSAMPLER_KEY)

    if preset.pipeline_type in PIPELINES_REQUIRING_DISTILLED_LORA:
        keep_existing_ic_lora = False
        if preset.pipeline_type == "lipdub":
            required_lora_key = DEFAULT_IC_LORA_KEY
        elif preset.pipeline_type == "ic_lora":
            required_lora_key = ic_lora_model_key or DEFAULT_INGREDIENTS_IC_LORA_KEY
            if required_lora_key not in CHECKPOINTS:
                raise ValueError(f"Unknown IC-LoRA model key: {required_lora_key}")
            keep_existing_ic_lora = (
                ic_lora_model_key is None
                and preset.distilled_lora_path
                and preset.distilled_lora_path != "None"
                and Path(preset.distilled_lora_path).exists()
            )
        else:
            required_lora_key = DEFAULT_DISTILLED_LORA_KEY
        required_lora_name = CHECKPOINTS[required_lora_key]["filename"]
        if not keep_existing_ic_lora and (
            not preset.distilled_lora_path
            or preset.distilled_lora_path == "None"
            or Path(preset.distilled_lora_path).name != required_lora_name
            or not Path(preset.distilled_lora_path).exists()
        ):
            existing = get_model_path(required_lora_key)
            if existing:
                preset.distilled_lora_path = str(existing)
            else:
                print(f"Auto-downloading required LoRA: {required_lora_key}")
                preset.distilled_lora_path = download_model_file(required_lora_key)

    if missing_gemma_files(preset.gemma_path):
        print(f"Auto-downloading Gemma text encoder: {GEMMA_REPO_ID}")
        preset.gemma_path = download_gemma_model(preset.gemma_path or "./models/gemma")
    
    return checkpoint_path, upsampler_path


# ============================================================================
# Pydantic Models for API
# ============================================================================

class ImageInput(BaseModel):
    """Single image input with frame index and strength."""
    image_base64: str = Field(..., description="Base64 encoded image")
    frame_index: int = Field(default=0, ge=0, description="Frame index where this image applies")
    strength: float = Field(default=1.0, ge=0.0, le=1.0, description="Conditioning strength")


class VideoConditioningInput(BaseModel):
    """Video conditioning input for IC-LoRA pipeline."""
    video_base64: str = Field(..., description="Base64 encoded video file")
    strength: float = Field(default=1.0, ge=0.0, le=1.0, description="Conditioning strength")


class GenerationRequest(BaseModel):
    """Request model for video generation."""
    preset_name: Optional[str] = Field(
        default=None,
        description="Preset name to use. If not provided, uses default preset."
    )
    
    # Required generation input
    prompt: str = Field(..., description="Prompt for video generation (required)")
    negative_prompt: Optional[str] = Field(default="", description="Negative prompt")
    
    # Pipeline settings (override preset values)
    pipeline_type: Optional[str] = Field(default=None, description="Override pipeline type")
    checkpoint_path: Optional[str] = Field(default=None, description="Override checkpoint path")
    spatial_upsampler_path: Optional[str] = Field(default=None, description="Override upsampler path")
    distilled_lora_path: Optional[str] = Field(default=None, description="Override distilled LoRA path")
    gemma_path: Optional[str] = Field(default=None, description="Override Gemma path")
    enable_fp8: Optional[bool] = Field(default=None, description="Override FP8 setting")
    skip_memory_cleanup: Optional[bool] = Field(default=None, description="Override skip memory cleanup setting")
    
    # Generation settings (override preset values)
    height: Optional[int] = Field(default=None, ge=256, le=2048, description="Override height")
    width: Optional[int] = Field(default=None, ge=256, le=2048, description="Override width")
    num_frames: Optional[int] = Field(default=None, ge=9, le=1201, description="Override frame count")
    frame_rate: Optional[float] = Field(default=None, ge=8, le=60, description="Override FPS")
    num_inference_steps: Optional[int] = Field(default=None, ge=4, le=100, description="Override steps")
    cfg_guidance_scale: Optional[float] = Field(default=None, ge=1.0, le=15.0, description="Override CFG")
    seed: Optional[int] = Field(default=None, description="Override seed (-1 for random)")
    image_strength: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Image strength")
    
    # Multiple images support (new format)
    images: Optional[List[ImageInput]] = Field(default=None, description="List of image inputs with frame indices")

    # Audio conditioning for A2V / lipsync-style generation
    audio_base64: Optional[str] = Field(default=None, description="Base64 encoded audio file")
    audio_filename: Optional[str] = Field(default=None, description="Original audio filename")
    audio_start_time: float = Field(default=0.0, ge=0.0, description="Audio start time in seconds")
    audio_max_duration: Optional[float] = Field(default=None, gt=0.0, description="Maximum audio duration in seconds")
    
    # Video conditioning for IC-LoRA pipeline
    video_conditioning: Optional[List[VideoConditioningInput]] = Field(
        default=None, 
        description="Video conditioning inputs for IC-LoRA (depth maps, pose, edges)"
    )
    ic_lora_model_key: Optional[str] = Field(default=None, description="CHECKPOINTS key for the IC-LoRA adapter")
    ic_lora_strength: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="IC-LoRA adapter strength")
    ic_lora_attention_strength: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Reference/control attention strength for IC-LoRA",
    )
    ic_lora_reference_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Reference video latent conditioning strength",
    )
    ic_lora_reference_image_base64: Optional[str] = Field(
        default=None,
        description="Base64 encoded still reference image to loop into an IC-LoRA reference video",
    )
    ic_lora_reference_image_filename: Optional[str] = Field(default=None, description="Original reference image filename")
    
    # Legacy single image support (backward compatible)
    input_image_base64: Optional[str] = Field(default=None, description="Base64 encoded input image (legacy)")


class PresetCreate(BaseModel):
    """Model for creating/updating a preset (no prompt - that's a generation input)."""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    
    # Pipeline settings
    pipeline_type: str = Field(default="distilled")
    checkpoint_path: Optional[str] = None
    distilled_lora_path: str = Field(default="None")
    spatial_upsampler_path: Optional[str] = None
    gemma_path: str = Field(default="./models/gemma")
    
    # Generation parameters (NO prompt/negative_prompt - those are generation inputs)
    height: int = Field(default=1024, ge=256, le=2048)
    width: int = Field(default=1536, ge=256, le=2048)
    num_frames: int = Field(default=121, ge=9, le=1201)
    frame_rate: float = Field(default=24.0, ge=8, le=60)
    num_inference_steps: int = Field(default=30, ge=4, le=100)
    cfg_guidance_scale: float = Field(default=3.0, ge=1.0, le=15.0)
    seed: int = Field(default=-1)
    enable_fp8: bool = Field(default=True)
    skip_memory_cleanup: bool = Field(default=True)
    image_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    lora_strength: float = Field(default=1.0, ge=0.0, le=2.0)


class PresetResponse(BaseModel):
    """Response model for preset data (no prompt - that's a generation input)."""
    name: str
    description: str
    is_default: bool
    created_at: str
    updated_at: str
    pipeline_type: str
    height: int
    width: int
    num_frames: int
    frame_rate: float
    num_inference_steps: int
    cfg_guidance_scale: float
    seed: int
    enable_fp8: bool
    skip_memory_cleanup: bool
    image_strength: float
    lora_strength: float = 1.0
    # Model paths
    checkpoint_path: Optional[str] = None
    spatial_upsampler_path: Optional[str] = None
    distilled_lora_path: Optional[str] = None
    gemma_path: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    cuda_available: bool
    gpu_name: Optional[str]
    gpu_memory_gb: Optional[float]
    pipeline_loaded: bool
    is_generating: bool


# ============================================================================
# Pipeline Manager (handles caching)
# ============================================================================

def _normalize_path(p: Optional[str]) -> Optional[str]:
    """Normalize a model path for reliable cache comparison.
    
    Resolves relative paths to absolute, normalizes slashes,
    and lowercases on Windows to prevent spurious cache misses.
    """
    if not p or p == "None":
        return p
    try:
        return str(Path(p).resolve())
    except Exception:
        return p


class PipelineManager:
    """
    Manages pipeline loading and caching.
    Keeps models in VRAM for faster subsequent generations.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._is_generating = False
        self._current_job_id: Optional[str] = None
        self._skip_memory_cleanup = False
        self._pipeline_cache = {
            "pipeline": None,
            "pipeline_type": None,
            "checkpoint_path": None,
            "spatial_upsampler_path": None,
            "gemma_path": None,
            "distilled_lora_path": None,
            "lora_strength": None,
            "enable_fp8": None,
        }
    
    def is_pipeline_loaded(self) -> bool:
        """Check if a pipeline is currently cached."""
        return self._pipeline_cache["pipeline"] is not None
    
    def is_generating(self) -> bool:
        """Check if currently generating."""
        return self._is_generating
    
    def try_start_generation(self, job_id: str) -> bool:
        """Mark a generation as active if no other job is running."""
        with self._lock:
            if self._is_generating:
                return False
            self._is_generating = True
            self._current_job_id = job_id
            return True

    def set_generating(self, is_generating: bool, job_id: Optional[str] = None):
        """Set the generating state."""
        with self._lock:
            self._is_generating = is_generating
            self._current_job_id = job_id

    def _reset_cache_metadata(self):
        self._pipeline_cache.update(
            {
                "pipeline": None,
                "pipeline_type": None,
                "checkpoint_path": None,
                "spatial_upsampler_path": None,
                "gemma_path": None,
                "distilled_lora_path": None,
                "lora_strength": None,
                "enable_fp8": None,
            }
        )

    def set_skip_memory_cleanup(self, skip: bool):
        """Update the skip_memory_cleanup flag."""
        self._skip_memory_cleanup = skip
        # Also sync the environment variable for vendor-level caching
        os.environ["LTX_KEEP_PIPELINE_MODELS"] = "1" if skip else "0"
        if skip:
            print("[PipelineManager] VRAM persistence ENABLED — models stay in VRAM between stages and generations")
        else:
            print("[PipelineManager] VRAM persistence DISABLED — models freed after each stage")

    def unload_cached_pipeline(self) -> tuple[bool, str]:
        """Manually unload the cached pipeline and release VRAM."""
        with self._lock:
            if self._is_generating:
                return False, "Cannot unload the pipeline while a generation is running."
            pipeline = self._pipeline_cache["pipeline"]
            if pipeline is None:
                return True, "No pipeline is currently loaded."
            # force=True bypasses LTX_KEEP_PIPELINE_MODELS for explicit unloads
            release_pipeline_models(pipeline, force=True)
            del pipeline
            self._reset_cache_metadata()
            return True, "Pipeline unloaded and VRAM cache cleared."
    
    def get_cached_pipeline(self, preset: GenerationPreset):
        """Get or create cached pipeline.
        
        Uses normalized paths for comparison to prevent spurious cache misses
        caused by relative vs absolute paths or different slash conventions.
        """
        cache = self._pipeline_cache
        
        # Normalize all paths for reliable comparison
        cache_valid = (
            cache["pipeline"] is not None
            and cache["pipeline_type"] == preset.pipeline_type
            and cache["checkpoint_path"] == _normalize_path(preset.checkpoint_path)
            and cache["spatial_upsampler_path"] == _normalize_path(preset.spatial_upsampler_path)
            and cache["gemma_path"] == _normalize_path(preset.gemma_path)
            and cache["distilled_lora_path"] == _normalize_path(preset.distilled_lora_path)
            and cache["lora_strength"] == float(getattr(preset, "lora_strength", 1.0))
            and cache["enable_fp8"] == preset.enable_fp8
        )
        
        if cache_valid:
            print("[PipelineManager] Cache HIT — reusing loaded pipeline")
            return cache["pipeline"], None
        
        # Clear old pipeline
        if cache["pipeline"] is not None:
            print("[PipelineManager] Cache MISS — pipeline settings changed, rebuilding")
            previous_pipeline_type = cache["pipeline_type"]
            pipeline_type_changed = previous_pipeline_type != preset.pipeline_type
            crosses_a2v_checkpoint = "a2vid_two_stage" in {previous_pipeline_type, preset.pipeline_type}
            should_empty_cache = False
            if pipeline_type_changed and crosses_a2v_checkpoint:
                print("[PipelineManager] A2V checkpoint boundary crossed - force-releasing previous pipeline")
                release_pipeline_models(cache["pipeline"], force=True)
                should_empty_cache = True
            elif pipeline_type_changed:
                print("[PipelineManager] Shared distilled pipeline switch - keeping resident model modules")
            elif not self._skip_memory_cleanup:
                release_pipeline_models(cache["pipeline"])
                should_empty_cache = True
            else:
                print("[PipelineManager] skip_memory_cleanup enabled — skipping release_pipeline_models")
            del cache["pipeline"]
            self._reset_cache_metadata()
            if should_empty_cache:
                torch.cuda.empty_cache()
        else:
            print("[PipelineManager] No cached pipeline — building fresh")
        
        try:
            gemma_ready, gemma_error = validate_gemma_root(preset.gemma_path)
            if not gemma_ready:
                return None, gemma_error

            if preset.pipeline_type in PIPELINES_REQUIRING_UPSAMPLER:
                if not preset.spatial_upsampler_path or not Path(preset.spatial_upsampler_path).exists():
                    return None, f"Spatial upsampler is required for {preset.pipeline_type} pipeline."

            if preset.pipeline_type in PIPELINES_REQUIRING_DISTILLED_LORA:
                if (
                    not preset.distilled_lora_path
                    or preset.distilled_lora_path == "None"
                    or not Path(preset.distilled_lora_path).exists()
                ):
                    return None, "This pipeline requires a LoRA. Download the required LTX-2.3 LoRA from Models."

            pipeline = create_pipeline(
                pipeline_type=preset.pipeline_type,
                checkpoint_path=preset.checkpoint_path,
                spatial_upsampler_path=preset.spatial_upsampler_path if preset.spatial_upsampler_path else None,
                gemma_path=preset.gemma_path,
                lora_path=preset.distilled_lora_path,
                enable_fp8=preset.enable_fp8,
                lora_strength=float(getattr(preset, "lora_strength", 1.0)),
            )
            
            # Cache the pipeline with normalized paths for reliable future comparisons
            cache["pipeline"] = pipeline
            cache["pipeline_type"] = preset.pipeline_type
            cache["checkpoint_path"] = _normalize_path(preset.checkpoint_path)
            cache["spatial_upsampler_path"] = _normalize_path(preset.spatial_upsampler_path)
            cache["gemma_path"] = _normalize_path(preset.gemma_path)
            cache["distilled_lora_path"] = _normalize_path(preset.distilled_lora_path)
            cache["lora_strength"] = float(getattr(preset, "lora_strength", 1.0))
            cache["enable_fp8"] = preset.enable_fp8
            print(f"[PipelineManager] Pipeline cached: type={preset.pipeline_type}, fp8={preset.enable_fp8}")
            
            return pipeline, None
            
        except ImportError as e:
            return None, f"LTX Pipelines not installed: {e}"
        except Exception as e:
            import traceback
            return None, f"Failed to load pipeline: {e}\n{traceback.format_exc()}"


# ============================================================================
# FastAPI Application
# ============================================================================

# Global pipeline manager
_pipeline_manager: Optional[PipelineManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize pipeline manager."""
    global _pipeline_manager
    
    # Startup
    ensure_directories()
    
    _pipeline_manager = PipelineManager()
    print("LTX-2 API started. Streaming generation enabled.")
    print(f"Models directory: {MODELS_DIR.absolute()}")
    print(f"Outputs directory: {OUTPUTS_DIR.absolute()}")
    print("Web UI: http://localhost:8000/")
    print("API Docs: http://localhost:8000/docs")
    
    yield
    
    # Shutdown
    if _pipeline_manager:
        _pipeline_manager.unload_cached_pipeline()
    print("LTX-2 API stopped.")


app = FastAPI(
    title="LTX-2.3 Video Generation API",
    description="RESTful API for AI video generation using LTX-2.3 models",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI."""
    html_path = TEMPLATES_DIR / "index.html"
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
        <html>
        <head><title>LTX-2.3 API</title></head>
        <body style="background: #0a0a0f; color: #f0f0f5; font-family: sans-serif; padding: 2rem;">
            <h1>🎬 LTX-2.3 Video Generation API</h1>
            <p>Web UI template not found. Please ensure templates/index.html exists.</p>
            <p><a href="/docs" style="color: #a855f7;">📚 View API Documentation</a></p>
        </body>
        </html>
        """)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_name = None
    gpu_memory = None
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    
    pipeline_loaded = _pipeline_manager.is_pipeline_loaded() if _pipeline_manager else False
    is_generating = _pipeline_manager.is_generating() if _pipeline_manager else False
    
    return HealthResponse(
        status="healthy",
        cuda_available=torch.cuda.is_available(),
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory,
        pipeline_loaded=pipeline_loaded,
        is_generating=is_generating,
    )


@app.post("/pipeline/unload")
async def unload_pipeline():
    """Unload the cached pipeline and free VRAM on demand."""
    if not _pipeline_manager:
        raise HTTPException(status_code=503, detail="Generation service not ready")

    success, message = _pipeline_manager.unload_cached_pipeline()
    if not success:
        raise HTTPException(status_code=409, detail=message)
    return {"message": message, "pipeline_loaded": _pipeline_manager.is_pipeline_loaded()}


# ============================================================================
# Model Endpoints
# ============================================================================

@app.get("/models")
async def list_models():
    """List all available models and their download status."""
    result = {}
    for key, info in CHECKPOINTS.items():
        status, path = check_model_status(key)
        result[key] = {
            "filename": info["filename"],
            "size": info["size"],
            "description": info["description"],
            "type": info["type"],
            "status": status,
            "path": path,
        }
    return result


@app.get("/models/{model_key}")
async def get_model_info(model_key: str):
    """Get information about a specific model."""
    if model_key not in CHECKPOINTS:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")
    
    info = CHECKPOINTS[model_key]
    status, path = check_model_status(model_key)
    
    return {
        "key": model_key,
        "filename": info["filename"],
        "size": info["size"],
        "description": info["description"],
        "type": info["type"],
        "status": status,
        "path": path,
    }


@app.post("/models/{model_key}/download")
async def download_model(model_key: str, background_tasks: BackgroundTasks):
    """Download a model from HuggingFace."""
    if model_key not in CHECKPOINTS:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")
    
    # Check if already downloaded
    status, path = check_model_status(model_key)
    if status == "ready":
        return {"message": f"Model '{model_key}' is already downloaded", "path": path}
    
    try:
        # Download synchronously (could be made async with background task for large files)
        downloaded_path = download_model_file(model_key)
        return {"message": f"Model '{model_key}' downloaded successfully", "path": downloaded_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


# ============================================================================
# Generation Endpoints (Streaming Only)
# ============================================================================


@app.post("/generate")
async def generate_video_stream(request: GenerationRequest):
    """
    Generate a video with real-time progress streaming via SSE.
    
    This endpoint streams progress updates during generation, eliminating
    the need for polling. Events include:
    - init: Job initialized with job_id
    - stage: Current generation stage (loading models, stage 1, stage 2, encoding)
    - step: Denoising step progress within a stage
    - complete: Generation finished with download URL
    - error: Generation failed with error message
    """
    if not _pipeline_manager:
        raise HTTPException(status_code=503, detail="Generation service not ready")
    
    # Get preset (optional - if not specified, use default as base for any missing values)
    preset_manager = get_preset_manager()
    if request.preset_name:
        preset = preset_manager.get_preset(request.preset_name)
        if not preset:
            raise HTTPException(status_code=404, detail=f"Preset '{request.preset_name}' not found")
    else:
        # Use default preset as base for any missing values
        preset = preset_manager.get_default_preset()
    
    # Handle input images (multiple images with frame indices)
    image_inputs = []
    video_conditioning_inputs = []  # List of (path, strength)
    audio_path = None
    temp_dir = OUTPUTS_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        from PIL import Image
        
        # Handle new multiple images format
        if request.images:
            for i, img_input in enumerate(request.images):
                image_data = base64.b64decode(img_input.image_base64)
                image = Image.open(io.BytesIO(image_data))
                image_path = str(temp_dir / f"input_{uuid.uuid4().hex[:8]}_{i}.png")
                image.save(image_path)
                image_inputs.append(make_image_conditioning(image_path, img_input.frame_index, img_input.strength))
        
        # Handle legacy single image format (backward compatible)
        elif request.input_image_base64:
            image_data = base64.b64decode(request.input_image_base64)
            image = Image.open(io.BytesIO(image_data))
            image_path = str(temp_dir / f"input_{uuid.uuid4().hex[:8]}.png")
            image.save(image_path)
            strength = request.image_strength if request.image_strength is not None else preset.image_strength
            image_inputs.append(make_image_conditioning(image_path, 0, strength))
        
        # Handle video conditioning for IC-LoRA
        if request.video_conditioning:
            for i, vid_input in enumerate(request.video_conditioning):
                video_data = base64.b64decode(vid_input.video_base64)
                video_path = str(temp_dir / f"video_cond_{uuid.uuid4().hex[:8]}_{i}.mp4")
                with open(video_path, 'wb') as f:
                    f.write(video_data)
                video_conditioning_inputs.append((video_path, vid_input.strength))

        if request.ic_lora_reference_image_base64:
            image_data = base64.b64decode(request.ic_lora_reference_image_base64)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            suffix = Path(request.ic_lora_reference_image_filename or "").suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
                suffix = ".png"
            image_path = str(temp_dir / f"ic_lora_ref_{uuid.uuid4().hex[:8]}{suffix}")
            image.save(image_path)
            target_width = request.width if request.width is not None else preset.width
            target_height = request.height if request.height is not None else preset.height
            target_frames = request.num_frames if request.num_frames is not None else preset.num_frames
            target_fps = request.frame_rate if request.frame_rate is not None else preset.frame_rate
            video_path = _create_static_reference_video(
                image_path=image_path,
                width=int(target_width),
                height=int(target_height),
                num_frames=int(target_frames),
                frame_rate=float(target_fps),
                temp_dir=temp_dir,
            )
            video_conditioning_inputs.append((video_path, float(request.ic_lora_reference_strength)))

        if request.audio_base64:
            audio_data = base64.b64decode(request.audio_base64)
            suffix = Path(request.audio_filename or "").suffix.lower()
            if suffix not in {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}:
                suffix = ".wav"
            audio_path = str(temp_dir / f"audio_{uuid.uuid4().hex[:8]}{suffix}")
            with open(audio_path, "wb") as f:
                f.write(audio_data)
                
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input media: {e}")
    
    job_id = str(uuid.uuid4())
    
    async def event_generator():
        """Generate SSE events during video generation."""
        progress_queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        
        def send_progress(data: dict):
            """Thread-safe callback to send progress events."""
            try:
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(progress_queue.put(data))
                )
            except Exception:
                pass  # Ignore if loop is closed
        
        # Send initial event
        yield f"data: {json.dumps({'type': 'init', 'job_id': job_id})}\n\n"

        if not _pipeline_manager.try_start_generation(job_id):
            yield f"data: {json.dumps({'type': 'error', 'message': 'Another generation is already running.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
        
        # Run generation in thread pool
        executor = ThreadPoolExecutor(max_workers=1)
        
        def run_generation():
            """Run the generation with progress callbacks."""
            try:
                set_progress_callback(send_progress)
                send_progress({"type": "stage", "stage": "loading", "message": "Loading models..."})
                
                output_path = _generate_video_with_progress(
                    job_id=job_id,
                    request=request,
                    preset=preset,
                    image_inputs=image_inputs,
                    video_conditioning_inputs=video_conditioning_inputs,
                    audio_path=audio_path,
                    progress_callback=send_progress,
                )
                
                send_progress({
                    "type": "complete",
                    "job_id": job_id,
                    "output_path": output_path,
                    "download_url": f"/generate/{job_id}/download"
                })
            except Exception as e:
                import traceback
                send_progress({
                    "type": "error",
                    "message": str(e),
                    "traceback": traceback.format_exc()
                })
            finally:
                set_progress_callback(None)
                _pipeline_manager.set_generating(False)
                send_progress({"type": "done"})
        
        # Start generation in background
        future = loop.run_in_executor(executor, run_generation)
        
        # Stream progress events
        while True:
            try:
                # Wait for progress with timeout
                data = await asyncio.wait_for(progress_queue.get(), timeout=60.0)
                yield f"data: {json.dumps(data)}\n\n"
                
                if data.get("type") in ("complete", "error", "done"):
                    break
            except asyncio.TimeoutError:
                # Send keepalive
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
            except Exception:
                break
        
        # Wait for future to complete
        try:
            await asyncio.wait_for(asyncio.wrap_future(future), timeout=5.0)
        except Exception:
            pass
        
        executor.shutdown(wait=False)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# Store completed streaming jobs for download
_streaming_jobs: Dict[str, str] = {}


def _generate_video_with_progress(
    job_id: str,
    request: GenerationRequest,
    preset: GenerationPreset,
    image_inputs: List[tuple],
    video_conditioning_inputs: List[tuple],
    audio_path: Optional[str],
    progress_callback,
) -> str:
    """Generate video with progress callbacks. Returns output path."""
    
    # Determine pipeline type and inference steps for stage detection
    pipeline_type = request.pipeline_type if request.pipeline_type else preset.pipeline_type
    num_inference_steps = request.num_inference_steps if request.num_inference_steps is not None else preset.num_inference_steps
    
    # Capture stdout/stderr to parse tqdm progress
    with OutputCaptureContext(progress_callback, pipeline_type, num_inference_steps):
        return _generate_video_internal(
            job_id=job_id,
            request=request,
            preset=preset,
            image_inputs=image_inputs,
            video_conditioning_inputs=video_conditioning_inputs,
            audio_path=audio_path,
            progress_callback=progress_callback,
        )


def _generate_video_internal(
    job_id: str,
    request: GenerationRequest,
    preset: GenerationPreset,
    image_inputs: List[tuple],
    video_conditioning_inputs: List[tuple],
    audio_path: Optional[str],
    progress_callback,
) -> str:
    """Internal generation function."""
    import random
    from PIL import Image
    
    # Apply request overrides to preset (API and UI can override any setting)
    if request.pipeline_type is not None:
        preset.pipeline_type = request.pipeline_type
    if request.checkpoint_path is not None:
        preset.checkpoint_path = request.checkpoint_path
    if request.spatial_upsampler_path is not None:
        preset.spatial_upsampler_path = request.spatial_upsampler_path
    if request.distilled_lora_path is not None:
        preset.distilled_lora_path = request.distilled_lora_path
    if request.gemma_path is not None:
        preset.gemma_path = request.gemma_path
    if request.enable_fp8 is not None:
        preset.enable_fp8 = request.enable_fp8
    if request.skip_memory_cleanup is not None:
        preset.skip_memory_cleanup = request.skip_memory_cleanup
    if request.ic_lora_strength is not None:
        preset.lora_strength = request.ic_lora_strength

    if preset.pipeline_type in LTX23_SPECIALIZED_PIPELINE_TYPES:
        raise ValueError(
            f"{preset.pipeline_type} is available in LTX-2.3 as a specialized CLI pipeline, "
            "but this WebUI does not have the required audio/source-video/HDR controls for it yet."
        )
    if preset.pipeline_type not in SUPPORTED_PIPELINE_TYPES:
        raise ValueError(f"Unknown pipeline type: {preset.pipeline_type}")
    
    # Auto-download required models if not present
    try:
        checkpoint_path, upsampler_path = auto_download_required_models(preset, request.ic_lora_model_key)
        preset.checkpoint_path = checkpoint_path
        preset.spatial_upsampler_path = upsampler_path
    except Exception as e:
        raise RuntimeError(f"Failed to download required models: {e}")
    
    # Prompt is now required in the request (not from preset)
    prompt = request.prompt
    negative_prompt = request.negative_prompt if request.negative_prompt else ""
    
    # Merge request overrides with preset for generation parameters
    height = request.height if request.height is not None else preset.height
    width = request.width if request.width is not None else preset.width
    num_frames = request.num_frames if request.num_frames is not None else preset.num_frames
    frame_rate = request.frame_rate if request.frame_rate is not None else preset.frame_rate
    num_inference_steps = request.num_inference_steps if request.num_inference_steps is not None else preset.num_inference_steps
    cfg_guidance_scale = request.cfg_guidance_scale if request.cfg_guidance_scale is not None else preset.cfg_guidance_scale
    seed = request.seed if request.seed is not None else preset.seed
    
    if not prompt:
        raise ValueError("Prompt is required")
    uses_distilled_portrait_lipsync = bool(
        preset.pipeline_type == "distilled" and audio_path and image_inputs
    )
    if preset.pipeline_type in {"a2vid_two_stage", "lipdub"} and not audio_path:
        raise ValueError(f"Audio input is required for the {preset.pipeline_type} pipeline.")
    if preset.pipeline_type == "lipdub" and not image_inputs:
        raise ValueError("A portrait image is required for lipsync generation.")
    if preset.pipeline_type == "distilled" and audio_path and not image_inputs:
        raise ValueError("Audio-only generation requires the a2vid_two_stage pipeline.")
    if preset.pipeline_type in {"a2vid_two_stage", "lipdub"} or uses_distilled_portrait_lipsync:
        _validate_audio_video_shape(int(width), int(height), int(num_frames))
    
    # Handle seed
    if seed is None or seed < 0:
        seed = random.randint(0, 2**32 - 1)
    seed = int(seed)
    
    # Set FP8 optimization
    if preset.enable_fp8:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Configure VRAM persistence (skip memory cleanup between stages)
    _pipeline_manager.set_skip_memory_cleanup(preset.skip_memory_cleanup)
    
    # Get pipeline from cache
    progress_callback({"type": "stage", "stage": "loading_pipeline", "message": "Loading pipeline..."})
    pipeline, error = _pipeline_manager.get_cached_pipeline(preset)
    if error:
        raise RuntimeError(error)
    
    # Use the provided ImageConditioningInput objects directly.
    images = image_inputs if image_inputs else []
    
    # Use video conditioning inputs for IC-LoRA (list of (path, strength))
    video_conditioning = video_conditioning_inputs if video_conditioning_inputs else []
    if preset.pipeline_type == "ic_lora" and not video_conditioning:
        raise ValueError("IC-LoRA generation requires a reference image, reference video, or control-signal video.")
    
    # Import utilities
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.utils.media_io import encode_video
    
    tiling_config = TilingConfig.default()
    if preset.pipeline_type in ONE_STAGE_PIPELINES:
        video_chunks_number = 1
    else:
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
    
    # Determine stage info based on pipeline type
    # Stage 2 always uses STAGE_2_DISTILLED_SIGMA_VALUES (4 values, tqdm uses sigmas[:-1] = 3 steps)
    if uses_distilled_portrait_lipsync:
        stage_1_steps = 8
        stage_2_steps = 3
    elif preset.pipeline_type in GUIDANCELESS_PIPELINES:
        stage_1_steps = 8  # Distilled and IC-LoRA use 8 fixed sigmas (DISTILLED_SIGMA_VALUES)
        stage_2_steps = 3  # Stage 2 distilled sigmas
    elif preset.pipeline_type in ONE_STAGE_PIPELINES:
        stage_1_steps = num_inference_steps
        stage_2_steps = 0  # One-stage has no Stage 2
    else:
        # a2vid_two_stage, ti2vid_two_stages, ti2vid_two_stages_hq, keyframe_interpolation
        stage_1_steps = num_inference_steps
        stage_2_steps = 3  # Stage 2 uses STAGE_2_DISTILLED_SIGMA_VALUES (3 steps)
    
    progress_callback({
        "type": "info",
        "pipeline": preset.pipeline_type,
        "stage_1_steps": stage_1_steps,
        "stage_2_steps": stage_2_steps,
        "num_frames": num_frames,
        "resolution": f"{width}x{height}"
    })
    
    # Generate
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUTS_DIR / f"ltx2_stream_{timestamp}_{job_id[:8]}.mp4"
    
    progress_callback({"type": "stage", "stage": "stage_1", "message": "Stage 1: Generating video..."})
    guidance_kwargs = build_guidance_kwargs(
        pipeline_type=preset.pipeline_type,
        checkpoint_path=preset.checkpoint_path,
        negative_prompt=negative_prompt if negative_prompt else "",
        num_inference_steps=int(num_inference_steps),
        cfg_guidance_scale=float(cfg_guidance_scale),
    )
    
    with torch.no_grad():
        if uses_distilled_portrait_lipsync:
            audio_max_duration = (
                float(request.audio_max_duration)
                if request.audio_max_duration is not None
                else int(num_frames) / float(frame_rate)
            )
            video, audio = _run_distilled_portrait_lipsync(
                pipeline,
                prompt=prompt,
                seed=seed,
                height=int(height),
                width=int(width),
                num_frames=int(num_frames),
                frame_rate=float(frame_rate),
                images=images,
                audio_path=audio_path,
                audio_start_time=float(request.audio_start_time),
                audio_max_duration=audio_max_duration,
                tiling_config=tiling_config,
            )
        elif preset.pipeline_type == "distilled":
            video, audio = pipeline(
                prompt=prompt,
                seed=seed,
                height=int(height),
                width=int(width),
                num_frames=int(num_frames),
                frame_rate=float(frame_rate),
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=False,
            )
        elif preset.pipeline_type == "ic_lora":
            video, audio = pipeline(
                prompt=prompt,
                seed=seed,
                height=int(height),
                width=int(width),
                num_frames=int(num_frames),
                frame_rate=float(frame_rate),
                images=images,
                video_conditioning=video_conditioning,
                tiling_config=tiling_config,
                conditioning_attention_strength=float(
                    request.ic_lora_attention_strength
                    if request.ic_lora_attention_strength is not None
                    else 1.0
                ),
                enhance_prompt=False,
            )
        elif preset.pipeline_type == "ti2vid_one_stage":
            video, audio = pipeline(
                prompt=prompt,
                seed=seed,
                height=int(height),
                width=int(width),
                num_frames=int(num_frames),
                frame_rate=float(frame_rate),
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=False,
                **guidance_kwargs,
            )
        elif preset.pipeline_type == "a2vid_two_stage":
            from ltx_pipelines.utils.constants import DISTILLED_SIGMAS

            audio_max_duration = (
                float(request.audio_max_duration)
                if request.audio_max_duration is not None
                else int(num_frames) / float(frame_rate)
            )
            video, audio = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                height=int(height),
                width=int(width),
                num_frames=int(num_frames),
                frame_rate=float(frame_rate),
                num_inference_steps=int(num_inference_steps),
                video_guider_params=guidance_kwargs["video_guider_params"],
                images=images,
                audio_path=audio_path,
                audio_start_time=float(request.audio_start_time),
                audio_max_duration=audio_max_duration,
                tiling_config=tiling_config,
                enhance_prompt=False,
                max_batch_size=1,
                stage_1_sigmas=DISTILLED_SIGMAS,
            )
        elif preset.pipeline_type == "lipdub":
            reference_video_path = _create_lipdub_reference_video(
                image_path=images[0].path,
                audio_path=audio_path,
                width=int(width),
                height=int(height),
                num_frames=int(num_frames),
                frame_rate=float(frame_rate),
                temp_dir=Path(audio_path).parent,
                audio_start_time=float(request.audio_start_time),
                audio_max_duration=request.audio_max_duration,
            )
            try:
                from ltx_pipelines.utils.media_io import get_videostream_metadata

                ref_meta = get_videostream_metadata(reference_video_path)
                ref_frames = max(9, ((int(ref_meta.frames) - 1) // 8) * 8 + 1)
                video_chunks_number = get_video_chunks_number(ref_frames, tiling_config)
            except Exception as exc:
                print(f"[lipdub] Could not inspect reference video metadata: {exc}")

            video, audio = pipeline(
                prompt=prompt,
                seed=seed,
                height=int(height),
                width=int(width),
                images=images[:1],
                reference_video_path=reference_video_path,
                reference_strength=1.0,
                tiling_config=tiling_config,
                enhance_prompt=False,
            )
        else:
            video, audio = pipeline(
                prompt=prompt,
                seed=seed,
                height=int(height),
                width=int(width),
                num_frames=int(num_frames),
                frame_rate=float(frame_rate),
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=False,
                **guidance_kwargs,
            )
        
        progress_callback({"type": "stage", "stage": "encoding", "message": "Encoding video..."})
        
        encode_video_compat(
            encode_video,
            video=video,
            fps=float(frame_rate),
            audio=audio,
            output_path=str(output_path),
            video_chunks_number=video_chunks_number,
        )
    
    # Store for download
    _streaming_jobs[job_id] = str(output_path)
    
    return str(output_path)


@app.get("/generate/{job_id}/download")
async def download_video(job_id: str):
    """Download the generated video."""
    output_path = _streaming_jobs.get(job_id)
    
    if not output_path:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found or not completed")
    
    if not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output video not found")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=Path(output_path).name,
    )


# ============================================================================
# Preset Endpoints
# ============================================================================

@app.get("/presets", response_model=List[PresetResponse])
async def list_presets():
    """List all presets."""
    preset_manager = get_preset_manager()
    default_name = preset_manager.get_default_preset_name()
    
    presets = []
    for name in preset_manager.list_presets():
        preset = preset_manager.get_preset(name)
        if preset:
            presets.append(PresetResponse(
                name=preset.name,
                description=preset.description,
                is_default=(name == default_name),
                created_at=preset.created_at,
                updated_at=preset.updated_at,
                pipeline_type=preset.pipeline_type,
                height=preset.height,
                width=preset.width,
                num_frames=preset.num_frames,
                frame_rate=preset.frame_rate,
                num_inference_steps=preset.num_inference_steps,
                cfg_guidance_scale=preset.cfg_guidance_scale,
                seed=preset.seed,
                enable_fp8=preset.enable_fp8,
                skip_memory_cleanup=preset.skip_memory_cleanup,
                image_strength=preset.image_strength,
                lora_strength=getattr(preset, "lora_strength", 1.0),
                checkpoint_path=preset.checkpoint_path,
                spatial_upsampler_path=preset.spatial_upsampler_path,
                distilled_lora_path=preset.distilled_lora_path,
                gemma_path=preset.gemma_path,
            ))
    
    return presets


@app.get("/presets/{preset_name}", response_model=PresetResponse)
async def get_preset(preset_name: str):
    """Get a specific preset."""
    preset_manager = get_preset_manager()
    preset = preset_manager.get_preset(preset_name)
    
    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")
    
    return PresetResponse(
        name=preset.name,
        description=preset.description,
        is_default=(preset_name == preset_manager.get_default_preset_name()),
        created_at=preset.created_at,
        updated_at=preset.updated_at,
        pipeline_type=preset.pipeline_type,
        height=preset.height,
        width=preset.width,
        num_frames=preset.num_frames,
        frame_rate=preset.frame_rate,
        num_inference_steps=preset.num_inference_steps,
        cfg_guidance_scale=preset.cfg_guidance_scale,
        seed=preset.seed,
        enable_fp8=preset.enable_fp8,
        skip_memory_cleanup=preset.skip_memory_cleanup,
        image_strength=preset.image_strength,
        lora_strength=getattr(preset, "lora_strength", 1.0),
        checkpoint_path=preset.checkpoint_path,
        spatial_upsampler_path=preset.spatial_upsampler_path,
        distilled_lora_path=preset.distilled_lora_path,
        gemma_path=preset.gemma_path,
    )


@app.post("/presets", response_model=PresetResponse)
async def create_preset(preset_data: PresetCreate):
    """Create a new preset."""
    preset_manager = get_preset_manager()
    
    # Check if preset exists
    if preset_manager.get_preset(preset_data.name):
        raise HTTPException(status_code=409, detail=f"Preset '{preset_data.name}' already exists")
    
    preset = preset_manager.create_preset_from_settings(
        name=preset_data.name,
        description=preset_data.description,
        pipeline_type=preset_data.pipeline_type,
        checkpoint_path=preset_data.checkpoint_path or "",
        distilled_lora_path=preset_data.distilled_lora_path,
        spatial_upsampler_path=preset_data.spatial_upsampler_path or "",
        gemma_path=preset_data.gemma_path,
        height=preset_data.height,
        width=preset_data.width,
        num_frames=preset_data.num_frames,
        frame_rate=preset_data.frame_rate,
        num_inference_steps=preset_data.num_inference_steps,
        cfg_guidance_scale=preset_data.cfg_guidance_scale,
        seed=preset_data.seed,
        enable_fp8=preset_data.enable_fp8,
        skip_memory_cleanup=preset_data.skip_memory_cleanup,
        image_strength=preset_data.image_strength,
        lora_strength=preset_data.lora_strength,
    )
    
    preset_manager.save_preset(preset)
    
    return PresetResponse(
        name=preset.name,
        description=preset.description,
        is_default=False,
        created_at=preset.created_at,
        updated_at=preset.updated_at,
        pipeline_type=preset.pipeline_type,
        height=preset.height,
        width=preset.width,
        num_frames=preset.num_frames,
        frame_rate=preset.frame_rate,
        num_inference_steps=preset.num_inference_steps,
        cfg_guidance_scale=preset.cfg_guidance_scale,
        seed=preset.seed,
        enable_fp8=preset.enable_fp8,
        skip_memory_cleanup=preset.skip_memory_cleanup,
        image_strength=preset.image_strength,
        lora_strength=getattr(preset, "lora_strength", 1.0),
        checkpoint_path=preset.checkpoint_path,
        spatial_upsampler_path=preset.spatial_upsampler_path,
        distilled_lora_path=preset.distilled_lora_path,
        gemma_path=preset.gemma_path,
    )


@app.put("/presets/{preset_name}", response_model=PresetResponse)
async def update_preset(preset_name: str, preset_data: PresetCreate):
    """Update an existing preset."""
    preset_manager = get_preset_manager()
    
    existing = preset_manager.get_preset(preset_name)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")
    
    # Create updated preset (keep original name if data.name differs)
    preset = preset_manager.create_preset_from_settings(
        name=preset_name,  # Keep original name
        description=preset_data.description,
        pipeline_type=preset_data.pipeline_type,
        checkpoint_path=preset_data.checkpoint_path or existing.checkpoint_path,
        distilled_lora_path=preset_data.distilled_lora_path,
        spatial_upsampler_path=preset_data.spatial_upsampler_path or existing.spatial_upsampler_path,
        gemma_path=preset_data.gemma_path,
        height=preset_data.height,
        width=preset_data.width,
        num_frames=preset_data.num_frames,
        frame_rate=preset_data.frame_rate,
        num_inference_steps=preset_data.num_inference_steps,
        cfg_guidance_scale=preset_data.cfg_guidance_scale,
        seed=preset_data.seed,
        enable_fp8=preset_data.enable_fp8,
        skip_memory_cleanup=preset_data.skip_memory_cleanup,
        image_strength=preset_data.image_strength,
        lora_strength=preset_data.lora_strength,
    )
    preset.created_at = existing.created_at
    
    preset_manager.save_preset(preset, overwrite=True)
    
    return PresetResponse(
        name=preset.name,
        description=preset.description,
        is_default=(preset_name == preset_manager.get_default_preset_name()),
        created_at=preset.created_at,
        updated_at=preset.updated_at,
        pipeline_type=preset.pipeline_type,
        height=preset.height,
        width=preset.width,
        num_frames=preset.num_frames,
        frame_rate=preset.frame_rate,
        num_inference_steps=preset.num_inference_steps,
        cfg_guidance_scale=preset.cfg_guidance_scale,
        seed=preset.seed,
        enable_fp8=preset.enable_fp8,
        skip_memory_cleanup=preset.skip_memory_cleanup,
        image_strength=preset.image_strength,
        lora_strength=getattr(preset, "lora_strength", 1.0),
        checkpoint_path=preset.checkpoint_path,
        spatial_upsampler_path=preset.spatial_upsampler_path,
        distilled_lora_path=preset.distilled_lora_path,
        gemma_path=preset.gemma_path,
    )


@app.delete("/presets/{preset_name}")
async def delete_preset(preset_name: str):
    """Delete a preset. Cannot delete the 'default' preset."""
    preset_manager = get_preset_manager()
    
    if preset_name == DEFAULT_PRESET_NAME:
        raise HTTPException(status_code=400, detail="Cannot delete the default preset")
    
    if not preset_manager.get_preset(preset_name):
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")
    
    preset_manager.delete_preset(preset_name)
    return {"message": f"Preset '{preset_name}' deleted"}


@app.post("/presets/{preset_name}/set-default")
async def set_default_preset(preset_name: str):
    """Set a preset as the default."""
    preset_manager = get_preset_manager()
    
    if not preset_manager.get_preset(preset_name):
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")
    
    preset_manager.set_default(preset_name)
    return {"message": f"Preset '{preset_name}' is now the default"}


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
