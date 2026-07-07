"""
LTX-2 WebUI - Video Generation Interface
A beautiful web interface for Lightricks LTX-2 video generation models.

LTX-2 supports:
- Synchronized audio-video generation
- Native 4K resolution at up to 50 FPS
- Clips up to 20 seconds long
- Text-to-video, image-to-video, video-to-video, keyframe interpolation

Requires: Python >= 3.12, CUDA >= 12.7, PyTorch ~= 2.7
"""

import os
import sys
import json
import time
import torch
import gradio as gr
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from PIL import Image
import tempfile
import shutil

from presets import get_preset_manager, GenerationPreset, DEFAULT_PRESET_NAME
from ltx2_compat import (
    CHECKPOINTS,
    DEFAULT_CHECKPOINT_CANDIDATES,
    DEFAULT_CHECKPOINT_KEY,
    DEFAULT_UPSAMPLER_CANDIDATES,
    DEFAULT_UPSAMPLER_KEY,
    HF_REPO_ID,
    ONE_STAGE_PIPELINES,
    PIPELINES_REQUIRING_DISTILLED_LORA,
    PIPELINES_REQUIRING_UPSAMPLER,
    build_guidance_kwargs,
    create_pipeline,
    encode_video_compat,
    is_ltx23_model_path,
    make_image_conditioning,
    model_repo_id,
    release_pipeline_models,
    validate_gemma_root,
)

# Constants
MODELS_DIR = Path("./models")
OUTPUTS_DIR = Path("./outputs")

# Pipeline types - based on README Pipeline Selection Guide
# Decision Tree:
# - Text-to-video only:
#   - Fastest inference → DistilledPipeline (8 sigmas, no CFG)
#   - Best quality → TI2VidTwoStagesPipeline (production recommended)
# - Image/Video conditioning:
#   - Reference videos → ICLoraPipeline
#   - Keyframe interpolation → KeyframeInterpolationPipeline
#   - Image-to-video → Any pipeline supports this
# Note: TI2VidOneStagePipeline is primarily for educational purposes
PIPELINE_TYPES = {
    "distilled": {
        "name": "⚡ Distilled Pipeline (Fastest)",
        "description": "🚀 Fastest inference with 8 predefined sigmas, no CFG needed. Best for: quick iterations, batch processing. Uses distilled checkpoint.",
        "recommended": True,
        "requires": ["distilled checkpoint", "spatial_upsampler", "gemma"],
        "features": {"stages": 2, "cfg": False, "upsampling": True, "conditioning": "Image"}
    },
    "ti2vid_two_stages": {
        "name": "🎬 Two-Stage Pipeline (Best Quality)",
        "description": "Production quality - Stage 1 with CFG guidance, Stage 2 upsamples 2x with distilled LoRA refinement. Best for: final renders, highest quality.",
        "recommended": False,
        "requires": ["checkpoint", "distilled_lora", "spatial_upsampler", "gemma"],
        "features": {"stages": 2, "cfg": True, "upsampling": True, "conditioning": "Image"}
    },
    "ti2vid_two_stages_hq": {
        "name": "🎞️ Two-Stage HQ Pipeline",
        "description": "LTX-2.3 HQ two-stage mode using the res_2s sampler. Defaults to fewer steps and is tuned for high-quality 16:9 output.",
        "recommended": False,
        "requires": ["checkpoint", "distilled_lora", "spatial_upsampler", "gemma"],
        "features": {"stages": 2, "cfg": True, "upsampling": True, "conditioning": "Image"}
    },
    "ti2vid_one_stage": {
        "name": "📚 One-Stage Pipeline (Educational)",
        "description": "⚠️ For learning/prototyping only. Single stage, no upsampling, lower resolution (512×768). NOT recommended for production.",
        "recommended": False,
        "requires": ["checkpoint", "gemma"],
        "features": {"stages": 1, "cfg": True, "upsampling": False, "conditioning": "Image"}
    },
    "ic_lora": {
        "name": "🎞️ IC-LoRA Pipeline (Video-to-Video)",
        "description": "Video-to-video transformations with reference video/image conditioning. Best for: style transfer, pose/depth control, video editing.",
        "recommended": False,
        "requires": ["checkpoint", "ic_lora", "spatial_upsampler", "gemma"],
        "features": {"stages": 2, "cfg": False, "upsampling": True, "conditioning": "Image + Video"}
    },
    "keyframe_interpolation": {
        "name": "🎨 Keyframe Interpolation Pipeline",
        "description": "Interpolate between keyframe images for smooth animations. Uses guiding latents for smoother transitions. Best for: animation, motion graphics.",
        "recommended": False,
        "requires": ["checkpoint", "distilled_lora", "spatial_upsampler", "gemma"],
        "features": {"stages": 2, "cfg": True, "upsampling": True, "conditioning": "Keyframes"}
    },
    "a2vid_two_stage": {
        "name": "Audio-to-Video Two-Stage Pipeline",
        "description": "LTX-2.3 specialized audio-to-video pipeline. CLI-oriented in this WebUI until audio file controls are added.",
        "recommended": False,
        "requires": ["checkpoint", "distilled_lora", "spatial_upsampler", "gemma", "audio input"],
        "features": {"stages": 2, "cfg": True, "upsampling": True, "conditioning": "Audio + Image"}
    },
    "retake": {
        "name": "Retake Pipeline",
        "description": "LTX-2.3 specialized video retake pipeline for regenerating a time region of an existing clip. CLI-oriented in this WebUI.",
        "recommended": False,
        "requires": ["checkpoint", "gemma", "source video", "time range"],
        "features": {"stages": 1, "cfg": True, "upsampling": False, "conditioning": "Video Region"}
    },
    "hdr_ic_lora": {
        "name": "HDR IC-LoRA Pipeline",
        "description": "LTX-2.3 specialized HDR video-to-video pipeline for linear float/EXR workflows. CLI-oriented in this WebUI.",
        "recommended": False,
        "requires": ["distilled checkpoint", "spatial_upsampler", "HDR IC-LoRA", "text embeddings", "source video"],
        "features": {"stages": 2, "cfg": False, "upsampling": True, "conditioning": "HDR Video"}
    },
    "lipdub": {
        "name": "LipDub Pipeline",
        "description": "LTX-2.3 specialized lip dubbing pipeline with reference video/audio conditioning. CLI-oriented in this WebUI.",
        "recommended": False,
        "requires": ["distilled checkpoint", "spatial_upsampler", "LipDub IC-LoRA", "gemma", "reference video"],
        "features": {"stages": 2, "cfg": True, "upsampling": True, "conditioning": "Video + Audio"}
    },
}

# Custom CSS for a stunning dark theme
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --primary-hue: 265;
    --accent-hue: 340;
    --bg-dark: #0a0a0f;
    --bg-card: #12121a;
    --bg-hover: #1a1a25;
    --border-color: #2a2a3a;
    --text-primary: #f0f0f5;
    --text-secondary: #9090a5;
    --accent-purple: #a855f7;
    --accent-pink: #ec4899;
    --accent-gradient: linear-gradient(135deg, #a855f7 0%, #ec4899 50%, #f97316 100%);
    --glow-purple: 0 0 30px rgba(168, 85, 247, 0.3);
    --glow-pink: 0 0 30px rgba(236, 72, 153, 0.3);
}

body, .gradio-container {
    background: var(--bg-dark) !important;
    font-family: 'Outfit', sans-serif !important;
}

.gradio-container {
    max-width: 1400px !important;
}

/* Header styling */
.header-container {
    text-align: center;
    padding: 2rem 0;
    margin-bottom: 1rem;
    position: relative;
}

.header-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 80%;
    height: 100%;
    background: radial-gradient(ellipse at center, rgba(168, 85, 247, 0.15) 0%, transparent 70%);
    pointer-events: none;
}

.header-title {
    font-size: 3.5rem;
    font-weight: 700;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.02em;
}

.header-subtitle {
    font-size: 1.1rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
}

/* Card styling */
.gr-panel, .gr-box, .gr-form {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
}

/* Tab styling */
.tab-nav {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
    border: 1px solid var(--border-color) !important;
}

.tab-nav button {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

.tab-nav button.selected {
    background: var(--accent-gradient) !important;
    color: white !important;
}

/* Input styling */
input, textarea, select {
    background: var(--bg-dark) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Outfit', sans-serif !important;
    transition: all 0.3s ease !important;
}

input:focus, textarea:focus, select:focus {
    border-color: var(--accent-purple) !important;
    box-shadow: var(--glow-purple) !important;
}

/* Button styling */
.gr-button {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.gr-button-primary {
    background: var(--accent-gradient) !important;
    border: none !important;
    color: white !important;
}

.gr-button-primary:hover {
    box-shadow: var(--glow-purple), var(--glow-pink) !important;
    transform: translateY(-2px);
}

.gr-button-secondary {
    background: var(--bg-hover) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
}

/* Slider styling */
.gr-slider input[type="range"] {
    accent-color: var(--accent-purple) !important;
}

/* Label styling */
label {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}

/* Accordion styling */
.gr-accordion {
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* Progress bar */
.progress-bar {
    background: var(--accent-gradient) !important;
}

/* Model card styling */
.model-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
}

.model-card:hover {
    border-color: var(--accent-purple);
    box-shadow: var(--glow-purple);
}

/* Status indicators */
.status-ready {
    color: #22c55e;
}

.status-downloading {
    color: #f59e0b;
}

.status-missing {
    color: #ef4444;
}

/* Code blocks */
code {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--bg-dark) !important;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    font-size: 0.9em;
}

/* Video output */
video {
    border-radius: 12px !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5) !important;
}

/* Gallery */
.gr-gallery {
    border-radius: 12px !important;
    overflow: hidden;
}

/* Markdown */
.gr-markdown {
    color: var(--text-secondary) !important;
}

.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    color: var(--text-primary) !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-dark);
}

::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent-purple);
}

/* Animation */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(168, 85, 247, 0.3); }
    50% { box-shadow: 0 0 40px rgba(236, 72, 153, 0.5); }
}

.generating {
    animation: pulse-glow 2s ease-in-out infinite;
}
"""


def ensure_directories():
    """Create necessary directories."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "checkpoints").mkdir(exist_ok=True)
    (MODELS_DIR / "loras").mkdir(exist_ok=True)
    (MODELS_DIR / "upsamplers").mkdir(exist_ok=True)
    (MODELS_DIR / "gemma").mkdir(exist_ok=True)


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


def check_model_status(model_key: str) -> Tuple[str, str]:
    """Check if a model is downloaded. Returns (status, status_text)."""
    path = get_model_path(model_key)
    if path and path.exists():
        size = path.stat().st_size / (1024 ** 3)  # GB
        return "ready", f"✅ Downloaded ({size:.2f} GB)"
    return "missing", "❌ Not downloaded"


def download_model(model_key: str, progress=gr.Progress()) -> str:
    """Download a model from HuggingFace."""
    if model_key not in CHECKPOINTS:
        return f"❌ Unknown model: {model_key}"
    
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
        return f"✅ Model already exists: {target_path}"
    
    try:
        progress(0, desc=f"Downloading {filename}...")
        
        downloaded_path = hf_hub_download(
            repo_id=model_repo_id(model_key),
            filename=filename,
            local_dir=target_dir,
        )
        
        progress(1, desc="Download complete!")
        return f"✅ Successfully downloaded to: {downloaded_path}\n\n📋 Click 'Refresh Model Lists' in the Generate tab to see the new model."
    
    except Exception as e:
        return f"❌ Download failed: {str(e)}"


def get_available_models() -> dict:
    """Get status of all available models."""
    statuses = {}
    for key in CHECKPOINTS:
        status, text = check_model_status(key)
        statuses[key] = {
            "status": status,
            "text": text,
            "info": CHECKPOINTS[key]
        }
    return statuses


def refresh_model_status() -> str:
    """Generate HTML for model status display."""
    statuses = get_available_models()
    
    html = "<div style='display: grid; gap: 0.75rem;'>"
    
    for key, data in statuses.items():
        info = data["info"]
        status_class = "status-ready" if data["status"] == "ready" else "status-missing"
        
        html += f"""
        <div class="model-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong style="color: var(--text-primary);">{key}</strong>
                    <span style="color: var(--text-secondary); font-size: 0.9em;"> ({info['size']})</span>
                </div>
                <span class="{status_class}" style="font-size: 0.9em;">{data['text']}</span>
            </div>
            <div style="color: var(--text-secondary); font-size: 0.85em; margin-top: 0.25rem;">
                {info['description']}
            </div>
        </div>
        """
    
    html += "</div>"
    return html


def get_checkpoint_choices() -> List[str]:
    """Get list of available checkpoint files."""
    choices = []
    checkpoint_dir = MODELS_DIR / "checkpoints"
    if checkpoint_dir.exists():
        for f in checkpoint_dir.glob("*.safetensors"):
            if is_ltx23_model_path(f):
                choices.append(str(f))
    # Sort with LTX-2.3 distilled first, then the LTX-2.3 dev checkpoint.
    def sort_key(x):
        x_lower = x.lower()
        if "ltx-2.3-22b-distilled" in x_lower:
            return (0, x)
        elif "ltx-2.3-22b-dev" in x_lower:
            return (1, x)
        else:
            return (2, x)
    choices.sort(key=sort_key)
    return choices if choices else ["No checkpoints found - download from Models tab"]


def get_default_checkpoint() -> str:
    """Get the default checkpoint path, preferring LTX-2.3 distilled."""
    choices = get_checkpoint_choices()
    if choices and "No checkpoints" not in choices[0]:
        return choices[0]
    return str(MODELS_DIR / "checkpoints" / DEFAULT_CHECKPOINT_CANDIDATES[0])


def get_default_upsampler() -> str:
    """Get the default spatial upsampler path, preferring LTX-2.3."""
    choices = get_upscaler_choices()
    if choices and "No upsamplers" not in choices[0]:
        return choices[0]
    return str(MODELS_DIR / "upsamplers" / DEFAULT_UPSAMPLER_CANDIDATES[0])


def get_lora_choices() -> List[str]:
    """Get list of available LoRA files."""
    choices = ["None"]
    lora_dir = MODELS_DIR / "loras"
    if lora_dir.exists():
        for f in lora_dir.glob("*.safetensors"):
            if is_ltx23_model_path(f):
                choices.append(str(f))
    return choices


def get_upscaler_choices() -> List[str]:
    """Get list of available upscaler files."""
    choices = []
    upscaler_dir = MODELS_DIR / "upsamplers"
    if upscaler_dir.exists():
        for f in upscaler_dir.glob("*.safetensors"):
            if is_ltx23_model_path(f):
                choices.append(str(f))
    choices.sort(key=lambda value: (0 if "ltx-2.3-spatial-upscaler-x2" in value.lower() else 1, value))
    return choices if choices else ["No upsamplers found - download from Models tab"]


# Global pipeline cache - keeps the active LTX pipeline in VRAM between generations.
# Some pipeline versions also expose per-stage caches; clear them when present.
_pipeline_cache = {
    "pipeline": None,
    "pipeline_type": None,
    "checkpoint_path": None,
    "spatial_upsampler_path": None,
    "gemma_path": None,
    "distilled_lora_path": None,
    "enable_fp8": None,
}


def clear_vram_cache() -> str:
    """Clear all cached models from VRAM."""
    global _pipeline_cache
    
    try:
        # Get VRAM usage before clearing
        if torch.cuda.is_available():
            vram_before = torch.cuda.memory_allocated() / (1024 ** 3)
        else:
            vram_before = 0
        
        # Clear the pipeline cache
        if _pipeline_cache["pipeline"] is not None:
            release_pipeline_models(_pipeline_cache["pipeline"], force=True)
        
        # Clear the pipeline reference
        _pipeline_cache["pipeline"] = None
        _pipeline_cache["pipeline_type"] = None
        _pipeline_cache["checkpoint_path"] = None
        _pipeline_cache["spatial_upsampler_path"] = None
        _pipeline_cache["gemma_path"] = None
        _pipeline_cache["distilled_lora_path"] = None
        _pipeline_cache["enable_fp8"] = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            vram_after = torch.cuda.memory_allocated() / (1024 ** 3)
            freed = vram_before - vram_after
            return f"✅ VRAM cache cleared!\n\n📊 Freed: {freed:.2f} GB\n💾 Current usage: {vram_after:.2f} GB"
        else:
            return "✅ Cache cleared (no CUDA device detected)"
            
    except Exception as e:
        return f"❌ Error clearing cache: {str(e)}"


def get_vram_status() -> str:
    """Get current VRAM usage status."""
    if not torch.cuda.is_available():
        return "No CUDA device detected"
    
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    
    # Check if pipeline is cached
    pipeline_status = "🟢 Pipeline cached" if _pipeline_cache["pipeline"] is not None else "⚪ No pipeline loaded"
    
    return f"""**VRAM Status:**
- Allocated: {allocated:.2f} GB
- Reserved: {reserved:.2f} GB  
- Total: {total:.1f} GB
- {pipeline_status}"""


def get_cached_pipeline(
    pipeline_type: str,
    checkpoint_path: str,
    spatial_upsampler_path: str,
    gemma_path: str,
    distilled_lora_path: str,
    enable_fp8: bool,
    progress=gr.Progress()
):
    """
    Get or create a cached pipeline. Keeps models in VRAM for faster subsequent generations.
    Only recreates pipeline if configuration changes.
    """
    global _pipeline_cache
    
    # Check if we can reuse the cached pipeline
    cache_valid = (
        _pipeline_cache["pipeline"] is not None
        and _pipeline_cache["pipeline_type"] == pipeline_type
        and _pipeline_cache["checkpoint_path"] == checkpoint_path
        and _pipeline_cache["spatial_upsampler_path"] == spatial_upsampler_path
        and _pipeline_cache["gemma_path"] == gemma_path
        and _pipeline_cache["distilled_lora_path"] == distilled_lora_path
        and _pipeline_cache["enable_fp8"] == enable_fp8
    )
    
    if cache_valid:
        progress(0.1, desc="Using cached pipeline (models already in VRAM)...")
        return _pipeline_cache["pipeline"], None
    
    # Clear old pipeline to free VRAM before loading new one
    if _pipeline_cache["pipeline"] is not None:
        progress(0.1, desc="Clearing old pipeline from VRAM...")
        release_pipeline_models(_pipeline_cache["pipeline"])
        del _pipeline_cache["pipeline"]
        _pipeline_cache["pipeline"] = None
        # Only empty cache if memory cleanup is enabled
        keep = os.environ.get("LTX_KEEP_PIPELINE_MODELS", "").lower()
        if keep not in {"1", "true", "yes", "on"}:
            torch.cuda.empty_cache()
    
    # Create new pipeline
    progress(0.15, desc=f"Loading {pipeline_type} pipeline (first run is slower)...")
    
    try:
        gemma_ready, gemma_error = validate_gemma_root(gemma_path)
        if not gemma_ready:
            return None, gemma_error

        if pipeline_type in PIPELINES_REQUIRING_UPSAMPLER:
            if not spatial_upsampler_path or not Path(spatial_upsampler_path).exists():
                return None, f"鉂?Spatial upsampler is required for {pipeline_type} pipeline.\n\nPlease download from the Models tab."

        if pipeline_type in PIPELINES_REQUIRING_DISTILLED_LORA:
            if not distilled_lora_path or distilled_lora_path == "None" or not Path(distilled_lora_path).exists():
                return None, "鉂?This pipeline requires a distilled LoRA.\n\nPlease download the LTX-2.3 distilled LoRA from the Models tab."

        pipeline = create_pipeline(
            pipeline_type=pipeline_type,
            checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path if spatial_upsampler_path else None,
            gemma_path=gemma_path,
            lora_path=distilled_lora_path,
            enable_fp8=enable_fp8,
        )

        _pipeline_cache["pipeline"] = pipeline
        _pipeline_cache["pipeline_type"] = pipeline_type
        _pipeline_cache["checkpoint_path"] = checkpoint_path
        _pipeline_cache["spatial_upsampler_path"] = spatial_upsampler_path
        _pipeline_cache["gemma_path"] = gemma_path
        _pipeline_cache["distilled_lora_path"] = distilled_lora_path
        _pipeline_cache["enable_fp8"] = enable_fp8

        return pipeline, None
        
    except ImportError as e:
        return None, f"""❌ LTX Pipelines not installed.

Please install from the LTX-2 repository:
```
cd LTX-2
pip install -e packages/ltx-core
pip install -e packages/ltx-pipelines
```

Error: {str(e)}"""
    except Exception as e:
        import traceback
        return None, f"❌ Failed to load pipeline:\n{str(e)}\n{traceback.format_exc()}"


def generate_video(
    pipeline_type: str,
    checkpoint_path: str,
    distilled_lora_path: str,
    spatial_upsampler_path: str,
    gemma_path: str,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    num_inference_steps: int,
    cfg_guidance_scale: float,
    seed: int,
    enable_fp8: bool,
    skip_memory_cleanup: bool,
    input_image: Optional[Image.Image],
    image_strength: float,
    reference_video: Optional[str],
    keyframe_images: Optional[List[Image.Image]],
    progress=gr.Progress()
) -> Tuple[Optional[str], str]:
    """Generate video using the selected pipeline. Keeps models in VRAM for faster subsequent runs."""
    
    # Validate inputs
    if not prompt:
        return None, "❌ Please enter a prompt"
    
    if not checkpoint_path or "No checkpoints" in checkpoint_path:
        return None, "❌ Please download and select a checkpoint from the Models tab"
    
    if not Path(checkpoint_path).exists():
        return None, f"❌ Checkpoint not found: {checkpoint_path}"
    
    # Check Gemma path
    gemma_ready, gemma_error = validate_gemma_root(gemma_path)
    if not gemma_ready:
        return None, f"Gemma text encoder not configured.\n\n{gemma_error}"
    
    # Generate output filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUTS_DIR / f"ltx2_{timestamp}.mp4"
    
    # Handle seed - generate random if -1 or None
    import random
    if seed is None or seed < 0:
        seed = random.randint(0, 2**32 - 1)
    seed = int(seed)  # Ensure it's an integer
    
    # Set environment variable for FP8 optimization
    if enable_fp8:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Set Skip Memory Cleanup environment variable
    os.environ["LTX_KEEP_PIPELINE_MODELS"] = "1" if skip_memory_cleanup else "0"
    
    try:
        # Get or create cached pipeline (keeps models in VRAM)
        pipeline, error = get_cached_pipeline(
            pipeline_type=pipeline_type,
            checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_path=gemma_path,
            distilled_lora_path=distilled_lora_path,
            enable_fp8=enable_fp8,
            progress=progress,
        )
        
        if error:
            return None, error
        
        progress(0.3, desc="Generating video...")
        
        # Prepare image conditioning
        images = []
        if input_image is not None:
            temp_img_path = OUTPUTS_DIR / f"temp_input_{timestamp}.png"
            input_image.save(temp_img_path)
            images = [make_image_conditioning(temp_img_path, 0, image_strength)]
        
        # Prepare keyframes for keyframe_interpolation pipeline
        if pipeline_type == "keyframe_interpolation" and keyframe_images:
            images = []  # Replace with keyframes
            for i, img in enumerate(keyframe_images):
                temp_path = OUTPUTS_DIR / f"temp_keyframe_{timestamp}_{i}.png"
                img.save(temp_path)
                frame_idx = int(i * (num_frames - 1) / max(1, len(keyframe_images) - 1))
                images.append(make_image_conditioning(temp_path, frame_idx, 1.0))
        
        # Import utilities needed for video encoding
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.media_io import encode_video
        
        # TilingConfig and video_chunks_number for two-stage pipelines
        # One-stage uses video_chunks_number=1 (no chunking)
        tiling_config = TilingConfig.default()
        if pipeline_type in ONE_STAGE_PIPELINES:
            video_chunks_number = 1
        else:
            video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
        guidance_kwargs = build_guidance_kwargs(
            pipeline_type=pipeline_type,
            checkpoint_path=checkpoint_path,
            negative_prompt=negative_prompt if negative_prompt else "",
            num_inference_steps=int(num_inference_steps),
            cfg_guidance_scale=float(cfg_guidance_scale),
        )
        
        # Generate video using the cached pipeline
        # Each pipeline has a different API - match the source code exactly
        with torch.no_grad():
            if pipeline_type == "distilled":
                # DistilledPipeline: no CFG, no negative prompt, has tiling_config
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
            elif pipeline_type == "ic_lora":
                # ICLoraPipeline: video conditioning support, has tiling_config
                video_conditioning = []
                if reference_video and Path(reference_video).exists():
                    video_conditioning = [(reference_video, 1.0)]
                
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
                    enhance_prompt=False,
                )
            elif pipeline_type == "ti2vid_one_stage":
                # TI2VidOneStagePipeline: CFG + negative prompt
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
            else:
                # TI2VidTwoStagesPipeline, KeyframeInterpolationPipeline: CFG + negative prompt, has tiling_config
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
            
            # Encode and save video
            progress(0.9, desc="Encoding video...")
            encode_video_compat(
                encode_video,
                video=video,
                fps=float(frame_rate),
                audio=audio,
                output_path=str(output_path),
                video_chunks_number=video_chunks_number,
            )
        
        progress(1.0, desc="Complete!")
        
        if output_path.exists():
            return str(output_path), f"✅ Video generated successfully!\nSaved to: {output_path}\n\n💡 Tip: Models are cached in VRAM - subsequent generations are significantly faster!"
        else:
            return None, "❌ Video generation completed but output file not found"
            
    except Exception as e:
        import traceback
        return None, f"❌ Generation failed:\n{str(e)}\n\n{traceback.format_exc()}"


def create_ui():
    """Create the Gradio interface."""
    
    ensure_directories()
    
    with gr.Blocks(
        title="LTX-2 WebUI",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="purple",
            secondary_hue="pink",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Outfit"),
            font_mono=gr.themes.GoogleFont("JetBrains Mono"),
        ).set(
            body_background_fill="#0a0a0f",
            body_background_fill_dark="#0a0a0f",
            block_background_fill="#12121a",
            block_background_fill_dark="#12121a",
            border_color_primary="#2a2a3a",
            border_color_primary_dark="#2a2a3a",
        )
    ) as demo:
        
        # Header
        gr.HTML("""
            <div class="header-container">
                <h1 class="header-title">🎬 LTX-2 WebUI</h1>
                <p class="header-subtitle">
                    Generate stunning AI videos with Lightricks LTX-2 • 4K @ 50 FPS • Up to 20 seconds • Audio-Video Sync<br>
                    <a href="https://huggingface.co/Lightricks/LTX-2.3" target="_blank" style="color: #a855f7;">HuggingFace</a> • 
                    <a href="https://github.com/Lightricks/LTX-2" target="_blank" style="color: #ec4899;">GitHub</a> •
                    <a href="https://docs.ltx.video" target="_blank" style="color: #f97316;">Docs</a>
                </p>
            </div>
        """)
        
        with gr.Tabs() as tabs:
            
            # ===== GENERATE TAB =====
            with gr.Tab("🎥 Generate", id="generate"):
                with gr.Row():
                    # Left Column - Settings
                    with gr.Column(scale=1):
                        
                        # Preset Management Section
                        with gr.Group():
                            gr.Markdown("### 💾 Presets")
                            with gr.Row():
                                preset_dropdown = gr.Dropdown(
                                    choices=get_preset_manager().list_presets(),
                                    value=get_preset_manager().get_default_preset_name(),
                                    label="Load Preset",
                                    scale=2
                                )
                                load_preset_btn = gr.Button("📂 Load", variant="secondary", size="sm", scale=1)
                            
                            with gr.Row():
                                preset_name_input = gr.Textbox(
                                    label="Preset Name",
                                    placeholder="Enter preset name...",
                                    scale=2
                                )
                                save_preset_btn = gr.Button("💾 Save", variant="primary", size="sm", scale=1)
                            
                            with gr.Row():
                                set_default_btn = gr.Button("⭐ Set as Default", variant="secondary", size="sm")
                                delete_preset_btn = gr.Button("🗑️ Delete", variant="secondary", size="sm")
                            
                            preset_status = gr.Markdown("")
                        
                        # Pipeline Selection
                        with gr.Group():
                            gr.Markdown("### 🔧 Pipeline")
                            pipeline_type = gr.Dropdown(
                                choices=list(PIPELINE_TYPES.keys()),
                                value="distilled",
                                label="Pipeline Type"
                            )
                            pipeline_info = gr.Markdown(
                                PIPELINE_TYPES["distilled"]["description"]
                            )
                        
                        # Model Selection
                        with gr.Group():
                            gr.Markdown("### 📦 Models")
                            gr.Markdown("*FastAPI downloads models on demand; use the Models tab here for manual downloads.*", elem_classes=["text-muted"])
                            
                            checkpoint_path = gr.Dropdown(
                                choices=get_checkpoint_choices(),
                                value=get_default_checkpoint(),
                                label="Checkpoint (Main model)",
                                allow_custom_value=True
                            )
                            
                            distilled_lora_path = gr.Dropdown(
                                choices=get_lora_choices(),
                                value="None",
                                label="Distilled LoRA (for two-stage, HQ, and keyframe pipelines)",
                                allow_custom_value=True
                            )
                            
                            spatial_upsampler_path = gr.Dropdown(
                                choices=get_upscaler_choices(),
                                value=get_default_upsampler(),
                                label="Spatial Upsampler (2x resolution - required)",
                                allow_custom_value=True
                            )
                            
                            # Auto-detect Gemma path
                            default_gemma = "./models/gemma" if (MODELS_DIR / "gemma").exists() else "./models/gemma"
                            gemma_path = gr.Textbox(
                                label="Gemma Path",
                                value=default_gemma,
                                placeholder="./models/gemma"
                            )
                            
                            refresh_btn = gr.Button("🔄 Refresh Model Lists", variant="secondary", size="sm")
                        
                        # Generation Settings (defaults match CLI: 1024x1536, 121 frames, 24fps)
                        with gr.Accordion("⚙️ Generation Settings", open=True):
                            with gr.Row():
                                height = gr.Slider(
                                    minimum=256, maximum=2048, value=1024, step=64,
                                    label="Height"
                                )
                                width = gr.Slider(
                                    minimum=256, maximum=2048, value=1536, step=64,
                                    label="Width"
                                )

                            quick_ratio = gr.Radio(
                                choices=[
                                    ("3:2 Default", "3:2"),
                                    ("16:9 HQ", "16:9"),
                                    ("9:16 Vertical", "9:16"),
                                    ("1:1 Square", "1:1"),
                                    ("2:3 Portrait", "2:3"),
                                ],
                                value="3:2",
                                label="Quick Ratio",
                            )
                            
                            with gr.Row():
                                num_frames = gr.Slider(
                                    minimum=9, maximum=257, value=121, step=8,
                                    label="Frames (9, 17, 25, ... 257)"
                                )
                                frame_rate = gr.Slider(
                                    minimum=8, maximum=60, value=24, step=1,
                                    label="FPS"
                                )

                            quick_duration = gr.Radio(
                                choices=[
                                    ("3s", "3"),
                                    ("5s", "5"),
                                    ("8s", "8"),
                                    ("10s", "10"),
                                ],
                                value="5",
                                label="Quick Duration",
                            )
                            
                            with gr.Row():
                                num_inference_steps = gr.Slider(
                                    minimum=4, maximum=100, value=30, step=1,
                                    label="Inference Steps (ignored by Distilled pipeline)"
                                )
                                cfg_guidance_scale = gr.Slider(
                                    minimum=1.0, maximum=15.0, value=3.0, step=0.1,
                                    label="CFG Scale (ignored by Distilled pipeline)"
                                )
                            
                            seed = gr.Number(
                                value=-1, label="Seed (-1 for random)"
                            )
                            
                            enable_fp8 = gr.Checkbox(
                                value=True, label="Enable FP8 Optimization (reduces memory)"
                            )
                            skip_memory_cleanup = gr.Checkbox(
                                value=True, label="Skip memory cleanup (keep pipeline models in VRAM between stages)"
                            )
                        
                        # Image Conditioning
                        with gr.Accordion("🖼️ Image Conditioning", open=False):
                            gr.Markdown("*Condition video on this starting image*")
                            input_image = gr.Image(
                                label="Input Image",
                                type="pil"
                            )
                            image_strength = gr.Slider(
                                minimum=0.0, maximum=1.0, value=1.0, step=0.1,
                                label="Image Strength"
                            )
                        
                        # Keyframe Interpolation
                        with gr.Accordion("🎞️ Keyframe Images", open=False, visible=False) as keyframe_accordion:
                            gr.Markdown("*Upload images to interpolate between*")
                            keyframe_images = gr.Gallery(
                                label="Keyframe Images",
                                type="pil",
                                columns=4
                            )
                        
                        # Video Conditioning (for IC-LoRA)
                        with gr.Accordion("📹 Reference Video", open=False, visible=False) as video_accordion:
                            gr.Markdown("*Reference video for video-to-video*")
                            reference_video = gr.Video(
                                label="Reference Video"
                            )
                    
                    # Right Column - Prompt & Output
                    with gr.Column(scale=1):
                        
                        # Prompt
                        with gr.Group():
                            gr.Markdown("### ✨ Prompt")
                            prompt = gr.Textbox(
                                label="Prompt",
                                placeholder="A beautiful sunset over the ocean with waves crashing on the shore...",
                                lines=4
                            )
                            negative_prompt = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="blurry, low quality, distorted...",
                                lines=2
                            )
                        
                        # Generate Button
                        generate_btn = gr.Button(
                            "🚀 Generate Video",
                            variant="primary",
                            size="lg"
                        )
                        
                        # Output
                        with gr.Group():
                            gr.Markdown("### 🎬 Output")
                            output_video = gr.Video(
                                label="Generated Video",
                                autoplay=True
                            )
                            output_status = gr.Markdown("")
                
                # Event handlers
                def update_pipeline_info(pipeline_type):
                    info = PIPELINE_TYPES.get(pipeline_type, {})
                    desc = info.get("description", "")
                    reqs = info.get("requires", [])
                    features = info.get("features", {})
                    
                    req_text = ", ".join(reqs)
                    feature_text = ""
                    if features:
                        stages = features.get("stages", "?")
                        cfg = "✅" if features.get("cfg") else "❌"
                        upsampling = "✅" if features.get("upsampling") else "❌"
                        conditioning = features.get("conditioning", "Image")
                        feature_text = f"\n\n**Features:** {stages} stages | CFG: {cfg} | Upsampling: {upsampling} | Conditioning: {conditioning}"
                    
                    return f"{desc}{feature_text}\n\n**Requires:** {req_text}"
                
                def update_visibility(pipeline_type):
                    show_keyframes = pipeline_type == "keyframe_interpolation"
                    show_video = pipeline_type == "ic_lora"
                    return (
                        gr.update(visible=show_keyframes),
                        gr.update(visible=show_video)
                    )

                def apply_quick_ratio(ratio):
                    presets = {
                        "3:2": (1024, 1536),
                        "16:9": (1088, 1920),
                        "9:16": (1792, 1024),
                        "1:1": (1024, 1024),
                        "2:3": (1536, 1024),
                    }
                    next_height, next_width = presets.get(ratio, presets["3:2"])
                    return gr.update(value=next_height), gr.update(value=next_width)

                def apply_quick_duration(duration, frame_rate):
                    fps = float(frame_rate or 24)
                    raw_frames = round(float(duration or 5) * fps) + 1
                    snapped_frames = round((raw_frames - 1) / 8) * 8 + 1
                    clamped_frames = max(9, min(257, snapped_frames))
                    return gr.update(value=clamped_frames)

                quick_ratio.change(
                    apply_quick_ratio,
                    inputs=[quick_ratio],
                    outputs=[height, width]
                )

                quick_duration.change(
                    apply_quick_duration,
                    inputs=[quick_duration, frame_rate],
                    outputs=[num_frames]
                )

                frame_rate.change(
                    apply_quick_duration,
                    inputs=[quick_duration, frame_rate],
                    outputs=[num_frames]
                )
                
                pipeline_type.change(
                    update_pipeline_info,
                    inputs=[pipeline_type],
                    outputs=[pipeline_info]
                ).then(
                    update_visibility,
                    inputs=[pipeline_type],
                    outputs=[keyframe_accordion, video_accordion]
                )
                
                def refresh_models():
                    return (
                        gr.update(choices=get_checkpoint_choices()),
                        gr.update(choices=get_lora_choices()),
                        gr.update(choices=get_upscaler_choices())
                    )
                
                refresh_btn.click(
                    refresh_models,
                    outputs=[checkpoint_path, distilled_lora_path, spatial_upsampler_path]
                )
                
                # Preset management handlers
                def load_preset(preset_name):
                    """Load a preset and return all setting values."""
                    preset_manager = get_preset_manager()
                    preset = preset_manager.get_preset(preset_name)
                    if not preset:
                        return [gr.update()] * 15 + [f"❌ Preset '{preset_name}' not found"]
                    
                    return [
                        gr.update(value=preset.pipeline_type),  # pipeline_type
                        gr.update(value=preset.checkpoint_path if preset.checkpoint_path else get_default_checkpoint()),  # checkpoint_path
                        gr.update(value=preset.distilled_lora_path),  # distilled_lora_path
                        gr.update(value=preset.spatial_upsampler_path if preset.spatial_upsampler_path else get_default_upsampler()),  # spatial_upsampler_path
                        gr.update(value=preset.gemma_path),  # gemma_path
                        gr.update(value=preset.height),  # height
                        gr.update(value=preset.width),  # width
                        gr.update(value=preset.num_frames),  # num_frames
                        gr.update(value=preset.frame_rate),  # frame_rate
                        gr.update(value=preset.num_inference_steps),  # num_inference_steps
                        gr.update(value=preset.cfg_guidance_scale),  # cfg_guidance_scale
                        gr.update(value=preset.seed),  # seed
                        gr.update(value=preset.enable_fp8),  # enable_fp8
                        gr.update(value=preset.skip_memory_cleanup),  # skip_memory_cleanup
                        gr.update(value=preset.image_strength),  # image_strength
                        f"✅ Loaded preset: **{preset_name}**"  # status
                    ]
                
                def save_preset(
                    preset_name_input, pipeline_type, checkpoint_path, distilled_lora_path,
                    spatial_upsampler_path, gemma_path, height, width, num_frames,
                    frame_rate, num_inference_steps, cfg_guidance_scale, seed, enable_fp8,
                    skip_memory_cleanup, image_strength
                ):
                    """Save current settings as a preset."""
                    preset_manager = get_preset_manager()
                    
                    if not preset_name_input or not preset_name_input.strip():
                        return gr.update(), "❌ Please enter a preset name"
                    
                    name = preset_name_input.strip()
                    existing = preset_manager.get_preset(name)
                    
                    preset = preset_manager.create_preset_from_settings(
                        name=name,
                        description=f"Saved from WebUI on {time.strftime('%Y-%m-%d %H:%M')}",
                        pipeline_type=pipeline_type,
                        checkpoint_path=checkpoint_path if checkpoint_path else "",
                        distilled_lora_path=distilled_lora_path if distilled_lora_path else "None",
                        spatial_upsampler_path=spatial_upsampler_path if spatial_upsampler_path else "",
                        gemma_path=gemma_path if gemma_path else "./models/gemma",
                        height=int(height),
                        width=int(width),
                        num_frames=int(num_frames),
                        frame_rate=float(frame_rate),
                        num_inference_steps=int(num_inference_steps),
                        cfg_guidance_scale=float(cfg_guidance_scale),
                        seed=int(seed),
                        enable_fp8=bool(enable_fp8),
                        skip_memory_cleanup=bool(skip_memory_cleanup),
                        image_strength=float(image_strength),
                    )
                    
                    preset_manager.save_preset(preset, overwrite=True)
                    
                    action = "updated" if existing else "created"
                    return (
                        gr.update(choices=preset_manager.list_presets(), value=name),
                        f"✅ Preset **{name}** {action} successfully!"
                    )
                
                def set_preset_as_default(preset_name):
                    """Set a preset as the default."""
                    preset_manager = get_preset_manager()
                    if preset_manager.set_default(preset_name):
                        return f"⭐ **{preset_name}** is now the default preset"
                    return f"❌ Failed to set default preset"
                
                def delete_preset(preset_name):
                    """Delete a preset."""
                    preset_manager = get_preset_manager()
                    if preset_name == DEFAULT_PRESET_NAME:
                        return gr.update(), f"❌ Cannot delete the default preset"
                    
                    if preset_manager.delete_preset(preset_name):
                        new_default = preset_manager.get_default_preset_name()
                        return (
                            gr.update(choices=preset_manager.list_presets(), value=new_default),
                            f"✅ Preset **{preset_name}** deleted"
                        )
                    return gr.update(), f"❌ Failed to delete preset"
                
                def refresh_presets():
                    """Refresh preset list."""
                    preset_manager = get_preset_manager()
                    return gr.update(choices=preset_manager.list_presets())
                
                load_preset_btn.click(
                    load_preset,
                    inputs=[preset_dropdown],
                    outputs=[
                        pipeline_type, checkpoint_path, distilled_lora_path,
                        spatial_upsampler_path, gemma_path, height, width, num_frames,
                        frame_rate, num_inference_steps, cfg_guidance_scale, seed, enable_fp8,
                        skip_memory_cleanup, image_strength, preset_status
                    ]
                )
                
                save_preset_btn.click(
                    save_preset,
                    inputs=[
                        preset_name_input, pipeline_type, checkpoint_path, distilled_lora_path,
                        spatial_upsampler_path, gemma_path, height, width, num_frames,
                        frame_rate, num_inference_steps, cfg_guidance_scale, seed, enable_fp8,
                        skip_memory_cleanup, image_strength
                    ],
                    outputs=[preset_dropdown, preset_status]
                )
                
                set_default_btn.click(
                    set_preset_as_default,
                    inputs=[preset_dropdown],
                    outputs=[preset_status]
                )
                
                delete_preset_btn.click(
                    delete_preset,
                    inputs=[preset_dropdown],
                    outputs=[preset_dropdown, preset_status]
                )
                
                generate_btn.click(
                    generate_video,
                    inputs=[
                        pipeline_type,
                        checkpoint_path,
                        distilled_lora_path,
                        spatial_upsampler_path,
                        gemma_path,
                        prompt,
                        negative_prompt,
                        height,
                        width,
                        num_frames,
                        frame_rate,
                        num_inference_steps,
                        cfg_guidance_scale,
                        seed,
                        enable_fp8,
                        skip_memory_cleanup,
                        input_image,
                        image_strength,
                        reference_video,
                        keyframe_images,
                    ],
                    outputs=[output_video, output_status]
                )
            
            # ===== MODELS TAB =====
            with gr.Tab("📦 Models", id="models"):
                gr.Markdown("""
                ### Download Models from HuggingFace
                
                Download the required model files from [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3).
                Models will be saved to the `./models` directory.
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        model_status_html = gr.HTML(
                            value=refresh_model_status(),
                            label="Model Status"
                        )
                    
                    with gr.Column(scale=1):
                        model_to_download = gr.Dropdown(
                            choices=list(CHECKPOINTS.keys()),
                            label="Select Model to Download"
                        )
                        
                        download_btn = gr.Button(
                            "⬇️ Download Selected Model",
                            variant="primary"
                        )
                        
                        refresh_status_btn = gr.Button(
                            "🔄 Refresh Status",
                            variant="secondary"
                        )
                        
                        download_status = gr.Markdown("")
                
                # Download handlers
                download_btn.click(
                    download_model,
                    inputs=[model_to_download],
                    outputs=[download_status]
                ).then(
                    refresh_model_status,
                    outputs=[model_status_html]
                ).then(
                    refresh_models,
                    outputs=[checkpoint_path, distilled_lora_path, spatial_upsampler_path]
                )
                
                refresh_status_btn.click(
                    refresh_model_status,
                    outputs=[model_status_html]
                ).then(
                    refresh_models,
                    outputs=[checkpoint_path, distilled_lora_path, spatial_upsampler_path]
                )
                
                # Gemma download section
                gr.Markdown("""
                ---
                ### Gemma 3 Text Encoder
                
                The Gemma 3 **12B** text encoder is required for all pipelines. We use the official **QAT** model which **does not require** HuggingFace authentication:
                
                ```bash
                # Download QAT version (no token required)
                huggingface-cli download Lightricks/gemma-3-12b-it-qat-q4_0-unquantized --local-dir ./models/gemma
                
                # Or using Python
                from huggingface_hub import snapshot_download
                snapshot_download("Lightricks/gemma-3-12b-it-qat-q4_0-unquantized", local_dir="./models/gemma")
                ```
                
                After downloading, set the **Gemma Path** in the Generate tab to `./models/gemma`.
                
                > ⚠️ **Important:** LTX-2 requires the 12B model. Gemma 2 and Gemma 3 4B will cause dimension mismatch errors!
                """)
            
            # ===== GALLERY TAB =====
            with gr.Tab("🖼️ Gallery", id="gallery"):
                gr.Markdown("### Generated Videos")
                
                def get_gallery_videos():
                    videos = []
                    if OUTPUTS_DIR.exists():
                        for f in sorted(OUTPUTS_DIR.glob("*.mp4"), key=os.path.getmtime, reverse=True)[:20]:
                            videos.append(str(f))
                    return videos
                
                gallery_videos = gr.Gallery(
                    value=get_gallery_videos,
                    label="Recent Generations",
                    columns=3,
                    object_fit="cover"
                )
                
                refresh_gallery_btn = gr.Button("🔄 Refresh Gallery", variant="secondary")
                refresh_gallery_btn.click(
                    get_gallery_videos,
                    outputs=[gallery_videos]
                )
            
            # ===== SETTINGS TAB =====
            with gr.Tab("⚙️ Settings", id="settings"):
                gr.Markdown("""
                ### Configuration
                
                Configure default paths and settings for the LTX-2 WebUI.
                """)
                
                with gr.Group():
                    gr.Markdown("#### Directories")
                    models_dir_input = gr.Textbox(
                        value=str(MODELS_DIR),
                        label="Models Directory"
                    )
                    outputs_dir_input = gr.Textbox(
                        value=str(OUTPUTS_DIR),
                        label="Outputs Directory"
                    )
                
                with gr.Group():
                    gr.Markdown("#### 🧠 VRAM Management")
                    gr.Markdown("""
                    Models are cached in VRAM for faster subsequent generations.
                    Use the button below to clear the cache and free VRAM.
                    """)
                    
                    vram_status = gr.Markdown(value=get_vram_status())
                    
                    with gr.Row():
                        clear_vram_btn = gr.Button("🗑️ Clear VRAM Cache", variant="secondary")
                        refresh_vram_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                    
                    clear_vram_result = gr.Markdown("")
                    
                    clear_vram_btn.click(
                        clear_vram_cache,
                        outputs=[clear_vram_result]
                    ).then(
                        get_vram_status,
                        outputs=[vram_status]
                    )
                    
                    refresh_vram_btn.click(
                        get_vram_status,
                        outputs=[vram_status]
                    )
                
                with gr.Group():
                    gr.Markdown("#### System Information")
                    
                    # GPU Info
                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0)
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        gpu_info = f"✅ **{gpu_name}** ({gpu_memory:.1f} GB)"
                    else:
                        gpu_info = "❌ No CUDA GPU detected"
                    
                    gr.Markdown(f"**GPU:** {gpu_info}")
                    gr.Markdown(f"**PyTorch:** {torch.__version__}")
                    gr.Markdown(f"**CUDA Available:** {torch.cuda.is_available()}")
            
            # ===== HELP TAB =====
            with gr.Tab("❓ Help", id="help"):
                gr.Markdown("""
                ### LTX-2 WebUI Help
                
                #### 🚀 Quick Start
                
                Run `./run.sh` to install dependencies and start the API. Required models are downloaded on first generation:
                
                - ✅ LTX-2.3 22B Distilled Checkpoint
                - ✅ Gemma 3 12B QAT Text Encoder (no HF token required)
                - ✅ Spatial Upsampler (~1GB) - doubles resolution
                
                Then enter a prompt and click **Generate**!
                
                ---
                
                #### 🎯 Pipeline Selection Guide
                
                **Decision Tree:**
                ```
                Do you need to condition on existing images/videos?
                ├─ YES → Do you have reference videos for video-to-video?
                │  ├─ YES → Use IC-LoRA Pipeline
                │  └─ NO → Do you have keyframe images to interpolate?
                │     ├─ YES → Use Keyframe Interpolation Pipeline
                │     └─ NO → Use any pipeline (all support image conditioning)
                │
                └─ NO → Text-to-video only
                   ├─ Need highest quality 16:9? → Use Two-Stage HQ Pipeline (LTX-2.3 res_2s)
                   ├─ Need production quality? → Use Two-Stage Pipeline
                   └─ Need fastest inference? → Use Distilled Pipeline (8 sigmas, default)
                ```
                
                > **Note:** One-Stage Pipeline is for educational purposes only. A2Vid, Retake, HDR IC-LoRA, and LipDub are available in the LTX-2.3 package as specialized CLI pipelines.
                
                ---
                
                #### 📊 Features Comparison
                
                | Pipeline | Stages | CFG | Upsampling | Conditioning | Best For |
                |----------|--------|-----|------------|--------------|----------|
                | **Distilled** ⚡ | 2 | ❌ | ✅ | Image | Fastest inference (8 sigmas) |
                | **Two-Stage** 🎬 | 2 | ✅ | ✅ | Image | **Production quality** (recommended) |
                | **Two-Stage HQ** 🎞️ | 2 | ✅ | ✅ | Image | LTX-2.3 res_2s high-quality 16:9 |
                | **One-Stage** 📚 | 1 | ✅ | ❌ | Image | Educational, prototyping |
                | **IC-LoRA** 🎞️ | 2 | ❌ | ✅ | Image + Video | Video-to-video transformations |
                | **Keyframe** 🎨 | 2 | ✅ | ✅ | Keyframes | Animation, interpolation |
                
                ---
                
                #### System Requirements
                
                - **Python**: >= 3.12
                - **CUDA**: >= 12.7
                - **PyTorch**: ~= 2.7
                - **GPU Memory**: 24GB+ recommended (lower with FP8)
                - **Disk Space**: enough for the LTX-2.3 checkpoint, Gemma encoder, and upscaler
                
                ---
                
                #### First-Time Setup
                
                1. **Run the launcher** (downloads models on demand):
                   ```bash
                   ./run.sh
                   ```
                   
                2. Or pre-download models manually / with `./run.sh --with-models`:
                   ```bash
                   # Gemma 3 12B QAT (no token required)
                   huggingface-cli download Lightricks/gemma-3-12b-it-qat-q4_0-unquantized --local-dir ./models/gemma
                   
                   # LTX-2 checkpoint
                   huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-22b-distilled-1.1.safetensors --local-dir ./models/checkpoints
                   
                   # Spatial upsampler
                   huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x2-1.1.safetensors --local-dir ./models/upsamplers
                   ```
                
                ---
                
                #### Model Checkpoints
                
                | Model | Size | Description |
                |-------|------|-------------|
                | `ltx-2.3-22b-distilled-1.1` | 22B | **Recommended** - fastest distilled pipeline |
                | `ltx-2.3-22b-dev` | 22B | Development checkpoint for CFG/two-stage quality |
                | `ltx-2.3-22b-distilled-lora-384-1.1` | adapter | Required for two-stage and keyframe refinement |
                
                ---
                
                #### How Resolution Works (Two-Stage Pipelines)
                
                When you set a resolution like **1024×1536** in the UI:
                1. **Stage 1**: Generates at **half resolution** (512×768) - faster, uses less VRAM
                2. **Stage 2**: **Spatial upsampler** doubles resolution to 1024×1536 + refinement
                
                This is why spatial upsampler is **required** for most pipelines!
                
                ---
                
                #### Tips
                
                - **Memory Issues?** Enable FP8 optimization
                - **Better Quality?** Use Two-Stage pipeline with the 2.3 dev checkpoint, 30+ inference steps, and CFG scale around 3.0
                - **Faster Generation?** Use Distilled pipeline (only 8 steps, no CFG needed!)
                - **Image Conditioning?** Upload a starting image - works with all pipelines
                - **Video-to-Video?** Use IC-LoRA pipeline with reference video
                
                ---
                
                #### Links
                
                - 📚 [LTX-2 Documentation](https://docs.ltx.video)
                - 🐙 [GitHub Repository](https://github.com/Lightricks/LTX-2)
                - 🤗 [HuggingFace Models](https://huggingface.co/Lightricks/LTX-2.3)
                - 💬 [Community](https://huggingface.co/Lightricks/LTX-2.3/discussions)
                """)
        
        # Footer
        gr.HTML("""
            <div style="text-align: center; padding: 2rem 0; color: var(--text-secondary); font-size: 0.9rem;">
                <p>Built with ❤️ for the AI video generation community</p>
                <p style="margin-top: 0.5rem;">
                    <a href="https://github.com/Lightricks/LTX-2" target="_blank" style="color: var(--accent-purple);">GitHub</a> •
                    <a href="https://huggingface.co/Lightricks/LTX-2.3" target="_blank" style="color: var(--accent-pink);">HuggingFace</a>
                </p>
            </div>
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )

