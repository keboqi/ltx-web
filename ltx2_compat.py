"""Compatibility helpers for the vendored LTX-2 package.

The WebUI keeps LTX-2 under ``./LTX-2`` while the official repository uses
package paths rooted at ``packages/``. This module centralizes the local import
path setup, LTX-2.3 model registry, and the 2.3 pipeline API details.
"""

from __future__ import annotations

import inspect
import gc
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
LTX2_ROOT = REPO_ROOT / "LTX-2"
HF_REPO_ID = "Lightricks/LTX-2.3"
GEMMA_REPO_ID = "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"

GEMMA_REQUIRED_FILES = (
    "config.json",
    "tokenizer.model",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
)

DEFAULT_CHECKPOINT_KEY = "ltx-2.3-22b-distilled-1.1"
DEFAULT_UPSAMPLER_KEY = "ltx-2.3-spatial-upscaler-x2-1.1"
DEFAULT_DISTILLED_LORA_KEY = "ltx-2.3-22b-distilled-lora-384-1.1"

DEFAULT_CHECKPOINT_CANDIDATES = [
    "ltx-2.3-22b-distilled-1.1.safetensors",
    "ltx-2.3-22b-dev.safetensors",
]

DEFAULT_UPSAMPLER_CANDIDATES = [
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors",
]

SUPPORTED_PIPELINE_TYPES = {
    "distilled",
    "ti2vid_two_stages",
    "ti2vid_two_stages_hq",
    "ti2vid_one_stage",
    "ic_lora",
    "keyframe_interpolation",
}

LTX23_SPECIALIZED_PIPELINE_TYPES = {
    "a2vid_two_stage",
    "retake",
    "hdr_ic_lora",
    "lipdub",
}

ALL_LTX23_PIPELINE_TYPES = SUPPORTED_PIPELINE_TYPES | LTX23_SPECIALIZED_PIPELINE_TYPES

PIPELINE_DESCRIPTIONS = {
    "distilled": "Fast text/image-to-video pipeline using the distilled checkpoint.",
    "ti2vid_two_stages": "Recommended production text/image-to-video pipeline.",
    "ti2vid_two_stages_hq": "Higher-quality two-stage text/image-to-video pipeline using the res_2s sampler.",
    "ti2vid_one_stage": "Single-stage text/image-to-video pipeline for education and quick prototyping.",
    "ic_lora": "Video-to-video and image-to-video control pipeline with IC-LoRA.",
    "keyframe_interpolation": "Two-stage interpolation between keyframe images.",
    "a2vid_two_stage": "Audio-to-video pipeline conditioned on an input audio file.",
    "retake": "Regenerates a time region of an existing video.",
    "hdr_ic_lora": "HDR IC-LoRA video-to-video pipeline for linear float/EXR workflows.",
    "lipdub": "Lip dubbing pipeline with reference video and audio conditioning.",
}

PIPELINES_REQUIRING_UPSAMPLER = {
    "distilled",
    "ic_lora",
    "ti2vid_two_stages",
    "ti2vid_two_stages_hq",
    "keyframe_interpolation",
}

PIPELINES_REQUIRING_DISTILLED_LORA = {
    "ti2vid_two_stages",
    "ti2vid_two_stages_hq",
    "keyframe_interpolation",
}

ONE_STAGE_PIPELINES = {"ti2vid_one_stage"}
GUIDANCELESS_PIPELINES = {"distilled", "ic_lora"}


def ensure_ltx2_paths() -> None:
    """Prefer the vendored LTX-2 source tree over any older site install."""
    for package in ("ltx-core", "ltx-pipelines", "ltx-trainer"):
        src = LTX2_ROOT / "packages" / package / "src"
        if src.exists():
            src_path = str(src)
            if src_path not in sys.path:
                sys.path.insert(0, src_path)


ensure_ltx2_paths()


CHECKPOINTS: dict[str, dict[str, str]] = {
    # LTX-2.3 models.
    "ltx-2.3-22b-distilled-1.1": {
        "filename": "ltx-2.3-22b-distilled-1.1.safetensors",
        "size": "22B",
        "description": "LTX-2.3 distilled checkpoint, recommended for the fast pipeline",
        "type": "checkpoint",
        "repo_id": HF_REPO_ID,
    },
    "ltx-2.3-22b-dev": {
        "filename": "ltx-2.3-22b-dev.safetensors",
        "size": "22B",
        "description": "LTX-2.3 development checkpoint for CFG/two-stage quality pipelines",
        "type": "checkpoint",
        "repo_id": HF_REPO_ID,
    },
    "ltx-2.3-22b-distilled-lora-384-1.1": {
        "filename": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "size": "adapter",
        "description": "LTX-2.3 distilled LoRA for two-stage refinement",
        "type": "lora",
        "repo_id": HF_REPO_ID,
    },
    "ltx-2.3-spatial-upscaler-x2-1.1": {
        "filename": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "size": "upscaler",
        "description": "LTX-2.3 2x spatial upscaler",
        "type": "upscaler",
        "repo_id": HF_REPO_ID,
    },
    "ltx-2.3-spatial-upscaler-x1.5-1.0": {
        "filename": "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors",
        "size": "upscaler",
        "description": "LTX-2.3 1.5x spatial upscaler",
        "type": "upscaler",
        "repo_id": HF_REPO_ID,
    },
    "ltx-2.3-temporal-upscaler-x2-1.0": {
        "filename": "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
        "size": "upscaler",
        "description": "LTX-2.3 2x temporal upscaler, reserved for future pipelines",
        "type": "upscaler",
        "repo_id": HF_REPO_ID,
    },
    "ltx-2.3-22b-ic-lora-union-control-ref0.5": {
        "filename": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        "size": "adapter",
        "description": "LTX-2.3 IC-LoRA union control adapter",
        "type": "lora",
        "repo_id": "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
    },
    "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5": {
        "filename": "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
        "size": "adapter",
        "description": "LTX-2.3 IC-LoRA motion-track control adapter",
        "type": "lora",
        "repo_id": "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
    },
}


def is_ltx23_model_path(path: str | Path) -> bool:
    """Return True when a local model file belongs to the LTX-2.3 registry."""
    name = Path(path).name.lower()
    return name.startswith("ltx-2.3-")


def model_repo_id(model_key: str) -> str:
    """Return the Hugging Face repo that owns a model key."""
    return CHECKPOINTS[model_key].get("repo_id", HF_REPO_ID)


def missing_gemma_files(gemma_root: str | Path | None) -> list[str]:
    """Return Gemma files missing from the local text encoder directory."""
    if not gemma_root:
        return list(GEMMA_REQUIRED_FILES)

    root = Path(gemma_root)
    if not root.exists():
        return list(GEMMA_REQUIRED_FILES)

    missing: list[str] = []
    for filename in GEMMA_REQUIRED_FILES:
        if not any(root.rglob(filename)):
            missing.append(filename)
    return missing


def validate_gemma_root(gemma_root: str | Path | None) -> tuple[bool, str | None]:
    """Validate the Gemma directory expected by the LTX-2.3 prompt encoder."""
    missing = missing_gemma_files(gemma_root)
    if not missing:
        return True, None

    path = gemma_root or "models/gemma"
    files = ", ".join(missing)
    return (
        False,
        (
            f"Gemma text encoder is incomplete at {path!s}. Missing: {files}.\n\n"
            f"Download the LTX-2.3-compatible Gemma mirror with:\n"
            f"hf download {GEMMA_REPO_ID} --local-dir ./models/gemma"
        ),
    )


def _has_fp8_scale_tensors(checkpoint_path: str) -> bool:
    try:
        from safetensors import safe_open

        with safe_open(checkpoint_path, framework="pt") as f:
            return any(key.endswith(".weight_scale") or ".weight_scale." in key for key in f.keys())
    except Exception:
        return False


def build_quantization_policy(enable_fp8: bool, checkpoint_path: str):
    """Map the WebUI's FP8 toggle to LTX-2.3 quantization policies."""
    if not enable_fp8:
        return None

    from ltx_pipelines.utils.quantization_factory import QuantizationKind

    if _has_fp8_scale_tensors(checkpoint_path):
        return QuantizationKind.FP8_SCALED_MM.to_policy(checkpoint_path)
    return QuantizationKind.FP8_CAST.to_policy(checkpoint_path)


def make_lora_list(lora_path: str | None):
    """Create a single LoRA descriptor list from a WebUI path field."""
    if not lora_path or lora_path == "None" or not Path(lora_path).exists():
        return []

    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps

    return [LoraPathStrengthAndSDOps(lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)]


def make_image_conditioning(path: str | Path, frame_idx: int, strength: float, crf: int | None = None):
    """Create the image conditioning object expected by LTX-2.3 pipelines."""
    from ltx_pipelines.utils.args import ImageConditioningInput

    if crf is None:
        return ImageConditioningInput(str(path), int(frame_idx), float(strength))
    return ImageConditioningInput(str(path), int(frame_idx), float(strength), int(crf))


def create_pipeline(
    *,
    pipeline_type: str,
    checkpoint_path: str,
    spatial_upsampler_path: str | None,
    gemma_path: str,
    lora_path: str | None,
    enable_fp8: bool,
):
    """Instantiate a local LTX-2.3 pipeline for the selected WebUI mode."""
    if pipeline_type not in SUPPORTED_PIPELINE_TYPES:
        if pipeline_type in LTX23_SPECIALIZED_PIPELINE_TYPES:
            raise ValueError(
                f"{pipeline_type} is an LTX-2.3 specialized CLI pipeline. "
                "The WebUI does not have the required input controls for it yet."
            )
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

    os.environ.setdefault("LTX_KEEP_PIPELINE_MODELS", "1")
    quantization = build_quantization_policy(enable_fp8, checkpoint_path)

    if pipeline_type == "distilled":
        from ltx_pipelines.distilled import DistilledPipeline

        return DistilledPipeline(
            distilled_checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=gemma_path,
            loras=[],
            quantization=quantization,
        )

    if pipeline_type == "ti2vid_two_stages":
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

        return TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=make_lora_list(lora_path),
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=gemma_path,
            loras=[],
            quantization=quantization,
        )

    if pipeline_type == "ti2vid_two_stages_hq":
        from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline

        distilled_lora = make_lora_list(lora_path)
        if not distilled_lora:
            raise ValueError("TI2VidTwoStagesHQPipeline requires a distilled LoRA.")

        return TI2VidTwoStagesHQPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            distilled_lora_strength_stage_1=0.25,
            distilled_lora_strength_stage_2=0.5,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=gemma_path,
            loras=(),
            quantization=quantization,
        )

    if pipeline_type == "ti2vid_one_stage":
        from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline

        return TI2VidOneStagePipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_path,
            loras=[],
            quantization=quantization,
        )

    if pipeline_type == "ic_lora":
        from ltx_pipelines.ic_lora import ICLoraPipeline

        return ICLoraPipeline(
            distilled_checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=gemma_path,
            loras=make_lora_list(lora_path),
            quantization=quantization,
        )

    if pipeline_type == "keyframe_interpolation":
        from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline

        return KeyframeInterpolationPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=make_lora_list(lora_path),
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=gemma_path,
            loras=[],
            quantization=quantization,
        )

    raise ValueError(f"Unknown pipeline type: {pipeline_type}")


def release_pipeline_models(pipeline: Any | None) -> None:
    """Release cached model modules held by a persistent LTX-2.3 pipeline."""
    if pipeline is None:
        return

    try:
        import torch
    except Exception:
        torch = None

    seen: set[int] = set()

    def release_obj(obj: Any) -> None:
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if torch is not None and isinstance(obj, torch.nn.Module):
            try:
                obj.to("meta")
            except Exception:
                pass
            return

        if isinstance(obj, dict):
            for value in list(obj.values()):
                release_obj(value)
            return

        if isinstance(obj, (list, tuple, set)):
            for value in list(obj):
                release_obj(value)
            return

        if not hasattr(obj, "__dict__"):
            return

        for name, value in list(vars(obj).items()):
            if name.startswith("_cached_"):
                release_obj(value)
                try:
                    setattr(obj, name, None)
                except Exception:
                    pass
            elif name in {
                "prompt_encoder",
                "image_conditioner",
                "audio_conditioner",
                "stage",
                "stage_1",
                "stage_2",
                "upsampler",
                "video_decoder",
                "audio_decoder",
            }:
                release_obj(value)

    release_obj(pipeline)
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def build_guidance_kwargs(
    *,
    pipeline_type: str,
    checkpoint_path: str,
    negative_prompt: str,
    num_inference_steps: int,
    cfg_guidance_scale: float,
) -> dict[str, Any]:
    """Build the extra LTX-2.3 call args for CFG/STG pipelines."""
    if pipeline_type in GUIDANCELESS_PIPELINES:
        return {}

    from ltx_pipelines.utils.constants import LTX_2_3_HQ_PARAMS, LTX_2_3_PARAMS, detect_params

    if pipeline_type == "ti2vid_two_stages_hq":
        params = LTX_2_3_HQ_PARAMS
    else:
        params = detect_params(checkpoint_path) if Path(checkpoint_path).exists() else LTX_2_3_PARAMS

    return {
        "negative_prompt": negative_prompt,
        "num_inference_steps": int(num_inference_steps),
        "video_guider_params": replace(params.video_guider_params, cfg_scale=float(cfg_guidance_scale)),
        "audio_guider_params": params.audio_guider_params,
        "max_batch_size": 1,
    }


def encode_video_compat(encode_video_func, **kwargs):
    """Call encode_video across old/new signatures."""
    accepted = inspect.signature(encode_video_func).parameters
    return encode_video_func(**{key: value for key, value in kwargs.items() if key in accepted})
