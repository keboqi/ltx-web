# LTX-2.3 WebUI

A web interface and REST API for [Lightricks LTX-2](https://github.com/Lightricks/LTX-2), updated for the LTX-2.3 package layout and pipeline API.

## Quick Start

```bash
./run.sh
```

The launcher installs WebUI dependencies, installs the vendored LTX-2 packages from `LTX-2/packages`, and starts the FastAPI UI at `http://localhost:8000`. Model weights are downloaded on demand during the first generation that needs them.

To prepare the environment without starting the API:

```bash
./run.sh --no-launch
```

To pre-download the default LTX-2.3 weights during setup:

```bash
./run.sh --with-models
```

## Default Models

| Model | Repository | Local path |
| --- | --- | --- |
| `ltx-2.3-22b-distilled-1.1.safetensors` | `Lightricks/LTX-2.3` | `models/checkpoints/` |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | `Lightricks/LTX-2.3` | `models/upsamplers/` |
| Gemma 3 12B QAT | `Lightricks/gemma-3-12b-it-qat-q4_0-unquantized` | `models/gemma/` |

The API can download these files automatically on first use. Manual download commands:

```bash
huggingface-cli download Lightricks/gemma-3-12b-it-qat-q4_0-unquantized --local-dir ./models/gemma
huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-22b-distilled-1.1.safetensors --local-dir ./models/checkpoints
huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x2-1.1.safetensors --local-dir ./models/upsamplers
```

## Interfaces

FastAPI/Web UI:

```bash
python api.py
```

Open `http://localhost:8000`.

Gradio UI:

```bash
python app.py
```

Open `http://localhost:7860`.

## Pipelines

The WebUI supports the LTX-2.3 pipelines that match the current prompt, image, and reference-video controls:

| Pipeline | Notes |
| --- | --- |
| `distilled` | Fast default pipeline using `distilled_checkpoint_path` and spatial upscaling. |
| `ti2vid_two_stages` | Uses LTX-2.3 `MultiModalGuiderParams`; requires the 2.3 distilled LoRA. |
| `ti2vid_two_stages_hq` | HQ two-stage pipeline using the LTX-2.3 `res_2s` sampler; requires the 2.3 distilled LoRA. |
| `ti2vid_one_stage` | Single-stage CFG pipeline for lower-resolution prototyping. |
| `ic_lora` | Distilled checkpoint plus optional IC-LoRA adapter. |
| `keyframe_interpolation` | Two-stage interpolation; requires the 2.3 distilled LoRA. |

The upstream `ltx-pipelines` package also includes specialized LTX-2.3 modules for `a2vid_two_stage`, `retake`, `hdr_ic_lora`, and `lipdub`. Those require additional audio, source-video, HDR, or reference-voice inputs, so the UI lists them as LTX-2.3 specialized CLI modes until matching controls are added.

The old single `cfg_guidance_scale` UI field is mapped to LTX-2.3 video CFG guidance while audio guidance uses the checkpoint-detected defaults from LTX-2.3.

## Pipeline Cache

The app keeps the selected pipeline and its loaded LTX-2.3 model blocks resident after a generation so repeat runs can reuse them. Use `POST /pipeline/unload` or the **Unload Pipeline** button in the Models tab when you need to free VRAM manually.

## Generate Settings

The FastAPI UI and Gradio UI include quick aspect-ratio switches for common LTX-2.3 sizes:

| Ratio | Resolution |
| --- | --- |
| `3:2` | `1536x1024` |
| `16:9` | `1920x1088` |
| `9:16` | `1024x1792` |
| `1:1` | `1024x1024` |
| `2:3` | `1024x1536` |

They also include quick duration switches. Frame count is calculated from the selected FPS as `duration * fps + 1`, then snapped to the nearest LTX-friendly `8n + 1` frame count within the supported range. At 24 FPS, `5s` becomes `121` frames and `8s` becomes `193` frames.

## API Example

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cinematic sunrise over a glass city"}'
```

The endpoint streams Server-Sent Events and returns a `download_url` when generation completes.

## Notes

- The vendored `LTX-2/` directory is refreshed from official `Lightricks/LTX-2` main.
- LTX-2.3 model downloads use `Lightricks/LTX-2.3`; legacy LTX-2 model keys are no longer shown by the app.
- Python dependencies include the current LTX packages' media stack (`av`, `openimageio`) and transformer stack.
