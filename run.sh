#!/bin/bash

set -e

#####################################################
#                CONFIGURATION                      #
#####################################################
# HuggingFace token (only needed if a model download is gated for your account)
# Get your token from: https://huggingface.co/settings/tokens

HF_TOKEN="${HF_TOKEN:-}"  # Optional
GEMMA_REPO="Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"
LTX_REPO="https://github.com/Lightricks/LTX-2.git"
CHECKPOINT_REPO="Lightricks/LTX-2.3"
DISTILLED_CHECKPOINT="models/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors"
SPATIAL_UPSCALER="models/upsamplers/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
LTX_WEB_MODEL_MODE="${LTX_WEB_MODEL_MODE:-${LTX_WEB_DOWNLOAD_MODE:-on-demand}}"
LTX_WEB_NO_LAUNCH="${LTX_WEB_NO_LAUNCH:-0}"
LAUNCH_ARGS=()

#####################################################

while [ "$#" -gt 0 ]; do
    case "$1" in
        --env-only|--no-models|--on-demand)
            LTX_WEB_MODEL_MODE="on-demand"
            shift
            ;;
        --with-models|--prefetch-models)
            LTX_WEB_MODEL_MODE="with-models"
            shift
            ;;
        --no-launch|--setup-only)
            LTX_WEB_NO_LAUNCH="1"
            shift
            ;;
        --)
            shift
            while [ "$#" -gt 0 ]; do
                LAUNCH_ARGS+=("$1")
                shift
            done
            ;;
        *)
            LAUNCH_ARGS+=("$1")
            shift
            ;;
    esac
done

case "$LTX_WEB_MODEL_MODE" in
    env-only|on-demand|no-models|deps-only|0|false|no)
        LTX_WEB_MODEL_MODE="on-demand"
        ;;
    with-models|prefetch|prefetch-models|models|full|1|true|yes)
        LTX_WEB_MODEL_MODE="with-models"
        ;;
    *)
        echo "Unknown LTX_WEB_MODEL_MODE: $LTX_WEB_MODEL_MODE"
        echo "Expected on-demand or with-models."
        exit 1
        ;;
esac

should_prefetch_models() {
    [ "$LTX_WEB_MODEL_MODE" = "with-models" ]
}

echo "========================================"
echo "       LTX-2.3 WebUI Launcher"
echo "========================================"
echo "Model mode: ${LTX_WEB_MODEL_MODE}"
echo

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



ensure_torchaudio_abi() {
    echo "Checking torch/torchaudio ABI compatibility..."
    if python -c "import torch, torchaudio" >/dev/null 2>&1; then
        return
    fi

    TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
    echo "torchaudio import failed; reinstalling torchaudio==${TORCH_VERSION} to match torch..."
    pip install --force-reinstall --no-deps "torchaudio==${TORCH_VERSION}"

    if ! python -c "import torch, torchaudio; print('torch', torch.__version__, 'torchaudio', torchaudio.__version__)"; then
        echo
        echo "torchaudio is still not compatible with the installed torch build."
        echo "Install a matching PyTorch stack for your CUDA image, then rerun this launcher."
        echo "Current torch version: ${TORCH_VERSION}"
        exit 1
    fi
}

if [ ! -f ".deps_installed" ]; then
    echo "Installing WebUI dependencies..."
    pip install -r requirements.txt
    touch .deps_installed
fi

if ! python -c "import sentencepiece" >/dev/null 2>&1; then
    echo "Installing missing tokenizer dependency: sentencepiece..."
    pip install sentencepiece
fi

ensure_torchaudio_abi

echo "Installing local LTX-2.3 packages..."
pip install -e LTX-2/packages/ltx-core
pip install -e LTX-2/packages/ltx-pipelines

gemma_ready() {
    for file in config.json tokenizer.model preprocessor_config.json processor_config.json tokenizer_config.json; do
        if [ ! -f "models/gemma/${file}" ]; then
            return 1
        fi
    done
    grep -q "gemma3" "models/gemma/config.json" 2>/dev/null
}

if should_prefetch_models; then
    if ! gemma_ready; then
        echo
        echo "================================================"
        echo "  Gemma 3 12B QAT text encoder is required"
        echo "================================================"
        echo
        mkdir -p models/gemma
        echo "Downloading ${GEMMA_REPO}..."
        hf download "$GEMMA_REPO" --local-dir ./models/gemma || {
            echo "Gemma download failed. Try manually:"
            echo "  hf download ${GEMMA_REPO} --local-dir ./models/gemma"
        }
    fi

    if [ ! -f "$DISTILLED_CHECKPOINT" ]; then
        echo
        echo "================================================"
        echo "  Downloading LTX-2.3 22B Distilled Checkpoint"
        echo "================================================"
        echo
        mkdir -p models/checkpoints
        if [ -n "$HF_TOKEN" ]; then
            hf download "$CHECKPOINT_REPO" "$(basename "$DISTILLED_CHECKPOINT")" --local-dir ./models/checkpoints --token "$HF_TOKEN"
        else
            hf download "$CHECKPOINT_REPO" "$(basename "$DISTILLED_CHECKPOINT")" --local-dir ./models/checkpoints
        fi
    fi

    if [ ! -f "$SPATIAL_UPSCALER" ]; then
        echo
        echo "Downloading LTX-2.3 spatial upscaler..."
        mkdir -p models/upsamplers
        if [ -n "$HF_TOKEN" ]; then
            hf download "$CHECKPOINT_REPO" "$(basename "$SPATIAL_UPSCALER")" --local-dir ./models/upsamplers --token "$HF_TOKEN"
        else
            hf download "$CHECKPOINT_REPO" "$(basename "$SPATIAL_UPSCALER")" --local-dir ./models/upsamplers
        fi
    fi
else
    echo
    echo "Skipping model downloads. Required LTX models will be downloaded on first generation."
fi

echo
echo "========================================"
echo "  Environment ready."
echo "========================================"
echo
if [ "$LTX_WEB_NO_LAUNCH" = "1" ]; then
    echo "Skipping API launch because LTX_WEB_NO_LAUNCH=1."
    exit 0
fi

echo "Starting FastAPI UI at http://localhost:8000"
echo "Default setup: LTX-2.3 distilled pipeline with on-demand model downloads"
echo

python api.py "${LAUNCH_ARGS[@]}"
