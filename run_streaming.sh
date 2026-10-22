#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VOICE_ASR_BACKEND=qwen3-streaming
export VOICE_PYTHON="${VOICE_PYTHON:-$TASK_ROOT/.local/envs/qwen-asr/bin/python}"
export VOICE_QWEN_NATIVE_MODEL_DIR="${VOICE_QWEN_NATIVE_MODEL_DIR:-$TASK_ROOT/.local/models/Qwen3-ASR-1.7B}"
export VOICE_QWEN_GPU_MEMORY="${VOICE_QWEN_GPU_MEMORY:-0.28}"
# Context is optional. Presets and custom vocabulary can be chosen in the UI.
export VOICE_QWEN_CONTEXT="${VOICE_QWEN_CONTEXT-}"
# Restore sentence punctuation across audio pauses when the local CPU model is
# installed. An explicit empty value keeps the original ASR punctuation.
if [[ -f "$TASK_ROOT/.local/models/punctuation/model.int8.onnx" ]]; then
    export VOICE_PUNCTUATION_MODEL="${VOICE_PUNCTUATION_MODEL-$TASK_ROOT/.local/models/punctuation/model.int8.onnx}"
fi
# Keep this service's caches separate from other GPU workloads on the host.
export HF_HOME="${HF_HOME:-$TASK_ROOT/.local/cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$TASK_ROOT/.local/cache/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$TASK_ROOT/.local/cache}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
exec bash "$TASK_ROOT/run.sh" "$@"
