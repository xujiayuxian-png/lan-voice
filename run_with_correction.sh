#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_LMS="${VOICE_LMS:-$(command -v lms || true)}"
TASK_LMS="${TASK_LMS:-$HOME/.lmstudio/bin/lms}"
TASK_PYTHON="${VOICE_PYTHON:-$TASK_ROOT/.local/envs/speech/bin/python}"
TASK_UI_LANGUAGE="${VOICE_UI_LANGUAGE:-en}"
TASK_EXPECT_LANGUAGE=false
for TASK_ARG in "$@"; do
    if [[ "$TASK_EXPECT_LANGUAGE" == true ]]; then
        TASK_UI_LANGUAGE="$TASK_ARG"
        TASK_EXPECT_LANGUAGE=false
    elif [[ "$TASK_ARG" == --ui-language ]]; then
        TASK_EXPECT_LANGUAGE=true
    elif [[ "$TASK_ARG" == --ui-language=* ]]; then
        TASK_UI_LANGUAGE="${TASK_ARG#--ui-language=}"
    fi
done
task_message() {
    if [[ "$TASK_UI_LANGUAGE" == zh-CN ]]; then echo "$2" >&2; else echo "$1" >&2; fi
}

if [[ ! -x "$TASK_PYTHON" ]]; then
    task_message "Voice Python environment not found: $TASK_PYTHON" "找不到语音识别 Python 环境：$TASK_PYTHON"
    exit 1
fi
if [[ ! -x "$TASK_LMS" ]]; then
    task_message "LM Studio CLI not found. Install lms or set VOICE_LMS." "找不到 LM Studio CLI，请安装 lms 或设置 VOICE_LMS。"
    exit 1
fi
if [[ -z "${VOICE_CORRECTION_MODEL:-}" ]]; then
    task_message "Set VOICE_CORRECTION_MODEL to your LM Studio model identifier." \
        "请用 VOICE_CORRECTION_MODEL 指定 LM Studio 中的模型标识。"
    exit 1
fi
if ! "$TASK_PYTHON" -c 'import pypinyin' >/dev/null 2>&1; then
    task_message "Install correction dependencies: $TASK_PYTHON -m pip install -r $TASK_ROOT/requirements-correction.txt" \
        "请安装纠错依赖：$TASK_PYTHON -m pip install -r $TASK_ROOT/requirements-correction.txt"
    exit 1
fi
export VOICE_CORRECTION_URL="${VOICE_CORRECTION_URL:-http://127.0.0.1:1234/v1}"
TASK_LM_PORT="$("$TASK_PYTHON" - "$VOICE_CORRECTION_URL" <<'PY'
import sys
from urllib.parse import urlsplit
url = urlsplit(sys.argv[1])
if url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"} or url.username or url.password or url.query or url.fragment:
    raise SystemExit("VOICE_CORRECTION_URL must be a local HTTP API URL")
print(url.port or 80)
PY
)"
task_model_ready() {
    "$TASK_PYTHON" - "$VOICE_CORRECTION_URL" "$VOICE_CORRECTION_MODEL" <<'PY'
import json
import sys
from urllib.request import ProxyHandler, build_opener
try:
    with build_opener(ProxyHandler({})).open(sys.argv[1].rstrip("/") + "/models", timeout=5) as response:
        models = json.load(response)["data"]
    raise SystemExit(0 if any(model["id"] == sys.argv[2] for model in models) else 1)
except Exception:
    raise SystemExit(1)
PY
}
if ! task_model_ready; then
    if [[ -z "${VOICE_LM_PATH:-}" ]]; then
        task_message "The correction model is not loaded. Load it in LM Studio, or set VOICE_LM_PATH to a downloaded model." \
            "纠错模型尚未加载，请在 LM Studio 中加载，或用 VOICE_LM_PATH 指定已下载的模型。"
        exit 1
    fi
    "$TASK_LMS" server start --bind 127.0.0.1 --port "$TASK_LM_PORT"
    # Only load the requested model. Do not unload any existing models.
    "$TASK_LMS" load "$VOICE_LM_PATH" --context-length 8192 \
        --identifier "$VOICE_CORRECTION_MODEL" -y
fi
exec bash "$TASK_ROOT/run.sh" "$@"
