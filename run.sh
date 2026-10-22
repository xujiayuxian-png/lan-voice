#!/usr/bin/env bash
set -euo pipefail
umask 077

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_PORT="${VOICE_PORT:-8788}"
TASK_HOST="${VOICE_HOST:-0.0.0.0}"
TASK_PYTHON="${VOICE_PYTHON:-$TASK_ROOT/.local/envs/speech/bin/python}"
TASK_UI_LANGUAGE="${VOICE_UI_LANGUAGE:-en}"
# Read the language flag before checking tools so early errors use it too.
TASK_EXPECT_UI_LANGUAGE=false
for TASK_ARG in "$@"; do
    if [[ "$TASK_EXPECT_UI_LANGUAGE" == true ]]; then
        TASK_UI_LANGUAGE="$TASK_ARG"
        TASK_EXPECT_UI_LANGUAGE=false
    elif [[ "$TASK_ARG" == --ui-language ]]; then
        TASK_EXPECT_UI_LANGUAGE=true
    elif [[ "$TASK_ARG" == --ui-language=* ]]; then
        TASK_UI_LANGUAGE="${TASK_ARG#--ui-language=}"
    fi
done
task_message() {
    if [[ "$TASK_UI_LANGUAGE" == zh-CN ]]; then
        echo "$2" >&2
    else
        echo "$1" >&2
    fi
}
ASR_BACKEND="${VOICE_ASR_BACKEND:-sensevoice}"
QWEN_NATIVE_MODEL_DIR="${VOICE_QWEN_NATIVE_MODEL_DIR:-$TASK_ROOT/.local/models/Qwen3-ASR-1.7B}"
if [[ "$ASR_BACKEND" == "qwen3-streaming" && -z "${VOICE_PYTHON:-}" ]]; then
    TASK_PYTHON="$TASK_ROOT/.local/envs/qwen-asr/bin/python"
fi
MODEL_DIR="${VOICE_MODEL_DIR:-$TASK_ROOT/.local/models/sensevoice}"
QWEN_MODEL_DIR="${VOICE_QWEN_MODEL_DIR:-$TASK_ROOT/.local/models/qwen3-asr-0.6b}"
QWEN_PROVIDER="${VOICE_QWEN_PROVIDER:-cpu}"
QWEN_HOTWORDS="${VOICE_QWEN_HOTWORDS:-}"
TLS_DIR="$TASK_ROOT/.local/tls"
TLS_CERT="$TLS_DIR/voice-input.crt"
TLS_KEY="$TLS_DIR/voice-input.key"
TOKEN_FILE="$TASK_ROOT/.local/token"

for TASK_TOOL in ffmpeg openssl; do
    if ! command -v "$TASK_TOOL" >/dev/null 2>&1; then
        task_message "Required command not found: $TASK_TOOL" "缺少必要命令：$TASK_TOOL"
        exit 1
    fi
done

if [[ ! -x "$TASK_PYTHON" ]]; then
    task_message "Voice Python environment not found: $TASK_PYTHON" "找不到语音识别 Python 环境：$TASK_PYTHON"
    task_message "Run: python3 scripts/setup.py environment --backend cpu (or streaming)." \
        "请运行：python3 scripts/setup.py environment --backend cpu（或 streaming）。"
    task_message "Alternatively set VOICE_PYTHON to your ASR environment." "也可用 VOICE_PYTHON 指定已有的识别环境。"
    exit 1
fi
if [[ "$ASR_BACKEND" == "sensevoice" ]]; then
    if [[ ! -f "$MODEL_DIR/model.int8.onnx" || ! -f "$MODEL_DIR/tokens.txt" ]]; then
        task_message "SenseVoice model not found under: $MODEL_DIR" "找不到 SenseVoice 模型：$MODEL_DIR"
        task_message "Run: python3 scripts/setup.py models --model sensevoice" "请运行：python3 scripts/setup.py models --model sensevoice"
        task_message "Alternatively set VOICE_MODEL_DIR to the model directory." "也可用 VOICE_MODEL_DIR 指定已有模型目录。"
        exit 1
    fi
elif [[ "$ASR_BACKEND" == "qwen3-streaming" ]]; then
    if [[ ! -f "$QWEN_NATIVE_MODEL_DIR/config.json" ||
          ! -f "$QWEN_NATIVE_MODEL_DIR/model.safetensors.index.json" ]]; then
        task_message "Native Qwen3-ASR model not found under: $QWEN_NATIVE_MODEL_DIR" "找不到原生 Qwen3-ASR 模型：$QWEN_NATIVE_MODEL_DIR"
        task_message "Run: python3 scripts/setup.py models --model qwen3-native" "请运行：python3 scripts/setup.py models --model qwen3-native"
        task_message "Alternatively set VOICE_QWEN_NATIVE_MODEL_DIR to your official model directory." "也可用 VOICE_QWEN_NATIVE_MODEL_DIR 指定已有官方模型目录。"
        exit 1
    fi
elif [[ "$ASR_BACKEND" == "qwen3" ]]; then
    if [[ ! -f "$QWEN_MODEL_DIR/conv_frontend.onnx" ||
          ! -f "$QWEN_MODEL_DIR/encoder.int8.onnx" ||
          ! -f "$QWEN_MODEL_DIR/decoder.int8.onnx" ||
          ! -d "$QWEN_MODEL_DIR/tokenizer" ]]; then
        task_message "Qwen3-ASR model not found under: $QWEN_MODEL_DIR" "找不到 Qwen3-ASR ONNX 模型：$QWEN_MODEL_DIR"
        task_message "Run: python3 scripts/setup.py models --model qwen3-onnx" "请运行：python3 scripts/setup.py models --model qwen3-onnx"
        task_message "Alternatively set VOICE_QWEN_MODEL_DIR to the extracted model directory." "也可用 VOICE_QWEN_MODEL_DIR 指定已有模型目录。"
        exit 1
    fi
else
    task_message "Unsupported VOICE_ASR_BACKEND: $ASR_BACKEND" "不支持的 VOICE_ASR_BACKEND：$ASR_BACKEND"
    exit 1
fi

mkdir -p "$TLS_DIR"
chmod 700 "$TASK_ROOT/.local" "$TLS_DIR"
if [[ ! -f "$TLS_CERT" || ! -f "$TLS_KEY" ]] ||
    ! openssl x509 -checkend 0 -noout -in "$TLS_CERT" >/dev/null 2>&1 ||
    ! "$TASK_PYTHON" - "$TLS_CERT" "$TLS_KEY" >/dev/null 2>&1 <<'PY'
import ssl
import sys
ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(*sys.argv[1:])
PY
then
    LAN_IP="$("$TASK_PYTHON" - <<'PY'
import socket
try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("192.0.2.1", 9))
        print(sock.getsockname()[0])
except OSError:
    print("127.0.0.1")
PY
)"
    (
        TLS_TEMP_DIR="$(mktemp -d "$TLS_DIR/.generate.XXXXXX")"
        trap 'rm -rf -- "$TLS_TEMP_DIR"' EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM
        if ! openssl req -x509 -newkey rsa:2048 -nodes -days 30 \
            -keyout "$TLS_TEMP_DIR/key" -out "$TLS_TEMP_DIR/cert" \
            -subj "/CN=voice-input.local" \
            -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:$LAN_IP" \
            >/dev/null 2>&1; then
            task_message "Could not generate the local HTTPS certificate." "无法生成本地 HTTPS 证书。"
            exit 1
        fi
        mv -- "$TLS_TEMP_DIR/key" "$TLS_KEY"
        mv -- "$TLS_TEMP_DIR/cert" "$TLS_CERT"
    )
fi
chmod 600 "$TLS_CERT" "$TLS_KEY"

if [[ -z "${VOICE_TOKEN:-}" ]]; then
    if [[ ! -s "$TOKEN_FILE" ]]; then
        "$TASK_PYTHON" -c 'import secrets; print(secrets.token_urlsafe(24))' > "$TOKEN_FILE"
    fi
    chmod 600 "$TOKEN_FILE"
    VOICE_TOKEN="$(< "$TOKEN_FILE")"
fi
export VOICE_TOKEN

cd "$TASK_ROOT"
exec "$TASK_PYTHON" "$TASK_ROOT/server.py" \
    --host "$TASK_HOST" \
    --port "$TASK_PORT" \
    --asr-backend "$ASR_BACKEND" \
    --model "$MODEL_DIR/model.int8.onnx" \
    --tokens "$MODEL_DIR/tokens.txt" \
    --qwen-model-dir "$QWEN_MODEL_DIR" \
    --qwen-provider "$QWEN_PROVIDER" \
    --qwen-hotwords "$QWEN_HOTWORDS" \
    --qwen-native-model-dir "$QWEN_NATIVE_MODEL_DIR" \
    --tls-cert "$TLS_CERT" \
    --tls-key "$TLS_KEY" \
    "$@"

