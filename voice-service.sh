#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="$ROOT/.local"
PID_FILE="$RUNTIME/server.pid"
LOG_FILE="$RUNTIME/server.log"

die() { echo "voice-service: $*" >&2; exit 1; }

read_pid() {
    [[ -s "$PID_FILE" ]] || return 1
    local pid
    pid="$(<"$PID_FILE")"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "$pid"
}

owns_process() {
    local pid="$1" cmd cwd
    [[ -d "/proc/$pid" ]] || return 1
    cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null || true)"
    [[ "$cwd" == "$ROOT" ]] || return 1
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    [[ "$cmd" == *"$ROOT/server.py"* ]]
}

running_pid() {
    local pid
    pid="$(read_pid 2>/dev/null || true)"
    [[ -n "$pid" ]] && owns_process "$pid" && printf '%s\n' "$pid"
}

start() {
    local pid
    if pid="$(running_pid)"; then
        echo "Voice service is already running (PID $pid)."
        return 0
    fi
    if pid="$(read_pid 2>/dev/null || true)"; then
        echo "Removing stale PID file." >&2
        rm -f "$PID_FILE"
    fi
    mkdir -p "$RUNTIME"
    export VOICE_QWEN_CONTEXT="${VOICE_QWEN_CONTEXT-}"
    export VOICE_UI_LANGUAGE="${VOICE_UI_LANGUAGE:-zh-CN}"
    nohup bash "$ROOT/run_streaming.sh" >>"$LOG_FILE" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$PID_FILE"
    for _ in {1..60}; do
        if owns_process "$pid"; then
            echo "Voice service started (PID $pid)."
            echo "Log: $LOG_FILE"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Voice service failed to start. See $LOG_FILE." >&2
            rm -f "$PID_FILE"
            return 1
        fi
        sleep 0.5
    done
    echo "Voice service is still initializing (PID $pid). See $LOG_FILE."
}

stop() {
    local pid
    pid="$(running_pid || true)"
    if [[ -z "$pid" ]]; then
        rm -f "$PID_FILE"
        echo "Voice service is not running."
        return 0
    fi
    kill -TERM -- "-$pid" 2>/dev/null || kill "$pid"
    for _ in {1..30}; do
        if ! owns_process "$pid"; then
            rm -f "$PID_FILE"
            echo "Voice service stopped."
            return 0
        fi
        sleep 0.5
    done
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid"
    rm -f "$PID_FILE"
    echo "Voice service stopped forcefully." >&2
}

status() {
    local pid
    if pid="$(running_pid)"; then
        echo "Voice service is running (PID $pid)."
        return 0
    fi
    echo "Voice service is stopped."
    return 1
}

case "${1:-}" in
    start) start ;;
    stop) stop ;;
    restart) stop; start ;;
    status) status ;;
    *) echo "Usage: $0 {start|stop|restart|status}" >&2; exit 2 ;;
esac
