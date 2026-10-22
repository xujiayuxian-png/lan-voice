#!/usr/bin/env python3
"""Exercise the launcher, HTTPS, real ASR and dry-run paste without touching focus."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import ssl
import subprocess
import sys
import time
from http.client import HTTPSConnection
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server import DEFAULT_MODEL


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_model_dir = Path(os.environ.get("VOICE_MODEL_DIR", DEFAULT_MODEL.parent))
    parser.add_argument("--audio", type=Path, default=default_model_dir / "test_wavs/zh.wav")
    parser.add_argument("--expect-text", help="Optional phrase required in every transcription")
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error("Sample audio is missing; supply --audio /path/to/speech.wav")

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    token = secrets.token_urlsafe(24)
    environment = {
        **os.environ, "VOICE_HOST": "127.0.0.1", "VOICE_PORT": str(port),
        "VOICE_TOKEN": token, "VOICE_PYTHON": sys.executable,
    }
    # This script only talks to its own local, self-signed HTTPS server.
    context = ssl._create_unverified_context()
    process = subprocess.Popen(
        ["bash", str(ROOT / "run.sh"), "--dry-run"], cwd=ROOT, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    def request(method, path, body=None, mime_type=None, authenticated=True):
        connection = HTTPSConnection("127.0.0.1", port, context=context, timeout=15)
        headers = {"X-Voice-Token": token} if authenticated else {}
        if mime_type:
            headers["Content-Type"] = mime_type
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            return response.status, raw
        finally:
            connection.close()

    try:
        deadline = time.monotonic() + 30
        while True:
            if process.poll() is not None:
                raise RuntimeError("The voice server exited during startup")
            try:
                code, body = request("GET", "/api/status")
                if code == 200:
                    break
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError("The voice server did not become ready within 30 seconds")
            time.sleep(0.05)
        state = json.loads(body)
        assert state["tls"] and state["injector"] == "dry-run", state
        assert state["transcriber"] != "unavailable", state
        assert request("GET", "/", authenticated=False)[0] == 403
        code, page = request("GET", "/")
        assert code == 200 and b'id="record"' in page
        # An idle TLS client must not block other clients in the accept loop.
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            assert request("GET", "/api/status")[0] == 200
        print("PASS: HTTPS startup, authentication, page and concurrent TLS connections")

        source = args.audio.read_bytes()
        formats = [
            ("WAV", "audio/wav", None),
            ("WebM/Opus", "audio/webm", ["-c:a", "libopus", "-f", "webm"]),
            ("MP4/AAC", "audio/mp4", [
                "-c:a", "aac", "-f", "mp4", "-movflags", "frag_keyframe+empty_moov",
            ]),
            ("Ogg/Opus", "audio/ogg", ["-c:a", "libopus", "-f", "ogg"]),
        ]
        for label, mime_type, options in formats:
            audio = source
            if options:
                audio = subprocess.run(
                    ["ffmpeg", "-nostdin", "-v", "error", "-i", "pipe:0", *options, "pipe:1"],
                    input=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=True, timeout=30,
                ).stdout
            code, body = request("POST", "/api/transcribe", audio, mime_type)
            result = json.loads(body)
            assert code == 200 and result.get("text"), result
            if args.expect_text:
                assert args.expect_text in result["text"], result
            print(f"PASS: {label} transcription ({result['elapsed_ms']} ms)")
        code, body = request(
            "POST", "/api/inject", json.dumps({"text": "语音输入集成测试"}).encode(),
            "application/json",
        )
        assert code == 200 and json.loads(body) == {"ok": True, "dry_run": True}
        print("PASS: dry-run injection (desktop untouched)")
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    finally:
        process.terminate()
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
        if process.returncode not in (0, -15):
            print(output.replace(token, "[redacted]"), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
