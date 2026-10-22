#!/usr/bin/env python3
"""LAN voice input bridge: phone audio -> local ASR -> focused desktop window."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import secrets
import socket
import ssl
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote_to_bytes, urlsplit

try:
    from .correction import LocalCorrector
    from .punctuation import TranscriptPunctuator
    from .streaming_asr import FinalizationError, MAX_ASR_CONTEXT_CHARS
except ImportError:
    from correction import LocalCorrector
    from punctuation import TranscriptPunctuator
    from streaming_asr import FinalizationError, MAX_ASR_CONTEXT_CHARS


ROOT = Path(__file__).resolve().parent
STATIC_PAGE = ROOT / "static" / "index.html"
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_AUDIO_SECONDS = 120
MAX_TEXT_CHARS = 20_000
MAX_TEXT_BYTES = MAX_TEXT_CHARS * 12 + 1024
MAX_PARTIAL_SEQUENCE = 1_000_000
MAX_STREAM_ID_CHARS = 128
STREAM_SESSION_TTL = 5 * 60
MAX_STREAM_SESSIONS = 8
DEFAULT_MODEL = Path(
    os.environ.get(
        "VOICE_MODEL",
        str(Path(os.environ.get("VOICE_MODEL_DIR", ROOT / ".local/models/sensevoice")) / "model.int8.onnx"),
    )
)
DEFAULT_TOKENS = Path(
    os.environ.get(
        "VOICE_TOKENS",
        str(Path(os.environ.get("VOICE_MODEL_DIR", ROOT / ".local/models/sensevoice")) / "tokens.txt"),
    )
)
DEFAULT_QWEN_MODEL_DIR = Path(
    os.environ.get(
        "VOICE_QWEN_MODEL_DIR",
        str(ROOT / ".local/models/qwen3-asr-0.6b"),
    )
)
DEFAULT_QWEN_NATIVE_MODEL_DIR = ROOT / ".local" / "models" / "Qwen3-ASR-1.7B"


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def local_ipv4_addresses() -> list[str]:
    addresses: set[str] = {"127.0.0.1"}
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(item[4][0])
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 9))
            addresses.add(sock.getsockname()[0])
    except OSError:
        pass
    return sorted(addresses, key=lambda value: (value == "127.0.0.1", value))


class TranscriptionError(RuntimeError):
    """An audio decoding or model inference failure."""


def split_on_pauses(samples: Any, sample_rate: int) -> list[Any]:
    """Split at clear pauses without dropping audio around language switches."""
    import numpy as np

    frame_size = max(1, sample_rate // 50)  # 20 ms
    frame_count = len(samples) // frame_size
    if frame_count < 2:
        return [samples]
    frames = samples[:frame_count * frame_size].reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    peak = float(rms.max())
    if peak == 0:
        return [samples]
    # A relative threshold also works for quiet microphones.
    quiet = rms <= peak * 0.05
    transitions = np.flatnonzero(np.diff(np.r_[False, quiet, False])).reshape(-1, 2)
    boundaries = [0]
    for start, end in transitions:
        if start == 0 or end == frame_count:
            continue
        if (end - start) * frame_size < sample_rate * 0.5:
            continue
        midpoint = int((start + end) * frame_size // 2)
        if midpoint - boundaries[-1] >= sample_rate and len(samples) - midpoint >= sample_rate:
            boundaries.append(midpoint)
    boundaries.append(len(samples))
    return [samples[start:end] for start, end in zip(boundaries, boundaries[1:])]


class SenseVoiceTranscriber:
    """Decode browser audio with the local SenseVoice ONNX model."""

    def __init__(self, model: Path, tokens: Path, threads: int) -> None:
        if not model.is_file():
            raise FileNotFoundError(f"ASR model not found: {model}")
        if not tokens.is_file():
            raise FileNotFoundError(f"ASR tokens not found: {tokens}")

        try:
            import sherpa_onnx
            import soundfile
        except ImportError as exc:
            raise RuntimeError(
                "The selected Python environment needs sherpa-onnx and soundfile"
            ) from exc

        self._soundfile = soundfile
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model),
            tokens=str(tokens),
            num_threads=max(1, threads),
            language="auto",
            use_itn=True,
            debug=False,
        )
        self._lock = threading.Lock()
        self.model = model
        self.description = "SenseVoice local ONNX"

    def decode_audio(self, audio: bytes, mime_type: str) -> tuple[Any, int]:
        if not audio:
            raise TranscriptionError("The uploaded audio is empty")

        converted = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-protocol_whitelist",
                "pipe",
                "-i",
                "pipe:0",
                "-t",
                str(MAX_AUDIO_SECONDS + 1),
                "-f",
                "wav",
                "-ar",
                "16000",
                "-ac",
                "1",
                "pipe:1",
            ],
            input=audio,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        if converted.returncode != 0:
            detail = converted.stderr.decode("utf-8", errors="replace").strip()
            raise TranscriptionError(
                f"Could not decode {mime_type or 'audio'}"
                + (f": {detail[-300:]}" if detail else "")
            )

        try:
            samples, sample_rate = self._soundfile.read(
                io.BytesIO(converted.stdout), dtype="float32", always_2d=True
            )
        except Exception as exc:
            raise TranscriptionError("The decoded audio could not be read") from exc
        if samples.size == 0:
            raise TranscriptionError("The decoded audio is empty")
        if len(samples) > sample_rate * MAX_AUDIO_SECONDS:
            raise TranscriptionError(f"Audio duration exceeds the {MAX_AUDIO_SECONDS}s limit")

        mono = samples[:, 0]
        if samples.shape[1] > 1:
            mono = samples.mean(axis=1)

        return mono, sample_rate

    def transcribe(self, audio: bytes, mime_type: str) -> str:
        mono, sample_rate = self.decode_audio(audio, mime_type)

        texts = []
        with self._lock:
            for segment in split_on_pauses(mono, sample_rate):
                stream = self._recognizer.create_stream()
                stream.accept_waveform(sample_rate, segment)
                self._recognizer.decode_stream(stream)
                text = getattr(stream.result, "text", "").strip()
                if text:
                    texts.append(text)
        return " ".join(" ".join(texts).split())


class Qwen3ASRTranscriber(SenseVoiceTranscriber):
    """Decode audio with the sherpa-onnx Qwen3-ASR offline model.

    Qwen3-ASR is an offline recognizer in the installed sherpa-onnx runtime.
    The streaming endpoint therefore feeds it increasingly large snapshots;
    the browser replaces its draft whenever a later snapshot is returned.
    """

    def __init__(
        self,
        model_dir: Path,
        threads: int,
        provider: str = "cpu",
        hotwords: str = "",
    ) -> None:
        required = {
            "conv_frontend": model_dir / "conv_frontend.onnx",
            "encoder": model_dir / "encoder.int8.onnx",
            "decoder": model_dir / "decoder.int8.onnx",
            "tokenizer": model_dir / "tokenizer",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Qwen3-ASR model files not found: " + ", ".join(missing)
            )
        try:
            import sherpa_onnx
            import soundfile
        except ImportError as exc:
            raise RuntimeError(
                "The selected Python environment needs sherpa-onnx and soundfile"
            ) from exc

        self._soundfile = soundfile
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=str(required["conv_frontend"]),
            encoder=str(required["encoder"]),
            decoder=str(required["decoder"]),
            tokenizer=str(required["tokenizer"]),
            num_threads=max(1, threads),
            provider=provider,
            hotwords=hotwords,
            debug=False,
        )
        self._lock = threading.Lock()
        self.model = model_dir
        self.description = "Qwen3-ASR local ONNX"


class Qwen3StreamingTranscriber(SenseVoiceTranscriber):
    """Native Qwen streaming drafts, followed by sentence audio verification."""

    partial_mode = "qwen3-native"
    asr_context_configurable = True

    def __init__(
        self, model_dir: Path, gpu_memory: float = 0.28, context: str = "",
        punctuation_model: Path | None = None,
    ) -> None:
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(f"Qwen3-ASR model not found: {model_dir}")
        if not 0 < gpu_memory < 1:
            raise ValueError("Qwen GPU memory fraction must be between zero and one")
        try:
            import soundfile
            from qwen_asr import Qwen3ASRModel
            from streaming_asr import QwenStreamingSessions
        except ImportError as exc:
            raise RuntimeError(
                "Native Qwen3-ASR needs the isolated qwen-asr/vLLM environment; "
                "start with run_streaming.sh"
            ) from exc
        self._soundfile = soundfile
        self._lock = threading.Lock()
        punctuator = TranscriptPunctuator(punctuation_model) if punctuation_model else None
        engine = Qwen3ASRModel.LLM(
            model=str(model_dir), dtype="bfloat16", max_new_tokens=1024,
            gpu_memory_utilization=gpu_memory, max_model_len=4096,
            max_num_seqs=1, enforce_eager=True, enable_prefix_caching=False,
            limit_mm_per_prompt={"audio": 1}, disable_log_stats=True,
        )
        self._sessions = QwenStreamingSessions(engine, context=context, punctuator=punctuator)
        self.context = context
        self.punctuation_model = punctuator.description if punctuator else None
        self.model = model_dir
        self.description = "Qwen3-ASR-1.7B native streaming + audio finalization"

    def transcribe(
        self, audio: bytes, mime_type: str, *, context: str | None = None
    ) -> str:
        return self.transcribe_partial(
            audio, mime_type, stream_id=secrets.token_urlsafe(16),
            sequence=0, final=True, **({"context": context} if context is not None else {}),
        )["text"]

    def transcribe_partial(
        self, audio: bytes, mime_type: str, *, stream_id: str,
        sequence: int, final: bool, context: str | None = None,
    ) -> dict[str, Any]:
        # Reject a context change before decoding, including on final uploads:
        # rejecting request metadata must preserve the original live session.
        with self._lock:
            self._sessions.resolve_context(stream_id, context)
        try:
            samples, _ = self.decode_audio(audio, mime_type)
        except Exception:
            # An incomplete intermediate container can be retried with the
            # next cumulative snapshot. A failed final upload must release
            # its old native state so retries cannot exhaust session slots.
            if final:
                with self._lock:
                    self._sessions.discard(stream_id)
            raise
        with self._lock:
            return self._sessions.snapshot(
                stream_id, sequence, samples, final,
                **({"context": context} if context is not None else {}),
            )



class DryRunInjector:
    """Keep integration tests and local endpoint checks side-effect free."""

    def __init__(self) -> None:
        self.last_text = ""

    def inject(self, text: str) -> None:
        self.last_text = text

    @property
    def description(self) -> str:
        return "dry-run"


class X11PasteInjector:
    """Put text on the X11 clipboard and paste it into the focused window."""

    def __init__(self) -> None:
        self.helper = ROOT / "clipboard_paste.py"

    def inject(self, text: str) -> None:
        if not os.environ.get("DISPLAY"):
            raise RuntimeError("DISPLAY is not set; the desktop paste bridge is unavailable")
        completed = subprocess.run(
            [sys.executable, str(self.helper)],
            input=text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
            env=os.environ.copy(),
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "unknown X11 paste error"
            raise RuntimeError(detail[-500:])

    @property
    def description(self) -> str:
        return "X11 clipboard + Ctrl+V"


class VoiceApplication:
    def __init__(
        self,
        token: str,
        transcriber: SenseVoiceTranscriber | None,
        injector: DryRunInjector | X11PasteInjector,
        tls_enabled: bool,
        corrector: LocalCorrector | None = None,
        ui_language: str = "en",
    ) -> None:
        self.token = token
        self.transcriber = transcriber
        self.injector = injector
        self.tls_enabled = tls_enabled
        self.corrector = corrector
        self.ui_language = ui_language
        self.correction_lock = threading.Lock()
        self.transcribe_lock = threading.Lock()
        self.inject_lock = threading.Lock()
        self.stream_sessions: dict[str, dict[str, Any]] = {}
        self.stream_sessions_lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        return {
            "transcriber": (
                getattr(self.transcriber, "description", "local ASR")
                if self.transcriber else "unavailable"
            ),
            "model": str(self.transcriber.model) if self.transcriber else None,
            "partial_transcription": (
                getattr(self.transcriber, "partial_mode", "snapshot")
                if self.transcriber else None
            ),
            # `/api/transcribe/stream` is a local compatibility path for
            # MediaRecorder chunks. It buffers chunks and re-decodes the
            # accumulated container; it is deliberately reported as an
            # available streaming draft capability, not as token streaming.
            "streaming": bool(self.transcriber),
            "streaming_transcription": bool(self.transcriber),
            "injector": self.injector.description,
            "tls": self.tls_enabled,
            "max_audio_mb": MAX_AUDIO_BYTES // (1024 * 1024),
            "max_audio_seconds": MAX_AUDIO_SECONDS,
            "max_text_chars": MAX_TEXT_CHARS,
            "correction_model": self.corrector.model if self.corrector else None,
            "asr_context": getattr(self.transcriber, "context", ""),
            "asr_context_configurable": (
                getattr(self.transcriber, "asr_context_configurable", False) is True
            ),
            "max_asr_context_chars": MAX_ASR_CONTEXT_CHARS,
            "punctuation_model": getattr(self.transcriber, "punctuation_model", None),
            "ui_language": self.ui_language,
        }


class VoiceRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 30

    @property
    def application(self) -> VoiceApplication:
        return self.server.application  # type: ignore[attr-defined]

    def _request_token(self) -> str:
        header_token = self.headers.get("X-Voice-Token", "")
        if header_token:
            return header_token
        return parse_qs(urlsplit(self.path).query).get("token", [""])[0]

    def _authorized(self) -> bool:
        return secrets.compare_digest(
            self._request_token().encode("utf-8"), self.application.token.encode("utf-8")
        )

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, value: Any) -> None:
        self._send(status, json_bytes(value), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def _read_body(self, limit: int = MAX_AUDIO_BYTES) -> bytes | None:
        if self.headers.get("Transfer-Encoding"):
            self._error(400, "Transfer-Encoding is not supported")
            return None
        if len(self.headers.get_all("Content-Length", [])) > 1:
            self._error(400, "Multiple Content-Length headers are not supported")
            return None
        try:
            size = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            self._error(400, "Invalid Content-Length")
            return None
        if size < 0:
            self._error(411, "Content-Length is required")
            return None
        if size > limit:
            self._error(413, "Request body is too large")
            return None
        try:
            body = self.rfile.read(size)
        except TimeoutError:
            self._error(408, "Upload timed out; please try again")
            return None
        if len(body) != size:
            self._error(400, "The uploaded body is incomplete")
            return None
        return body

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
            return
        if not self._authorized():
            self._error(403, "Invalid or missing voice input token")
            return
        if path == "/api/status":
            self._send_json(200, self.application.status())
            return
        if path == "/":
            try:
                body = STATIC_PAGE.read_bytes()
            except OSError:
                self._error(500, "The voice input page is missing")
                return
            self._send(200, body, "text/html; charset=utf-8")
            return
        self._error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self._error(403, "Invalid or missing voice input token")
            return
        path = urlsplit(self.path).path
        if path == "/api/transcribe/stream":
            self._transcribe_stream()
            return
        if path == "/api/transcribe/partial":
            self._transcribe_partial()
            return
        if path == "/api/transcribe":
            self._transcribe()
            return
        if path == "/api/inject":
            self._inject()
            return
        if path == "/api/correct":
            self._correct()
            return
        self._error(404, "Not found")

    def _correct(self) -> None:
        body = self._read_body(MAX_TEXT_BYTES)
        if body is None:
            return
        try:
            payload = json.loads(body.decode("utf-8"))
            text = payload.get("text") if isinstance(payload, dict) else None
            if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
                raise ValueError("Text is required and must fit the text limit")
            text.encode("utf-8")
        except (ValueError, UnicodeError):
            self._error(400, "Invalid correction text")
            return
        if self.application.corrector is None:
            self._send_json(200, {"text": text, "original_text": text,
                                  "status": "disabled", "elapsed_ms": 0})
            return
        if not self.application.correction_lock.acquire(blocking=False):
            self._error(409, "Correction is busy; please try again")
            return
        try:
            result = self.application.corrector.correct(text)
        finally:
            self.application.correction_lock.release()
        self._send_json(200, result)

    def _transcribe(self) -> None:
        try:
            context = self._asr_context()
        except ValueError as exc:
            self._error(400, str(exc))
            return
        result = self._transcribe_audio_request(asr_context=context)
        if result is None:
            return
        text, elapsed_ms = result
        payload = {"text": text, "elapsed_ms": elapsed_ms}
        if getattr(self.application.transcriber, "asr_context_configurable", False) is True:
            payload["asr_context"] = (
                getattr(self.application.transcriber, "context", "") if context is None else context
            )
        self._send_json(200, payload)

    def _transcribe_partial(self) -> None:
        """Decode a cumulative recording and return a replaceable draft.

        MediaRecorder emits container fragments that cannot reliably be decoded
        as independent files. The browser sends the complete audio accumulated
        so far. Native Qwen feeds only new PCM into its streaming state and
        verifies complete sentences from audio; other backends re-decode the
        snapshot. Returned ``text`` replaces the previous draft.
        ``X-Voice-Stream-Sequence`` lets a client discard an out-of-order
        response and ``X-Voice-Stream-Final`` marks the last snapshot.
        """
        sequence = self._partial_sequence()
        if sequence is None:
            return
        final = self._partial_final()
        if final is None:
            return
        stream_id = self.headers.get("X-Voice-Stream-Id", "").strip()
        if len(stream_id) > 128:
            self._error(400, "Stream id is too long")
            return
        native = callable(getattr(self.application.transcriber, "transcribe_partial", None))
        if native and not stream_id:
            self._error(400, "Stream id is required")
            return
        try:
            context = self._asr_context()
        except ValueError as exc:
            self._error(400, str(exc))
            return
        result = self._transcribe_audio_request(
            snapshot_metadata=(stream_id, sequence, final) if native else None,
            asr_context=context,
        )
        if result is None:
            return
        text, elapsed_ms = result
        payload = text if isinstance(text, dict) else {"text": text, "mode": "snapshot"}
        self._send_json(
            200,
            {
                **payload,
                "partial": not final,
                "final": final,
                "sequence": sequence,
                "stream_id": stream_id or None,
                "elapsed_ms": elapsed_ms,
            },
        )

    def _transcribe_stream(self) -> None:
        """Buffer MediaRecorder chunks and return a replaceable draft.

        This compatibility endpoint accepts one consecutive container chunk per
        request.  It is useful with browsers where a chunk after the first one
        has no standalone WebM/MP4 header.  The server buffers at most one
        short-lived recording per stream id and re-decodes the accumulated
        bytes with the existing offline recognizer.  ``/api/transcribe/partial``
        remains the stateless endpoint for clients that already build a
        cumulative Blob themselves.
        """
        sequence = self._partial_sequence()
        if sequence is None:
            return
        final = self._partial_final()
        if final is None:
            return
        stream_id = self.headers.get("X-Voice-Stream-Id", "").strip()
        if not stream_id:
            self._error(400, "Stream id is required")
            return
        if len(stream_id) > MAX_STREAM_ID_CHARS:
            self._error(400, "Stream id is too long")
            return
        body = self._read_body()
        if body is None:
            return

        # Acquire the recognizer lock before changing session state. This keeps
        # a second browser request from consuming a sequence while the previous
        # snapshot is still being decoded.
        if self.application.transcriber is None:
            self._error(503, "The local transcription model is unavailable")
            return
        if not self.application.transcribe_lock.acquire(blocking=False):
            self._error(409, "Transcription is busy; please try again")
            return

        previous: dict[str, Any] | None = None
        previous_audio = b""
        combined = b""
        mime_type = self.headers.get("Content-Type", "application/octet-stream")
        try:
            now = time.monotonic()
            with self.application.stream_sessions_lock:
                stale = [
                    key for key, value in self.application.stream_sessions.items()
                    if now - value["updated_at"] > STREAM_SESSION_TTL
                ]
                for key in stale:
                    self.application.stream_sessions.pop(key, None)
                previous = self.application.stream_sessions.get(stream_id)
                if previous is None:
                    if sequence != 0:
                        self._error(409, "Stream sequence must start at zero")
                        return
                    if len(self.application.stream_sessions) >= MAX_STREAM_SESSIONS:
                        self._error(429, "Too many active transcription streams")
                        return
                    previous_audio = b""
                    session = {
                        "audio": b"",
                        "next_sequence": 0,
                        "mime_type": mime_type,
                        "updated_at": now,
                    }
                    self.application.stream_sessions[stream_id] = session
                    previous = None
                else:
                    expected = previous["next_sequence"]
                    if sequence != expected:
                        self._error(409, f"Expected stream sequence {expected}")
                        return
                    previous_audio = previous["audio"]
                    session = previous
                    mime_type = previous["mime_type"]
                combined = previous_audio + body
                if len(combined) > MAX_AUDIO_BYTES:
                    self._error(413, "Stream audio is too large")
                    return
                session["audio"] = combined
                session["next_sequence"] = sequence + 1
                session["updated_at"] = now

            result = self._transcribe_audio_request(
                audio=combined, mime_type=mime_type, lock_already_acquired=True
            )
            if result is None:
                # A transient decoder failure must leave the same sequence
                # retryable instead of silently dropping that chunk.
                with self.application.stream_sessions_lock:
                    if previous is None:
                        self.application.stream_sessions.pop(stream_id, None)
                    else:
                        current = self.application.stream_sessions.get(stream_id)
                        if current is not None:
                            current["audio"] = previous_audio
                            current["next_sequence"] = sequence
                return
            text, elapsed_ms = result
            if final:
                with self.application.stream_sessions_lock:
                    self.application.stream_sessions.pop(stream_id, None)
            self._send_json(
                200,
                {
                    "text": text,
                    "partial": not final,
                    "final": final,
                    "sequence": sequence,
                    "stream_id": stream_id,
                    "mode": "stream-buffer",
                    "elapsed_ms": elapsed_ms,
                },
            )
        finally:
            self.application.transcribe_lock.release()

    def _partial_sequence(self) -> int | None:
        raw = self.headers.get("X-Voice-Stream-Seq")
        if raw is None:
            raw = self.headers.get("X-Voice-Stream-Sequence", "0")
        raw = raw.strip()
        try:
            sequence = int(raw)
        except ValueError:
            self._error(400, "Invalid stream sequence")
            return None
        if sequence < 0 or sequence > MAX_PARTIAL_SEQUENCE:
            self._error(400, "Invalid stream sequence")
            return None
        return sequence

    def _partial_final(self) -> bool | None:
        raw = self.headers.get("X-Voice-Stream-Final", "false").strip().lower()
        if raw in {"", "0", "false", "no"}:
            return False
        if raw in {"1", "true", "yes"}:
            return True
        self._error(400, "Invalid stream final flag")
        return None

    def _asr_context(self) -> str | None:
        values = self.headers.get_all("X-Voice-ASR-Context", [])
        if not values:
            return None
        if len(values) != 1:
            raise ValueError("Multiple ASR context headers are not supported")
        if getattr(self.application.transcriber, "asr_context_configurable", False) is not True:
            raise ValueError("This ASR backend does not support per-recording context")
        encoded = values[0]
        # Four UTF-8 bytes, each encoded as %XX, is the largest valid character.
        if len(encoded) > MAX_ASR_CONTEXT_CHARS * 12:
            raise ValueError(f"ASR context must contain at most {MAX_ASR_CONTEXT_CHARS} characters")
        if re.fullmatch(r"(?:[A-Za-z0-9_.!~*'()-]|%[0-9A-Fa-f]{2})*", encoded) is None:
            raise ValueError("ASR context must use encodeURIComponent UTF-8 encoding")
        try:
            context = unquote_to_bytes(encoded).decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ValueError("ASR context must be valid UTF-8 text") from exc
        if len(context) > MAX_ASR_CONTEXT_CHARS:
            raise ValueError(f"ASR context must contain at most {MAX_ASR_CONTEXT_CHARS} characters")
        return context

    def _transcribe_audio_request(
        self,
        audio: bytes | None = None,
        mime_type: str | None = None,
        lock_already_acquired: bool = False,
        snapshot_metadata: tuple[str, int, bool] | None = None,
        asr_context: str | None = None,
    ) -> tuple[str | dict[str, Any], int] | None:
        if self.application.transcriber is None:
            self._error(503, "The local transcription model is unavailable")
            return None
        if audio is None:
            audio = self._read_body()
            if audio is None:
                return None
        if not audio:
            self._error(400, "The uploaded audio is empty")
            return None
        acquired = lock_already_acquired
        if not acquired:
            if not self.application.transcribe_lock.acquire(blocking=False):
                self._error(409, "Transcription is busy; please try again")
                return None
            acquired = True
        started = time.perf_counter()
        try:
            content_type = mime_type or self.headers.get("Content-Type", "application/octet-stream")
            context_options = {} if asr_context is None else {"context": asr_context}
            if snapshot_metadata is None:
                text = self.application.transcriber.transcribe(audio, content_type, **context_options)
            else:
                stream_id, sequence, final = snapshot_metadata
                text = self.application.transcriber.transcribe_partial(
                    audio, content_type, stream_id=stream_id,
                    sequence=sequence, final=final, **context_options,
                )
        except subprocess.TimeoutExpired:
            self._error(504, "Audio decoding timed out")
            return None
        except OSError:
            self._error(503, "Audio decoder unavailable; check that ffmpeg is installed")
            return None
        except FinalizationError as exc:
            self._send_json(422, {
                "error": "The final audio verification failed. Review the preserved draft before sending.",
                "error_code": "finalization_failed",
                "draft_text": exc.draft_text,
            })
            return None
        except TranscriptionError as exc:
            self._error(422, str(exc))
            return None
        except RuntimeError as exc:
            # Only decoding errors can resolve when more container bytes
            # arrive. A native engine failure must reach the browser now.
            self._error(503, str(exc))
            return None
        except ValueError as exc:
            self._error(400, str(exc))
            return None
        finally:
            if acquired and not lock_already_acquired:
                self.application.transcribe_lock.release()
        return text, round((time.perf_counter() - started) * 1000)

    def _inject(self) -> None:
        body = self._read_body(MAX_TEXT_BYTES)
        if body is None:
            return
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(400, "Invalid JSON")
            return
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            self._error(400, "Text is required")
            return
        if len(text) > MAX_TEXT_CHARS:
            self._error(413, "Text is too long")
            return
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            self._error(400, "Text contains invalid Unicode")
            return
        if not self.application.inject_lock.acquire(blocking=False):
            self._error(409, "Desktop paste is busy; please try again")
            return
        try:
            self.application.injector.inject(text)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            self._error(503, str(exc))
            return
        finally:
            self.application.inject_lock.release()
        self._send_json(200, {"ok": True, "dry_run": isinstance(self.application.injector, DryRunInjector)})

    def log_message(self, format: str, *args: Any) -> None:
        # The base handler also logs malformed requests and timeouts here.
        # Only log access metadata, never raw request lines (which contain tokens).
        path = urlsplit(getattr(self, "path", "")).path
        code = args[1] if format == '"%s" %s %s' and len(args) > 1 else "error"
        print(f"{self.log_date_time_string()} {getattr(self, 'command', '')} {path!r} {code}", flush=True)


class VoiceHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], application: VoiceApplication) -> None:
        super().__init__(address, VoiceRequestHandler)
        self.application = application


def build_tls_context(cert: Path, key: Path) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=cert, keyfile=key)
    return context


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--token", default=os.environ.get("VOICE_TOKEN", ""))
    parser.add_argument(
        "--ui-language", choices=("en", "zh-CN"),
        default=os.environ.get("VOICE_UI_LANGUAGE", "en"),
        help="Startup and default page language (VOICE_UI_LANGUAGE; default: en)",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--asr-backend",
        choices=("sensevoice", "qwen3", "qwen3-streaming"),
        default=os.environ.get("VOICE_ASR_BACKEND", "sensevoice"),
        help="sensevoice/qwen3: ONNX snapshots; qwen3-streaming: native Qwen3-ASR-1.7B",
    )
    parser.add_argument(
        "--qwen-model-dir",
        type=Path,
        default=DEFAULT_QWEN_MODEL_DIR,
        help="Directory containing the sherpa-onnx Qwen3-ASR model files",
    )
    parser.add_argument(
        "--qwen-provider",
        default=os.environ.get("VOICE_QWEN_PROVIDER", "cpu"),
        choices=("cpu", "cuda"),
    )
    parser.add_argument(
        "--qwen-hotwords",
        default=os.environ.get("VOICE_QWEN_HOTWORDS", ""),
        help="Optional comma-separated Qwen3-ASR hotwords",
    )
    parser.add_argument(
        "--qwen-native-model-dir", type=Path,
        default=os.environ.get("VOICE_QWEN_NATIVE_MODEL_DIR", str(DEFAULT_QWEN_NATIVE_MODEL_DIR)),
    )
    parser.add_argument(
        "--qwen-gpu-memory", type=float,
        default=float(os.environ.get("VOICE_QWEN_GPU_MEMORY", "0.28")),
        help="Fraction of total GPU memory reserved for native Qwen (default 0.28)",
    )
    parser.add_argument("--qwen-context", default=os.environ.get("VOICE_QWEN_CONTEXT", ""))
    parser.add_argument(
        "--punctuation-model", type=Path,
        default=os.environ.get("VOICE_PUNCTUATION_MODEL") or None,
        help="Optional local CT-Transformer ONNX model for Chinese sentence punctuation",
    )
    parser.add_argument("--tls-cert", type=Path)
    parser.add_argument("--tls-key", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Do not paste into the desktop")
    parser.add_argument("--correction-model", default=os.environ.get("VOICE_CORRECTION_MODEL", ""))
    parser.add_argument("--correction-url", default=os.environ.get(
        "VOICE_CORRECTION_URL", "http://127.0.0.1:1234/v1"))
    parser.add_argument(
        "--allow-no-model",
        action="store_true",
        help="Start the page even if the local model cannot be loaded",
    )
    args = parser.parse_args(argv)
    if args.ui_language not in {"en", "zh-CN"}:
        parser.error("VOICE_UI_LANGUAGE must be en or zh-CN")
    return args


def startup_message(language: str, english: str, chinese: str) -> str:
    """Keep command-line messages independent of the browser's translation."""
    return chinese if language == "zh-CN" else english


def main() -> int:
    args = parse_args()
    message = lambda english, chinese: startup_message(args.ui_language, english, chinese)
    token = args.token or secrets.token_urlsafe(12)
    if args.tls_cert and not args.tls_key or args.tls_key and not args.tls_cert:
        raise SystemExit(message(
            "--tls-cert and --tls-key must be supplied together",
            "--tls-cert 和 --tls-key 必须同时提供",
        ))

    transcriber: SenseVoiceTranscriber | None = None
    try:
        if args.asr_backend == "qwen3-streaming":
            transcriber = Qwen3StreamingTranscriber(
                args.qwen_native_model_dir, args.qwen_gpu_memory, args.qwen_context,
                punctuation_model=args.punctuation_model,
            )
        elif args.asr_backend == "qwen3":
            transcriber = Qwen3ASRTranscriber(
                args.qwen_model_dir,
                args.threads,
                provider=args.qwen_provider,
                hotwords=args.qwen_hotwords,
            )
        else:
            transcriber = SenseVoiceTranscriber(args.model, args.tokens, args.threads)
    except Exception as exc:
        if not args.allow_no_model:
            raise SystemExit(message(
                f"Could not load local transcription model: {exc}",
                f"无法加载本地语音识别模型：{exc}",
            )) from exc
        print(message(
            f"Warning: local transcription unavailable: {exc}",
            f"提示：本地语音识别暂不可用：{exc}",
        ), file=sys.stderr)

    injector = DryRunInjector() if args.dry_run else X11PasteInjector()
    application = VoiceApplication(
        token=token,
        transcriber=transcriber,
        injector=injector,
        tls_enabled=bool(args.tls_cert),
        corrector=LocalCorrector(args.correction_model, args.correction_url)
        if args.correction_model else None,
        ui_language=args.ui_language,
    )
    server = VoiceHTTPServer((args.host, args.port), application)
    if args.tls_cert and args.tls_key:
        server.socket = build_tls_context(args.tls_cert, args.tls_key).wrap_socket(
            server.socket, server_side=True, do_handshake_on_connect=False
        )

    scheme = "https" if args.tls_cert else "http"
    print(message("Voice input server is running.", "语音输入服务已启动。"), flush=True)
    print(message("Desktop: ", "电脑访问：") + f"{scheme}://127.0.0.1:{args.port}/?token={quote(token)}", flush=True)
    print(message("Phone URLs:", "手机访问地址："), flush=True)
    for address in local_ipv4_addresses():
        print(f"  {scheme}://{address}:{args.port}/?token={quote(token)}", flush=True)
    print(message("ASR: ", "语音识别：") + application.status()['transcriber'], flush=True)
    print(message("Paste: ", "粘贴方式：") + application.status()['injector'], flush=True)
    if args.tls_cert:
        print(message(
            "The first phone visit may require accepting the local certificate warning.",
            "手机首次访问时，可能需要接受本地 HTTPS 证书提示。",
        ), flush=True)
    print(message(
        "Focus the destination input field on the desktop before pressing send.",
        "发送前，请让电脑上接收文字的输入框保持焦点。",
    ), flush=True)

    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print(message("Stopping voice input server.", "正在关闭语音输入服务。"), flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

