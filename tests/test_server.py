from contextlib import redirect_stdout
import io
import json
import os
import socket
import subprocess
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch
from urllib.parse import quote

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import (
    MAX_ASR_CONTEXT_CHARS, Qwen3StreamingTranscriber, TranscriptionError,
    VoiceApplication, VoiceHTTPServer, VoiceRequestHandler,
)
from streaming_asr import QwenStreamingSessions
import server as voice_server


class FakeTranscriber:
    model = "fake-model"

    def transcribe(self, audio, mime_type):
        assert audio == b"fake-audio"
        assert mime_type == "audio/webm"
        return "本地转写结果"


class FakeInjector:
    description = "fake-injector"

    def __init__(self):
        self.values = []

    def inject(self, text):
        self.values.append(text)


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.injector = FakeInjector()
        app = VoiceApplication("test-token", FakeTranscriber(), self.injector, False)
        self.server = VoiceHTTPServer(("127.0.0.1", 0), app)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method, path, body=b"", headers=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=2)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def native_transcriber(self, context="service default"):
        native = Qwen3StreamingTranscriber.__new__(Qwen3StreamingTranscriber)
        native._lock = threading.Lock()
        native.context = context
        native.model = "cpu-test-qwen"
        native.description = "CPU Qwen SDK stub"
        native.decode_audio = Mock(return_value=(np.full(16000, 0.1, dtype=np.float32), 16000))
        engine = Mock()
        engine.init_streaming_state.side_effect = lambda **kwargs: SimpleNamespace(text="")
        engine.streaming_transcribe.side_effect = lambda samples, state: setattr(state, "text", "draft")
        engine.transcribe.side_effect = lambda audio, context: [SimpleNamespace(text="confirmed")]
        native._sessions = QwenStreamingSessions(engine, context=context)
        self.server.application.transcriber = native
        return native, engine

    def test_requires_token(self):
        status, body = self.request("GET", "/")
        self.assertEqual(status, 403)
        self.assertIn(b"token", body)

    def test_status_and_transcribe(self):
        headers = {"X-Voice-Token": "test-token"}
        status, body = self.request("GET", "/api/status", headers=headers)
        self.assertEqual(status, 200)
        status_data = json.loads(body)
        self.assertEqual(status_data["injector"], "fake-injector")
        self.assertEqual(status_data["partial_transcription"], "snapshot")
        self.assertTrue(status_data["streaming"])
        self.assertTrue(status_data["streaming_transcription"])
        self.assertEqual(status_data["asr_context"], "")
        self.assertFalse(status_data["asr_context_configurable"])
        self.assertEqual(status_data["max_asr_context_chars"], 500)
        self.assertEqual(status_data["ui_language"], "en")

        status, body = self.request(
            "POST",
            "/api/transcribe",
            body=b"fake-audio",
            headers={**headers, "Content-Type": "audio/webm"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["text"], "本地转写结果")

    def test_partial_transcription_returns_replaceable_snapshot(self):
        headers = {
            "X-Voice-Token": "test-token",
            "Content-Type": "audio/webm",
            "X-Voice-Stream-Id": "recording-1",
            "X-Voice-Stream-Sequence": "7",
            "X-Voice-Stream-Final": "false",
        }
        status, body = self.request(
            "POST", "/api/transcribe/partial", b"fake-audio", headers
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(body),
            {
                "text": "本地转写结果",
                "partial": True,
                "final": False,
                "sequence": 7,
                "stream_id": "recording-1",
                "mode": "snapshot",
                "elapsed_ms": ANY,
            },
        )

    def test_partial_transcription_accepts_final_aliases(self):
        for value in ("1", "true", "yes"):
            with self.subTest(value=value):
                status, body = self.request(
                    "POST",
                    "/api/transcribe/partial",
                    b"fake-audio",
                    {
                        "X-Voice-Token": "test-token",
                        "Content-Type": "audio/webm",
                        "X-Voice-Stream-Final": value,
                    },
                )
                self.assertEqual(status, 200)
                payload = json.loads(body)
                self.assertTrue(payload["final"])
                self.assertFalse(payload["partial"])

    def test_native_partial_preserves_audio_finalization_and_skipped_sequence(self):
        from unittest.mock import Mock
        native = Mock(
            model="qwen-1.7b", description="Qwen native", partial_mode="qwen3-native",
            context="产品发布会议。", punctuation_model="CPU punctuation",
        )
        native.transcribe_partial.return_value = {
            "text": "已确认。 New draft.", "confirmed_text": "已确认。",
            "draft_text": "New draft.", "mode": "qwen3-native", "final": False,
        }
        self.server.application.transcriber = native
        headers = {
            "X-Voice-Token": "test-token", "Content-Type": "audio/webm",
            "X-Voice-Stream-Id": "native-recording", "X-Voice-Stream-Sequence": "7",
        }
        status, body = self.request("POST", "/api/transcribe/partial", b"cumulative", headers)
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["mode"], "qwen3-native")
        self.assertEqual(payload["confirmed_text"], "已确认。")
        self.assertEqual(payload["draft_text"], "New draft.")
        self.assertTrue(payload["partial"])
        native.transcribe_partial.assert_called_once_with(
            b"cumulative", "audio/webm", stream_id="native-recording", sequence=7, final=False
        )
        native.transcribe.assert_not_called()

        status, body = self.request("GET", "/api/status", headers=headers)
        self.assertEqual(json.loads(body)["partial_transcription"], "qwen3-native")
        self.assertEqual(json.loads(body)["asr_context"], "产品发布会议。")
        self.assertEqual(json.loads(body)["punctuation_model"], "CPU punctuation")

    def test_native_partial_requires_id_and_releases_lock_after_engine_error(self):
        from unittest.mock import Mock
        native = Mock(model="qwen-1.7b")
        self.server.application.transcriber = native
        headers = {"X-Voice-Token": "test-token", "Content-Type": "audio/webm"}
        status, _ = self.request("POST", "/api/transcribe/partial", b"audio", headers)
        self.assertEqual(status, 400)
        native.transcribe_partial.assert_not_called()

        headers["X-Voice-Stream-Id"] = "native-recording"
        native.transcribe_partial.side_effect = RuntimeError("engine failed")
        status, _ = self.request("POST", "/api/transcribe/partial", b"audio", headers)
        self.assertEqual(status, 503)
        self.assertFalse(self.server.application.transcribe_lock.locked())
        native.transcribe_partial.side_effect = None
        native.transcribe_partial.return_value = {"text": "Recovered", "mode": "qwen3-native"}
        headers["X-Voice-Stream-Final"] = "true"
        status, body = self.request("POST", "/api/transcribe/partial", b"audio", headers)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["text"], "Recovered")
        self.assertTrue(json.loads(body)["final"])

    def test_native_context_status_and_unicode_overrides_on_both_audio_endpoints(self):
        native, engine = self.native_transcriber(context="会议记录默认背景")
        common = {"X-Voice-Token": "test-token", "Content-Type": "audio/webm"}
        status, body = self.request("GET", "/api/status", headers=common)
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertTrue(payload["asr_context_configurable"])
        self.assertEqual(payload["asr_context"], "会议记录默认背景")
        self.assertEqual(payload["max_asr_context_chars"], MAX_ASR_CONTEXT_CHARS)
        for path in ("/api/transcribe", "/api/transcribe/partial"):
            for context in (None, "", "日常生活和聊天", "Python testing", "会议 action items, 0.25 GB"):
                with self.subTest(path=path, context=context):
                    headers = {
                        **common, "X-Voice-Stream-Id": "new-recording",
                        "X-Voice-Stream-Final": "true",
                    }
                    if context is not None:
                        headers["X-Voice-ASR-Context"] = quote(context, safe="-_.!~*'()")
                    status, body = self.request("POST", path, b"audio", headers)
                    self.assertEqual(status, 200, body)
                    effective = native.context if context is None else context
                    self.assertEqual(json.loads(body)["asr_context"], effective)
                    self.assertEqual(engine.transcribe.call_args.kwargs["context"], effective)
                    self.assertEqual(native.context, "会议记录默认背景")
                    self.assertEqual(native._sessions.context, native.context)
                    self.assertEqual(native._sessions.sessions, {})

    def test_context_header_rejects_invalid_encoding_before_audio_decode(self):
        native, engine = self.native_transcriber()
        common = {
            "X-Voice-Token": "test-token", "Content-Type": "audio/webm",
            "X-Voice-Stream-Id": "invalid-recording",
        }
        for path in ("/api/transcribe", "/api/transcribe/partial"):
            for encoded in (
                "%", "%GG", "%E4%BD", "%FF", "%C0%AF", "%ED%A0%80",
                "%F4%90%80%80", "literal space", "a+b", "é", "a/b", "a,b",
            ):
                with self.subTest(path=path, context=encoded):
                    status, _ = self.request("POST", path, b"audio", {
                        **common, "X-Voice-ASR-Context": encoded,
                    })
                    self.assertEqual(status, 400)
        native.decode_audio.assert_not_called()
        engine.transcribe.assert_not_called()
        self.assertEqual(native._sessions.sessions, {})
        self.assertFalse(self.server.application.transcribe_lock.locked())

    def test_context_limit_counts_unicode_characters_and_rejects_oversized_header_before_decoding(self):
        native, _ = self.native_transcriber()
        common = {"X-Voice-Token": "test-token", "Content-Type": "audio/webm"}
        for character in ("a", "中", "😀"):
            with self.subTest(character=character):
                context = character * MAX_ASR_CONTEXT_CHARS
                status, body = self.request("POST", "/api/transcribe", b"audio", {
                    **common, "X-Voice-ASR-Context": quote(context, safe="-_.!~*'()"),
                })
                self.assertEqual(status, 200, body)
                self.assertEqual(json.loads(body)["asr_context"], context)
                native.decode_audio.reset_mock()
                status, _ = self.request("POST", "/api/transcribe", b"audio", {
                    **common, "X-Voice-ASR-Context": quote(context + character, safe="-_.!~*'()"),
                })
                self.assertEqual(status, 400)
                native.decode_audio.assert_not_called()
        with patch("server.unquote_to_bytes") as decode_context:
            status, _ = self.request("POST", "/api/transcribe", b"audio", {
                **common, "X-Voice-ASR-Context": "x" * (MAX_ASR_CONTEXT_CHARS * 12 + 1),
            })
            self.assertEqual(status, 400)
            decode_context.assert_not_called()

    def test_unsupported_backend_rejects_context_including_explicit_off(self):
        for path in ("/api/transcribe", "/api/transcribe/partial"):
            for context in ("", "release planning"):
                with self.subTest(path=path, context=context), patch.object(
                    self.server.application.transcriber, "transcribe"
                ) as transcribe:
                    status, _ = self.request("POST", path, b"fake-audio", {
                        "X-Voice-Token": "test-token", "Content-Type": "audio/webm",
                        "X-Voice-ASR-Context": context,
                    })
                    self.assertEqual(status, 400)
                    transcribe.assert_not_called()

    def test_duplicate_context_headers_are_rejected_before_audio_decode(self):
        native, _ = self.native_transcriber()
        connection = HTTPConnection("127.0.0.1", self.port, timeout=2)
        try:
            connection.putrequest("POST", "/api/transcribe")
            connection.putheader("X-Voice-Token", "test-token")
            connection.putheader("Content-Length", "5")
            connection.putheader("X-Voice-ASR-Context", "one")
            connection.putheader("X-Voice-ASR-Context", "two")
            connection.endheaders(b"audio")
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            response.read()
        finally:
            connection.close()
        native.decode_audio.assert_not_called()

    def test_native_recording_context_isolation_and_rejected_change_can_continue(self):
        native, engine = self.native_transcriber()
        common = {"X-Voice-Token": "test-token", "Content-Type": "audio/webm"}
        for stream_id, context in (("first", "release planning"), ("second", "daily chat")):
            status, body = self.request("POST", "/api/transcribe/partial", b"audio", {
                **common, "X-Voice-Stream-Id": stream_id,
                "X-Voice-ASR-Context": quote(context, safe="-_.!~*'()"),
            })
            self.assertEqual(status, 200, body)
            self.assertEqual(json.loads(body)["asr_context"], context)
        first = native._sessions.sessions["first"]
        calls_before_rejection = native.decode_audio.call_count
        status, _ = self.request("POST", "/api/transcribe/partial", b"invalid final", {
            **common, "X-Voice-Stream-Id": "first", "X-Voice-Stream-Sequence": "1",
            "X-Voice-Stream-Final": "true", "X-Voice-ASR-Context": "",
        })
        self.assertEqual(status, 400)
        self.assertEqual(native.decode_audio.call_count, calls_before_rejection)
        self.assertIs(native._sessions.sessions["first"], first)
        self.assertEqual(first.sequence, 0)
        self.assertFalse(self.server.application.transcribe_lock.locked())
        for stream_id, expected in (("first", "release planning"), ("second", "daily chat")):
            status, body = self.request("POST", "/api/transcribe/partial", b"audio", {
                **common, "X-Voice-Stream-Id": stream_id, "X-Voice-Stream-Sequence": "1",
                "X-Voice-Stream-Final": "true",
            })
            self.assertEqual(status, 200, body)
            self.assertEqual(json.loads(body)["asr_context"], expected)
            self.assertEqual(engine.transcribe.call_args.kwargs["context"], expected)
        self.assertTrue(all(call.kwargs["context"] == "" for call in engine.init_streaming_state.call_args_list))
        self.assertEqual(native._sessions.sessions, {})
        self.assertEqual(native.context, "service default")

    def test_native_final_decode_failure_releases_context_without_affecting_another_session(self):
        native, _ = self.native_transcriber()
        common = {"X-Voice-Token": "test-token", "Content-Type": "audio/webm"}
        for stream_id in ("failed", "live"):
            status, _ = self.request("POST", "/api/transcribe/partial", b"audio", {
                **common, "X-Voice-Stream-Id": stream_id, "X-Voice-ASR-Context": stream_id,
            })
            self.assertEqual(status, 200)
        native.decode_audio.side_effect = TranscriptionError("Invalid final WebM")
        status, _ = self.request("POST", "/api/transcribe/partial", b"audio", {
            **common, "X-Voice-Stream-Id": "failed", "X-Voice-Stream-Sequence": "1",
            "X-Voice-Stream-Final": "true",
        })
        self.assertEqual(status, 422)
        self.assertNotIn("failed", native._sessions.sessions)
        self.assertEqual(native._sessions.sessions["live"].context, "live")
        self.assertFalse(native._lock.locked())
        self.assertFalse(self.server.application.transcribe_lock.locked())
        native.decode_audio.side_effect = None
        status, body = self.request("POST", "/api/transcribe/partial", b"audio", {
            **common, "X-Voice-Stream-Id": "failed", "X-Voice-Stream-Final": "true",
            "X-Voice-ASR-Context": "replacement",
        })
        self.assertEqual(status, 200, body)
        self.assertEqual(json.loads(body)["asr_context"], "replacement")

    def test_unconfirmed_native_final_returns_draft_and_releases_request_lock(self):
        from unittest.mock import Mock
        from streaming_asr import FinalizationError

        native = Mock(model="qwen-1.7b")
        native.transcribe_partial.side_effect = FinalizationError("已确认前文。 未确认的最后一句")
        self.server.application.transcriber = native
        headers = {
            "X-Voice-Token": "test-token", "Content-Type": "audio/webm",
            "X-Voice-Stream-Id": "recording", "X-Voice-Stream-Final": "true",
        }
        status, body = self.request("POST", "/api/transcribe/partial", b"audio", headers)
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(body)["error_code"], "finalization_failed")
        self.assertTrue(json.loads(body)["error"].isascii())
        self.assertEqual(json.loads(body)["draft_text"], "已确认前文。 未确认的最后一句")
        self.assertFalse(self.server.application.transcribe_lock.locked())

    def test_partial_transcription_rejects_invalid_metadata(self):
        common = {
            "X-Voice-Token": "test-token",
            "Content-Type": "audio/webm",
        }
        for header, value in (
            ("X-Voice-Stream-Sequence", "abc"),
            ("X-Voice-Stream-Sequence", "-1"),
            ("X-Voice-Stream-Final", "maybe"),
        ):
            with self.subTest(header=header, value=value):
                status, _ = self.request(
                    "POST",
                    "/api/transcribe/partial",
                    b"fake-audio",
                    {**common, header: value},
                )
                self.assertEqual(status, 400)

        status, _ = self.request(
            "POST",
            "/api/transcribe/partial",
            b"fake-audio",
            {**common, "X-Voice-Stream-Id": "x" * 129},
        )
        self.assertEqual(status, 400)

    def test_stream_transcription_accumulates_media_recorder_chunks(self):
        seen = []

        def transcribe(audio, mime_type):
            seen.append((audio, mime_type))
            return f"草稿 {len(seen)}"

        with patch.object(
            self.server.application.transcriber, "transcribe", side_effect=transcribe
        ):
            common = {
                "X-Voice-Token": "test-token",
                "Content-Type": "audio/webm",
                "X-Voice-Stream-Id": "recording-2",
            }
            status, body = self.request(
                "POST", "/api/transcribe/stream", b"header-and-first", {
                    **common, "X-Voice-Stream-Seq": "0",
                }
            )
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["text"], "草稿 1")
            status, body = self.request(
                "POST", "/api/transcribe/stream", b"-second", {
                    **common, "X-Voice-Stream-Seq": "1",
                }
            )
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["text"], "草稿 2")
            status, body = self.request(
                "POST", "/api/transcribe/stream", b"-last", {
                    **common,
                    "X-Voice-Stream-Seq": "2",
                    "X-Voice-Stream-Final": "true",
                }
            )
            self.assertEqual(status, 200)
            payload = json.loads(body)
            self.assertTrue(payload["final"])
            self.assertEqual(payload["mode"], "stream-buffer")

        self.assertEqual(
            seen,
            [
                (b"header-and-first", "audio/webm"),
                (b"header-and-first-second", "audio/webm"),
                (b"header-and-first-second-last", "audio/webm"),
            ],
        )

    def test_stream_transcription_rejects_missing_or_out_of_order_sequence(self):
        common = {
            "X-Voice-Token": "test-token",
            "Content-Type": "audio/webm",
            "X-Voice-Stream-Id": "recording-3",
        }
        status, _ = self.request(
            "POST", "/api/transcribe/stream", b"chunk", {
                **common, "X-Voice-Stream-Seq": "1",
            }
        )
        self.assertEqual(status, 409)
        status, _ = self.request(
            "POST", "/api/transcribe/stream", b"chunk", {
                "X-Voice-Token": "test-token", "Content-Type": "audio/webm",
            }
        )
        self.assertEqual(status, 400)

    def test_inject(self):
        status, body = self.request(
            "POST",
            "/api/inject",
            body=json.dumps({"text": "送入当前输入框"}).encode(),
            headers={
                "X-Voice-Token": "test-token",
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(self.injector.values, ["送入当前输入框"])

    def test_non_ascii_token_is_rejected_without_disconnect(self):
        status, _ = self.request("GET", "/?token=%E4%B8%AD%E6%96%87")
        self.assertEqual(status, 403)

    def test_auth_is_required_for_both_mutating_endpoints(self):
        for path in (
            "/api/transcribe", "/api/transcribe/partial", "/api/transcribe/stream",
            "/api/inject", "/api/correct",
        ):
            with self.subTest(path=path):
                status, _ = self.request("POST", path, b"{}")
                self.assertEqual(status, 403)
        self.assertEqual(self.injector.values, [])

    def test_correction_disabled_enabled_and_busy(self):
        from unittest.mock import Mock
        headers = {"X-Voice-Token": "test-token"}
        body = json.dumps({"text": "检查这段代马"}).encode()
        status, data = self.request("POST", "/api/correct", body, headers)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["status"], "disabled")
        corrector = Mock(model="test-correction")
        corrector.correct.return_value = {"text": "检查这段代码", "status": "corrected"}
        self.server.application.corrector = corrector
        status, data = self.request("POST", "/api/correct", body, headers)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["text"], "检查这段代码")
        corrector.correct.assert_called_once_with("检查这段代马")
        with self.server.application.correction_lock:
            status, _ = self.request("POST", "/api/correct", body, headers)
        self.assertEqual(status, 409)
        self.assertEqual(self.injector.values, [])

    def test_invalid_correction_text(self):
        for value in ([], {"text": ""}, {"text": 5}, {"text": "\ud800"}):
            status, _ = self.request("POST", "/api/correct", json.dumps(value).encode(),
                                     {"X-Voice-Token": "test-token"})
            self.assertEqual(status, 400)

    def test_empty_audio_is_rejected_before_inference(self):
        status, _ = self.request(
            "POST", "/api/transcribe", headers={"X-Voice-Token": "test-token"}
        )
        self.assertEqual(status, 400)

    def test_incomplete_upload_is_not_transcribed(self):
        with socket.create_connection(("127.0.0.1", self.port), timeout=2) as connection:
            connection.sendall(
                b"POST /api/transcribe HTTP/1.1\r\nHost: localhost\r\n"
                b"X-Voice-Token: test-token\r\nContent-Type: audio/webm\r\n"
                b"Content-Length: 100\r\n\r\nfake-audio"
            )
            connection.shutdown(socket.SHUT_WR)
            response = connection.makefile("rb").read()
        self.assertIn(b" 400 ", response.split(b"\r\n", 1)[0])

    def test_stalled_upload_times_out(self):
        with patch.object(VoiceRequestHandler, "timeout", 0.1, create=True):
            with socket.create_connection(("127.0.0.1", self.port), timeout=2) as connection:
                connection.sendall(
                    b"POST /api/transcribe HTTP/1.1\r\nHost: localhost\r\n"
                    b"X-Voice-Token: test-token\r\nContent-Length: 100\r\n\r\n"
                )
                response = connection.makefile("rb").read()
        self.assertIn(b" 408 ", response.split(b"\r\n", 1)[0])

    def test_oversized_audio_is_rejected_without_reading_it(self):
        status, _ = self.request(
            "POST", "/api/transcribe",
            headers={"X-Voice-Token": "test-token", "Content-Length": str(26 * 1024**2)},
        )
        self.assertEqual(status, 413)

    def test_invalid_text_does_not_reach_injector(self):
        for payload, expected in (
            ([], 400), ({"text": ""}, 400), ({"text": 7}, 400),
            ({"text": "a" * 20001}, 413), ({"text": "\ud800"}, 400),
        ):
            with self.subTest(payload_type=type(payload), expected=expected):
                status, _ = self.request(
                    "POST", "/api/inject", json.dumps(payload).encode(),
                    {"X-Voice-Token": "test-token"},
                )
                self.assertEqual(status, expected)
        self.assertEqual(self.injector.values, [])

    def test_missing_decoder_returns_json_error(self):
        with patch.object(self.server.application.transcriber, "transcribe",
                          side_effect=FileNotFoundError("ffmpeg")):
            status, body = self.request(
                "POST", "/api/transcribe", b"fake-audio",
                {"X-Voice-Token": "test-token", "Content-Type": "audio/webm"},
            )
        self.assertEqual(status, 503)
        self.assertIn("error", json.loads(body))

    def test_decoder_timeout_returns_json_error(self):
        with patch.object(self.server.application.transcriber, "transcribe",
                          side_effect=subprocess.TimeoutExpired("ffmpeg", 60)):
            status, body = self.request(
                "POST", "/api/transcribe", b"fake-audio",
                {"X-Voice-Token": "test-token", "Content-Type": "audio/webm"},
            )
        self.assertEqual(status, 504)
        self.assertIn("error", json.loads(body))

    def test_simultaneous_pastes_are_rejected_and_lock_is_released(self):
        entered = threading.Event()
        release = threading.Event()
        original_inject = self.injector.inject

        def slow_inject(text):
            entered.set()
            release.wait(5)
            original_inject(text)

        def send_text():
            return self.request(
                "POST", "/api/inject", b'{"text":"hello"}',
                {"X-Voice-Token": "test-token"},
            )

        with ThreadPoolExecutor(max_workers=1) as pool:
            with patch.object(self.injector, "inject", side_effect=slow_inject):
                first = pool.submit(send_text)
                try:
                    self.assertTrue(entered.wait(2))
                    status, _ = send_text()
                    self.assertEqual(status, 409)
                finally:
                    release.set()
                self.assertEqual(first.result()[0], 200)
        self.assertEqual(send_text()[0], 200)
        self.assertEqual(self.injector.values, ["hello", "hello"])

    def test_simultaneous_transcriptions_return_busy(self):
        entered = threading.Event()
        release = threading.Event()
        original = self.server.application.transcriber.transcribe

        def slow_transcribe(audio, mime_type):
            entered.set()
            release.wait(5)
            return original(audio, mime_type)

        def send_audio():
            return self.request(
                "POST", "/api/transcribe", b"fake-audio",
                {"X-Voice-Token": "test-token", "Content-Type": "audio/webm"},
            )

        with ThreadPoolExecutor(max_workers=1) as pool:
            with patch.object(self.server.application.transcriber, "transcribe",
                              side_effect=slow_transcribe):
                first = pool.submit(send_audio)
                try:
                    self.assertTrue(entered.wait(2))
                    status, _ = send_audio()
                    self.assertEqual(status, 409)
                finally:
                    release.set()
                self.assertEqual(first.result()[0], 200)
        self.assertEqual(send_audio()[0], 200)


class CommandLineTests(unittest.TestCase):
    def test_language_flag_overrides_environment_and_defaults_to_english(self):
        with patch.dict(os.environ):
            os.environ.pop("VOICE_UI_LANGUAGE", None)
            self.assertEqual(voice_server.parse_args([]).ui_language, "en")
            os.environ["VOICE_UI_LANGUAGE"] = "zh-CN"
            self.assertEqual(voice_server.parse_args([]).ui_language, "zh-CN")
            self.assertEqual(voice_server.parse_args(["--ui-language", "en"]).ui_language, "en")

    def test_server_startup_uses_selected_language_and_generic_destination(self):
        for language, expected in (("en", "Voice input server is running."), ("zh-CN", "语音输入服务已启动")):
            with self.subTest(language=language):
                args = voice_server.parse_args([
                    "--ui-language", language, "--dry-run", "--port", "0", "--token", "public-test-token",
                ])
                args.correction_model = ""
                args.tls_cert = args.tls_key = None
                server = Mock()
                server.serve_forever.side_effect = KeyboardInterrupt
                output = io.StringIO()
                with patch.object(voice_server, "parse_args", return_value=args), \
                        patch.object(voice_server, "SenseVoiceTranscriber", return_value=FakeTranscriber()), \
                        patch.object(voice_server, "VoiceHTTPServer", return_value=server) as factory, \
                        patch.object(voice_server, "local_ipv4_addresses", return_value=[]), redirect_stdout(output):
                    self.assertEqual(voice_server.main(), 0)
                self.assertIn(expected, output.getvalue())
                self.assertIn("destination input field" if language == "en" else "接收文字的输入框", output.getvalue())
                self.assertEqual(factory.call_args.args[1].status()["ui_language"], language)
                server.server_close.assert_called_once()

    def test_imported_model_defaults_are_portable_without_environment_overrides(self):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("VOICE_")}
        result = subprocess.run([
            sys.executable, "-c", "import json,server; print(json.dumps([str(server.DEFAULT_MODEL),str(server.DEFAULT_TOKENS),str(server.DEFAULT_QWEN_MODEL_DIR),str(server.DEFAULT_QWEN_NATIVE_MODEL_DIR)]))",
        ], cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True, text=True, check=True)
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(json.loads(result.stdout), [
            str(root / ".local/models/sensevoice/model.int8.onnx"),
            str(root / ".local/models/sensevoice/tokens.txt"),
            str(root / ".local/models/qwen3-asr-0.6b"),
            str(root / ".local/models/Qwen3-ASR-1.7B"),
        ])


if __name__ == "__main__":
    unittest.main()

