"""Real ffmpeg decoding with a lightweight recognizer, no model download."""
import importlib.util
import io
import shutil
import sys
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from streaming_asr import AudioSession, QwenStreamingSessions


class NativeSessionCleanupTest(unittest.TestCase):
    def setUp(self):
        self.transcriber = server.Qwen3StreamingTranscriber.__new__(server.Qwen3StreamingTranscriber)
        self.transcriber._lock = threading.Lock()
        self.transcriber._sessions = QwenStreamingSessions(SimpleNamespace())
        self.transcriber._sessions.sessions["phone"] = AudioSession(updated_at=time.monotonic())
        self.transcriber.decode_audio = Mock(side_effect=server.TranscriptionError("Invalid final WebM"))

    def test_intermediate_decode_failure_keeps_state_for_next_cumulative_upload(self):
        with self.assertRaises(server.TranscriptionError):
            self.transcriber.transcribe_partial(
                b"unfinished", "audio/webm", stream_id="phone", sequence=4, final=False
            )
        self.assertIn("phone", self.transcriber._sessions.sessions)
        self.assertFalse(self.transcriber._lock.locked())

    def test_final_decode_failure_frees_previous_state_and_wrapper_lock(self):
        with self.assertRaises(server.TranscriptionError):
            self.transcriber.transcribe_partial(
                b"invalid-final", "audio/webm", stream_id="phone", sequence=5, final=True
            )
        self.assertNotIn("phone", self.transcriber._sessions.sessions)
        self.assertFalse(self.transcriber._lock.locked())


class LanguageConfigurationTest(unittest.TestCase):
    def test_recognizer_automatically_detects_language(self):
        # Inspect the actual SDK boundary without needing the external model.
        factory = Mock()
        sdk = SimpleNamespace(OfflineRecognizer=SimpleNamespace(from_sense_voice=factory))
        with tempfile.TemporaryDirectory() as directory:
            model, tokens = Path(directory) / "model.onnx", Path(directory) / "tokens.txt"
            model.touch()
            tokens.touch()
            with patch.dict(sys.modules, {"sherpa_onnx": sdk, "soundfile": SimpleNamespace()}):
                server.SenseVoiceTranscriber(model, tokens, 2)
        self.assertEqual(factory.call_args.kwargs["language"], "auto")

    def test_qwen3_configuration_uses_all_required_files_and_hotwords(self):
        factory = Mock()
        sdk = SimpleNamespace(
            OfflineRecognizer=SimpleNamespace(from_qwen3_asr=factory)
        )
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            for name in ("conv_frontend.onnx", "encoder.int8.onnx", "decoder.int8.onnx"):
                (model_dir / name).touch()
            (model_dir / "tokenizer").mkdir()
            with patch.dict(
                sys.modules,
                {"sherpa_onnx": sdk, "soundfile": SimpleNamespace()},
            ):
                transcriber = server.Qwen3ASRTranscriber(
                    model_dir, 3, provider="cuda", hotwords="议程,报告"
                )
        self.assertEqual(transcriber.description, "Qwen3-ASR local ONNX")
        self.assertEqual(factory.call_args.kwargs["num_threads"], 3)
        self.assertEqual(factory.call_args.kwargs["provider"], "cuda")
        self.assertEqual(factory.call_args.kwargs["hotwords"], "议程,报告")


@unittest.skipUnless(
    shutil.which("ffmpeg") and importlib.util.find_spec("soundfile"),
    "ffmpeg and soundfile are required for audio decoding tests",
)
class TranscriberTest(unittest.TestCase):
    def setUp(self):
        import soundfile
        self.transcriber = server.SenseVoiceTranscriber.__new__(server.SenseVoiceTranscriber)
        self.transcriber._soundfile = soundfile
        self.transcriber._lock = threading.Lock()
        self.received = []
        stream = SimpleNamespace(
            accept_waveform=lambda rate, samples: self.received.append((rate, samples)),
            result=SimpleNamespace(text="  本地   转写  "),
        )
        self.transcriber._recognizer = SimpleNamespace(
            create_stream=lambda: stream, decode_stream=lambda stream: None,
        )

    def audio(self, seconds):
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(b"\0" * int(seconds * 8000 * 4))
        return output.getvalue()

    def test_audio_is_converted_to_mono_16khz(self):
        text = self.transcriber.transcribe(self.audio(0.5), "audio/wav")
        self.assertEqual(text, "本地 转写")
        rate, samples = self.received[0]
        self.assertEqual(rate, 16000)
        self.assertEqual(samples.shape, (8000,))

    def test_long_audio_is_rejected_before_inference(self):
        with patch.object(server, "MAX_AUDIO_SECONDS", 1, create=True):
            with self.assertRaisesRegex(server.TranscriptionError, "long|limit|duration"):
                self.transcriber.transcribe(self.audio(2), "audio/wav")
        self.assertEqual(self.received, [])

    def test_invalid_audio_is_reported(self):
        with self.assertRaises(server.TranscriptionError):
            self.transcriber.transcribe(b"not an audio file", "audio/webm")
        self.assertEqual(self.received, [])

    def test_pauses_preserve_both_languages_and_all_audio_in_order(self):
        import numpy as np
        rate = 16000
        speech = np.sin(2 * np.pi * 440 * np.arange(rate * 2) / rate)
        # Quiet recordings should have the same phrase boundaries as loud ones.
        for volume in (0.5, 0.005):
            with self.subTest(volume=volume):
                self.received.clear()
                samples = np.concatenate((speech, np.zeros(12800), speech)) * volume
                audio = io.BytesIO()
                self.transcriber._soundfile.write(audio, samples, rate, format="WAV", subtype="FLOAT")
                results = iter(["Hello world.", "请检查 Python function。"])
                self.transcriber._recognizer.create_stream = lambda: SimpleNamespace(
                    accept_waveform=lambda rate, samples: self.received.append((rate, samples)),
                    result=SimpleNamespace(text=next(results)),
                )
                text = self.transcriber.transcribe(audio.getvalue(), "audio/wav")
                self.assertEqual(text, "Hello world. 请检查 Python function。")
                self.assertEqual(len(self.received), 2)
                # ffmpeg emits PCM16; compare against that known quantization.
                joined = np.concatenate([part for _, part in self.received])
                np.testing.assert_allclose(joined, samples, atol=1 / 32768)

    def test_brief_pauses_do_not_break_a_phrase(self):
        import numpy as np
        rate = 16000
        speech = 0.1 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)
        samples = np.concatenate((speech, np.zeros(3200), speech))
        audio = io.BytesIO()
        self.transcriber._soundfile.write(audio, samples, rate, format="WAV")
        self.transcriber.transcribe(audio.getvalue(), "audio/wav")
        self.assertEqual(len(self.received), 1)
