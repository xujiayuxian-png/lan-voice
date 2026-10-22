"""Opt-in real-model checks using the local model's Chinese/English samples."""
import importlib.util
import io
import os
from pathlib import Path
import shutil
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import DEFAULT_MODEL, SenseVoiceTranscriber

MODEL_DIR = Path(os.environ.get("VOICE_MODEL_DIR", DEFAULT_MODEL.parent))


@unittest.skipUnless(
    os.environ.get("VOICE_TEST_ASR") == "1"
    and importlib.util.find_spec("sherpa_onnx")
    and importlib.util.find_spec("soundfile")
    and shutil.which("ffmpeg"),
    "Set VOICE_TEST_ASR=1 with sherpa-onnx, soundfile, ffmpeg and local sample audio",
)
class MultilingualTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import soundfile as sf
        cls.np, cls.sf = np, sf
        cls.transcriber = SenseVoiceTranscriber(
            MODEL_DIR / "model.int8.onnx", MODEL_DIR / "tokens.txt", 4
        )
        cls.clips = {}
        for language in ("zh", "en"):
            samples, rate = sf.read(MODEL_DIR / "test_wavs" / f"{language}.wav")
            if rate != 16000 or samples.ndim != 1:
                raise ValueError("The reference fixtures must be mono 16 kHz audio")
            cls.clips[language] = samples

    def transcribe(self, languages):
        parts = []
        for language in languages:
            if parts:
                parts.append(self.np.zeros(4800))  # 300 ms between real speech clips.
            parts.append(self.clips[language])
        audio = io.BytesIO()
        self.sf.write(audio, self.np.concatenate(parts), 16000, format="WAV")
        return self.transcriber.transcribe(audio.getvalue(), "audio/wav")

    def test_chinese_is_preserved(self):
        text = self.transcribe(["zh"])
        self.assertIn("早上9点至下午5点", text)

    def test_english_remains_english(self):
        text = self.transcribe(["en"]).lower()
        self.assertIn("tribal chieftain", text)
        self.assertIn("50 pieces of gold", text)

    def test_mixed_language_in_one_sentence(self):
        audio = Path(__file__).parent / "fixtures" / "mixed_zh_en.wav"
        text = self.transcriber.transcribe(audio.read_bytes(), "audio/wav")
        self.assertIn("今天下午开会", text)
        self.assertIn("please send me the report", text.lower())
        self.assertIn("然后我们一起检查", text)

    def test_language_switches_do_not_drop_chinese_or_english(self):
        for languages in (["zh", "en", "zh"], ["en", "zh", "en"]):
            with self.subTest(languages=languages):
                text = self.transcribe(languages)
                self.assertEqual(text.count("早上9点至下午5点"), languages.count("zh"), text)
                self.assertEqual(text.lower().count("tribal chieftain"), languages.count("en"), text)
