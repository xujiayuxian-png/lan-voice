"""Audio/session regressions at the documented Qwen3-ASR SDK boundary."""
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streaming_asr import FinalizationError, QwenStreamingSessions, pause_boundary


RATE = 16000


def speech(seconds=1, amplitude=0.1):
    return (amplitude * np.sin(2 * np.pi * 440 * np.arange(round(RATE * seconds)) / RATE)).astype(np.float32)


class AudioEngine:
    """Strict signatures mirror official qwen_asr.py without loading weights."""

    def __init__(self):
        self.states = []
        self.fed = []
        self.recognized = []
        self.fail = None

    def init_streaming_state(
        self, context="", language=None, unfixed_chunk_num=2,
        unfixed_token_num=5, chunk_size_sec=2.0,
    ):
        if self.fail == "init":
            raise RuntimeError("ASR initialization failed")
        state = SimpleNamespace(text="", number=len(self.states) + 1)
        self.states.append((state, context, language, unfixed_chunk_num, unfixed_token_num, chunk_size_sec))
        return state

    def streaming_transcribe(self, pcm16k, state):
        if self.fail == "stream":
            raise RuntimeError("ASR streaming failed")
        self.fed.append((np.array(pcm16k, copy=True), state))
        state.text = f"草稿{len(self.fed)}"
        return state

    def transcribe(self, audio, context="", language=None, return_time_stamps=False):
        if self.fail == "recognize":
            raise RuntimeError("ASR finalization failed")
        samples, rate = audio
        if rate != RATE:
            raise AssertionError("The official SDK expects 16 kHz PCM")
        self.recognized.append((np.array(samples, copy=True), context, language))
        return [SimpleNamespace(text=f"确认{len(self.recognized)}")]


class StreamingSessionsTest(unittest.TestCase):
    def setUp(self):
        self.engine = AudioEngine()
        self.manager = QwenStreamingSessions(self.engine, context="会议、议程、Python")

    def test_skipped_snapshots_feed_only_new_pcm_and_replace_draft(self):
        samples = speech(3)
        first = self.manager.snapshot("phone", 4, samples[:RATE], False)
        second = self.manager.snapshot("phone", 9, samples[:RATE * 2], False)
        third = self.manager.snapshot("phone", 15, samples, False)
        self.assertEqual(first["draft_text"], "草稿1")
        self.assertEqual(second["text"], "草稿2")
        self.assertEqual(third["text"], "草稿3")
        self.assertEqual(third["confirmed_text"], "")
        self.assertEqual(len(self.engine.states), 1)
        self.assertEqual(self.engine.states[0][1:], ("", None, 2, 8, 1.0))
        np.testing.assert_array_equal(np.concatenate([part for part, _ in self.engine.fed]), samples)

    def test_repeated_audio_snapshot_does_not_feed_empty_pcm(self):
        samples = speech()
        self.manager.snapshot("phone", 0, samples, False)
        result = self.manager.snapshot("phone", 3, samples, False)
        self.assertEqual(result["draft_text"], "草稿1")
        self.assertEqual(len(self.engine.fed), 1)

    def test_punctuation_uses_accumulated_confirmed_text_without_delaying_drafts(self):
        seen = []

        def restore(text):
            seen.append(text)
            return text.replace("确认1 确认2", "确认1，确认2")

        manager = QwenStreamingSessions(self.engine, punctuator=SimpleNamespace(restore=restore))
        samples = speech()
        first = manager.snapshot("phone", 0, samples, False)
        self.assertEqual(first["draft_text"], "草稿1")
        self.assertEqual(seen, [])
        samples = np.concatenate((samples, np.zeros(round(RATE * 0.8)), speech()))
        paused = manager.snapshot("phone", 1, samples, False)
        self.assertEqual(seen, ["确认1"])
        self.assertEqual(manager.sessions["phone"].finalized, ["确认1"])
        manager.snapshot("phone", 2, samples, False)
        self.assertEqual(seen, ["确认1"], "unchanged confirmed text should use the cached punctuation")
        final = manager.snapshot("phone", 3, samples, True)
        self.assertEqual(seen, ["确认1", "确认1 确认2"])
        self.assertEqual(final["text"], "确认1，确认2")
        self.assertEqual(final["confirmed_text"], final["text"])
        self.assertEqual(final["punctuation_status"], "formatted")
        self.assertEqual(final["draft_text"], "")

    def test_punctuation_failure_preserves_audio_result_and_does_not_block_finalization(self):
        def fail(text):
            raise RuntimeError("punctuation unavailable")

        manager = QwenStreamingSessions(self.engine, punctuator=SimpleNamespace(restore=fail))
        result = manager.snapshot("phone", 0, speech(), True)
        self.assertEqual(result["text"], "确认1")
        self.assertEqual(result["punctuation_status"], "unavailable")
        self.assertTrue(result["final"])
        self.assertNotIn("phone", manager.sessions)

    def test_selected_context_is_used_only_for_pause_and_final_audio_rechecks(self):
        initial = speech()
        samples = np.concatenate((initial, np.zeros(round(RATE * 0.8)), speech()))
        for context in ("", "产品发布会议", "Python debugging", "会议 action items"):
            with self.subTest(context=context):
                engine = AudioEngine()
                manager = QwenStreamingSessions(engine, context="service default")
                first = manager.snapshot("phone", 0, initial, False, context=context)
                paused = manager.snapshot("phone", 2, samples, False)
                final = manager.snapshot("phone", 4, samples, True)
                self.assertEqual([item[1] for item in engine.recognized], [context, context])
                self.assertTrue(all(item[1] == "" for item in engine.states))
                for result in (first, paused, final):
                    self.assertEqual(result["asr_context"], context)
                self.assertEqual(manager.context, "service default")

    def test_recording_contexts_are_isolated_and_changed_final_preserves_live_session(self):
        self.manager.snapshot("first", 0, speech(), False, context="release planning")
        self.manager.snapshot("second", 0, speech(), False, context="日常聊天")
        first = self.manager.sessions["first"]
        state = first.state
        for changed_context in ("", "different"):
            with self.subTest(context=changed_context), self.assertRaisesRegex(ValueError, "cannot change"):
                self.manager.snapshot("first", 1, speech(), True, context=changed_context)
            self.assertIs(self.manager.sessions["first"], first)
            self.assertIs(first.state, state)
            self.assertEqual(first.sequence, 0)
        final = self.manager.snapshot("first", 1, speech(), True)
        self.assertEqual(final["asr_context"], "release planning")
        self.assertEqual(self.engine.recognized[-1][1], "release planning")
        self.assertIn("second", self.manager.sessions)
        final = self.manager.snapshot("second", 1, speech(), True, context="日常聊天")
        self.assertEqual(final["asr_context"], "日常聊天")
        self.assertEqual(self.engine.recognized[-1][1], "日常聊天")
        self.assertEqual(self.manager.context, "会议、议程、Python")

    def test_default_context_is_locked_by_first_snapshot(self):
        self.manager.snapshot("phone", 0, speech(), False)
        with self.assertRaisesRegex(ValueError, "cannot change"):
            self.manager.snapshot("phone", 1, speech(), False, context="")
        result = self.manager.snapshot("phone", 1, speech(), True, context=self.manager.context)
        self.assertEqual(result["asr_context"], "会议、议程、Python")

    def test_invalid_context_does_not_allocate_or_damage_sessions(self):
        self.manager.snapshot("live", 0, speech(), False, context="keep me")
        for context in ("x" * 501, "😀" * 501, "\ud800", 123, False):
            with self.subTest(context=repr(context)), self.assertRaises(ValueError):
                self.manager.snapshot("invalid", 0, speech(), False, context=context)
            self.assertNotIn("invalid", self.manager.sessions)
            self.assertEqual(self.manager.sessions["live"].context, "keep me")

    def test_expired_recording_can_choose_a_different_context(self):
        with patch("streaming_asr.time.monotonic", return_value=0):
            self.manager.snapshot("phone", 0, speech(), False, context="old context")
        with patch("streaming_asr.time.monotonic", return_value=180):
            result = self.manager.snapshot("phone", 0, speech(), True, context="new context")
        self.assertEqual(result["asr_context"], "new context")
        self.assertEqual(self.engine.recognized[-1][1], "new context")

    def test_engine_failure_releases_selected_context_without_affecting_other_recordings(self):
        self.manager.snapshot("live", 0, speech(), False, context="still active")
        self.engine.fail = "stream"
        with self.assertRaises(RuntimeError):
            self.manager.snapshot("failed", 0, speech(), False, context="discard me")
        self.assertNotIn("failed", self.manager.sessions)
        self.assertEqual(self.manager.sessions["live"].context, "still active")
        self.engine.fail = None
        self.manager.snapshot("failed", 1, speech(), True)
        self.assertEqual(self.engine.recognized[-1][1], self.manager.context)
        self.manager.snapshot("live", 1, speech(), True)
        self.assertEqual(self.engine.recognized[-1][1], "still active")

    def test_pause_finalizes_audio_and_preserves_speech_after_boundary(self):
        initial = speech()
        samples = np.concatenate((initial, np.zeros(int(RATE * 0.8), dtype=np.float32), speech(0.5, 0.2)))
        self.manager.snapshot("phone", 0, initial, False)
        result = self.manager.snapshot("phone", 4, samples, False)
        boundary = pause_boundary(samples)
        self.assertIsNotNone(boundary)
        self.assertEqual(result["confirmed_text"], "确认1")
        self.assertEqual(result["draft_text"], "草稿2")
        np.testing.assert_array_equal(self.engine.recognized[0][0], samples[:boundary])
        self.assertEqual(self.engine.recognized[0][1], "会议、议程、Python")
        np.testing.assert_array_equal(self.engine.fed[-1][0], samples[boundary:])
        self.assertIsNot(self.engine.fed[0][1], self.engine.fed[-1][1])

        extended = np.concatenate((samples, speech(0.6, 0.2)))
        self.manager.snapshot("phone", 8, extended, False)
        np.testing.assert_array_equal(self.engine.fed[-1][0], extended[len(samples):])

    def test_multiple_pauses_and_final_tail_preserve_every_audio_sample(self):
        silence = np.zeros(int(RATE * 0.8), dtype=np.float32)
        samples = np.concatenate((speech(), silence, speech(1, 0.2), silence, speech(0.5, 0.3)))
        partial = self.manager.snapshot("phone", 7, samples, False)
        self.assertEqual(partial["confirmed_text"], "确认1 确认2")
        final_samples = np.concatenate((samples, speech(0.3, 0.3)))
        final = self.manager.snapshot("phone", 11, final_samples, True)
        self.assertEqual(final["text"], "确认1 确认2 确认3")
        self.assertEqual(final["draft_text"], "")
        self.assertTrue(final["final"])
        self.assertNotIn("phone", self.manager.sessions)
        np.testing.assert_array_equal(
            np.concatenate([part for part, _, _ in self.engine.recognized]), final_samples
        )

    def test_empty_sentence_recheck_keeps_audio_for_later_recognition(self):
        initial = speech()
        samples = np.concatenate((initial, np.zeros(round(RATE * 0.8)), speech()))
        recognize = self.engine.transcribe

        def temporarily_empty(audio, context="", language=None, return_time_stamps=False):
            result = recognize(audio, context, language, return_time_stamps)
            if len(self.engine.recognized) == 1:
                result[0].text = ""
            return result

        self.engine.transcribe = temporarily_empty
        self.manager.snapshot("phone", 0, initial, False)
        result = self.manager.snapshot("phone", 5, samples, False)
        self.assertEqual(result["confirmed_text"], "")
        self.assertEqual(self.manager.sessions["phone"].offset, 0)
        self.assertEqual(len(self.engine.states), 1)
        np.testing.assert_array_equal(np.concatenate([part for part, _ in self.engine.fed]), samples)
        final = self.manager.snapshot("phone", 9, samples, True)
        self.assertEqual(final["text"], "确认2 确认3")
        np.testing.assert_array_equal(
            np.concatenate([part for part, _, _ in self.engine.recognized[1:]]), samples
        )

    def test_final_redecodes_complete_current_sentence_without_draft_prefix(self):
        first = speech()
        final_samples = np.concatenate((first, speech(0.25)))
        self.manager.snapshot("phone", 0, first, False)
        self.engine.states[0][0].text = "错误且不能作为最终识别前缀的草稿"
        result = self.manager.snapshot("phone", 5, final_samples, True)
        self.assertEqual(result["text"], "确认1")
        np.testing.assert_array_equal(self.engine.recognized[0][0], final_samples)
        self.assertEqual(self.engine.recognized[0][1:], ("会议、议程、Python", None))
        self.assertEqual(len(self.engine.fed), 1)
        self.assertNotIn("phone", self.manager.sessions)

    def test_empty_final_recheck_never_silently_discards_existing_draft(self):
        for with_confirmed in (False, True):
            with self.subTest(with_confirmed=with_confirmed):
                engine = AudioEngine()
                manager = QwenStreamingSessions(engine)
                samples = speech()
                if with_confirmed:
                    samples = np.concatenate((samples, np.zeros(round(RATE * 0.8)), speech()))
                partial = manager.snapshot("phone", 0, samples, False)
                self.assertTrue(partial["draft_text"])
                engine.transcribe = lambda audio, context: [SimpleNamespace(text="")]
                with self.assertRaises(FinalizationError) as caught:
                    manager.snapshot("phone", 2, samples, True)
                self.assertEqual(caught.exception.draft_text, partial["text"])
                self.assertNotIn("phone", manager.sessions)

    def test_minor_decoder_length_shrink_does_not_repeat_native_pcm(self):
        samples = speech(2)
        self.manager.snapshot("phone", 0, samples[:RATE], False)
        self.manager.snapshot("phone", 2, samples[:RATE - 320], False)
        self.assertEqual(len(self.engine.fed), 1)
        self.manager.snapshot("phone", 4, samples, False)
        np.testing.assert_array_equal(np.concatenate([part for part, _ in self.engine.fed]), samples)
        shorter_final = samples[:-320]
        result = self.manager.snapshot("phone", 8, shorter_final, True)
        self.assertTrue(result["final"])
        np.testing.assert_array_equal(self.engine.recognized[-1][0], shorter_final)

    def test_large_audio_shrink_is_rejected_without_damaging_retryable_state(self):
        samples = speech(2)
        self.manager.snapshot("phone", 0, samples, False)
        with self.assertRaisesRegex(ValueError, "complete recording"):
            self.manager.snapshot("phone", 1, samples[:RATE], False)
        result = self.manager.snapshot("phone", 2, samples, True)
        self.assertEqual(result["text"], "确认1")

    def test_engine_failures_discard_session_and_complete_audio_can_retry(self):
        for failure in ("init", "stream", "recognize"):
            with self.subTest(failure=failure):
                self.engine = AudioEngine()
                self.manager = QwenStreamingSessions(self.engine)
                self.engine.fail = failure
                with self.assertRaises(RuntimeError):
                    self.manager.snapshot("phone", 0, speech(), failure == "recognize")
                self.assertNotIn("phone", self.manager.sessions)
                self.engine.fail = None
                result = self.manager.snapshot("phone", 4, speech(), True)
                self.assertEqual(result["text"], "确认1")
                self.assertNotIn("phone", self.manager.sessions)

    def test_failure_after_pause_discards_partially_finalized_session(self):
        samples = np.concatenate((speech(), np.zeros(round(RATE * 0.8)), speech()))
        self.engine.fail = "stream"
        with self.assertRaises(RuntimeError):
            self.manager.snapshot("phone", 0, samples, False)
        self.assertEqual(len(self.engine.recognized), 1, "The first sentence was finalized before the failure")
        self.assertNotIn("phone", self.manager.sessions)
        self.engine.fail = None
        result = self.manager.snapshot("phone", 4, samples, True)
        self.assertEqual(result["text"], "确认2 确认3")
        np.testing.assert_array_equal(
            np.concatenate([audio for audio, _, _ in self.engine.recognized[1:]]), samples
        )

    def test_capacity_recovers_after_ttl_and_does_not_evict_active_recording(self):
        manager = QwenStreamingSessions(self.engine, max_sessions=1)
        with patch("streaming_asr.time.monotonic", return_value=10):
            manager.snapshot("first", 0, speech(), False)
        with patch("streaming_asr.time.monotonic", return_value=189):
            with self.assertRaisesRegex(RuntimeError, "Too many"):
                manager.snapshot("second", 0, speech(), False)
        self.assertIn("first", manager.sessions)
        with patch("streaming_asr.time.monotonic", return_value=190):
            manager.snapshot("second", 0, speech(), False)
        self.assertNotIn("first", manager.sessions)
        self.assertIn("second", manager.sessions)

    def test_sequence_rejection_keeps_live_state_and_accepts_later_final(self):
        samples = speech()
        self.manager.snapshot("phone", 3, samples, False)
        for sequence in (3, 2):
            with self.subTest(sequence=sequence), self.assertRaisesRegex(ValueError, "increase"):
                self.manager.snapshot("phone", sequence, samples, False)
        self.assertEqual(len(self.engine.fed), 1)
        self.assertEqual(self.manager.snapshot("phone", 6, samples, True)["text"], "确认1")

    def test_receiving_new_audio_refreshes_session_expiry(self):
        with patch("streaming_asr.time.monotonic", return_value=0):
            self.manager.snapshot("phone", 0, speech(), False)
        with patch("streaming_asr.time.monotonic", return_value=179):
            self.manager.snapshot("phone", 5, speech(2), False)
        with patch("streaming_asr.time.monotonic", return_value=180):
            result = self.manager.snapshot("phone", 9, speech(2), True)
        self.assertEqual(result["text"], "确认1")
        self.assertEqual(len(self.engine.states), 1)

    def test_invalid_first_sequence_does_not_allocate_session(self):
        for sequence in (-1, -5, "1", 1.5, True):
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                self.manager.snapshot("phone", sequence, speech(), False)
            self.assertEqual(self.manager.sessions, {})

    def test_silent_and_empty_audio_do_not_hallucinate_or_load_engine_state(self):
        for samples in (np.zeros(0, dtype=np.float32), np.zeros(RATE, dtype=np.float32)):
            with self.subTest(samples=len(samples)):
                self.assertEqual(self.manager.snapshot("phone", 0, samples, False)["text"], "")
                self.assertEqual(self.manager.snapshot("phone", 1, samples, True)["text"], "")
                self.assertNotIn("phone", self.manager.sessions)
        self.assertEqual(self.engine.states, [])
        self.assertEqual(self.engine.recognized, [])

    def test_invalid_audio_is_rejected_before_allocating_state(self):
        for samples in (np.ones((2, 20)), np.array([np.nan]), np.array([np.inf])):
            with self.subTest(shape=samples.shape), self.assertRaises(ValueError):
                self.manager.snapshot("phone", 0, samples, False)
            self.assertEqual(self.manager.sessions, {})

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is needed for cumulative Opus decoding")
    def test_real_cumulative_webm_feeds_only_new_decoded_pcm(self):
        ffmpeg = shutil.which("ffmpeg")
        samples = speech(2.005)
        encoded = subprocess.run(
            [ffmpeg, "-v", "error", "-f", "f32le", "-ar", str(RATE), "-ac", "1", "-i", "pipe:0",
             "-c:a", "libopus", "-cluster_time_limit", "500", "-f", "webm", "pipe:1"],
            input=samples.astype("<f4").tobytes(), capture_output=True, check=True, timeout=10,
        ).stdout
        final_pcm = None
        decoded_snapshots = []
        for sequence, fraction in ((0, 0.4), (3, 0.7), (8, 1.0)):
            decoded = subprocess.run(
                [ffmpeg, "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ar", str(RATE), "-ac", "1", "pipe:1"],
                input=encoded[:round(len(encoded) * fraction)], capture_output=True, check=True, timeout=10,
            )
            final_pcm = np.frombuffer(decoded.stdout, dtype="<f4")
            decoded_snapshots.append(final_pcm)
            self.assertGreater(len(final_pcm), 0)
            self.manager.snapshot("phone", sequence, final_pcm, fraction == 1)
        fed = np.concatenate([audio for audio, _ in self.engine.fed])
        # ffmpeg can revise a few resampling-edge samples when more Opus
        # packets arrive. The draft keeps already-fed PCM; final recognition
        # must instead use the exact fully decoded container from scratch.
        expected_new_pcm = np.concatenate((decoded_snapshots[0], decoded_snapshots[1][len(decoded_snapshots[0]):]))
        np.testing.assert_array_equal(fed, expected_new_pcm)
        np.testing.assert_array_equal(self.engine.recognized[0][0], final_pcm)
        self.assertNotIn("phone", self.manager.sessions)


class PauseBoundaryTest(unittest.TestCase):
    def test_relative_threshold_preserves_quiet_speech_and_brief_pauses(self):
        quiet = speech(amplitude=0.001)
        for gap, expected in ((0.2, False), (0.8, True)):
            samples = np.concatenate((quiet, np.zeros(round(gap * RATE)), quiet))
            with self.subTest(gap=gap):
                self.assertEqual(pause_boundary(samples) is not None, expected)
                if expected:
                    boundary = pause_boundary(samples)
                    self.assertGreater(boundary, len(quiet))
                    self.assertLess(boundary, len(quiet) + round(gap * RATE))


if __name__ == "__main__":
    unittest.main()
