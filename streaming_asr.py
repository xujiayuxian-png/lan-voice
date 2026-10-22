"""Revisable Qwen3-ASR speech recognition with audio-based sentence finalization."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any


# Finalizing an Opus/WebM container may remove encoder padding that was
# present when ffmpeg decoded an unfinished snapshot. Bound the tolerance
# to 100 ms; a genuinely truncated recording must still be rejected.
MAX_SNAPSHOT_SHRINK_SAMPLES = 1600
MAX_ASR_CONTEXT_CHARS = 500


class FinalizationError(RuntimeError):
    """A final audio recheck could not confirm an existing streaming draft."""

    def __init__(self, draft_text: str) -> None:
        super().__init__("最后一句的音频复核未成功，已保留草稿，请检查后重新录音或手动发送")
        self.draft_text = draft_text


@dataclass
class AudioSession:
    updated_at: float
    sequence: int = -1
    sample_count: int = 0
    offset: int = 0
    fed: int = 0
    state: Any = None
    finalized: list[str] = field(default_factory=list)
    context: str = ""
    punctuated_source: str = ""
    punctuated_text: str = ""
    punctuation_status: str = "pending"


def pause_boundary(samples: Any, sample_rate: int = 16000) -> int | None:
    """Find an audible pause, preserving the audio on both sides of it.

    This endpointing rule only selects boundaries. It never removes samples or
    filters speech, including quiet speech and short English phrases.
    """
    import numpy as np

    frame = sample_rate // 50
    count = len(samples) // frame
    if count < 40:
        return None
    frames = samples[:count * frame].reshape(count, frame)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    peak = float(rms.max())
    if peak < 0.0001:
        return None
    quiet = rms < peak * 0.05
    runs = np.flatnonzero(np.diff(np.r_[False, quiet, False])).reshape(-1, 2)
    for start, end in runs:
        if start * frame < sample_rate * 0.6:
            continue
        if (end - start) * frame >= sample_rate * 0.7:
            # Include the first 350 ms of silence in the previous sentence;
            # the rest stays at the beginning of the next one.
            return int((start + end) * frame // 2)
    return None


class QwenStreamingSessions:
    """Turn cumulative browser audio into bounded, revisable ASR sessions.

    The transport sends cumulative recordings and can skip superseded snapshots.
    Qwen receives only newly arrived PCM in its native streaming API. At a pause
    the entire current sentence is decoded again from audio without a text
    prefix; that result is finalized before starting the next sentence.
    """

    def __init__(
        self, engine: Any, context: str = "", max_sessions: int = 4,
        punctuator: Any = None,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("At least one speech recording session is required")
        self.engine = engine
        self.context = context
        self.max_sessions = max_sessions
        self.sessions: dict[str, AudioSession] = {}
        self.punctuator = punctuator

    def discard(self, stream_id: str) -> None:
        self.sessions.pop(stream_id, None)

    def resolve_context(self, stream_id: str, context: str | None = None) -> str:
        """Validate an override without changing an existing recording."""
        if context is not None:
            if not isinstance(context, str) or len(context) > MAX_ASR_CONTEXT_CHARS:
                raise ValueError(f"ASR context must contain at most {MAX_ASR_CONTEXT_CHARS} characters")
            try:
                context.encode("utf-8")
            except UnicodeError as exc:
                raise ValueError("ASR context must be valid UTF-8 text") from exc
        session = self.sessions.get(stream_id)
        if session is not None and time.monotonic() - session.updated_at < 180:
            if context is not None and context != session.context:
                raise ValueError("ASR context cannot change during a recording")
            return session.context
        return self.context if context is None else context

    def _recognize(self, samples: Any, context: str) -> str:
        import numpy as np

        if len(samples) < 1600 or float(np.max(np.abs(samples))) < 0.0001:
            return ""
        result = self.engine.transcribe(audio=(samples, 16000), context=context)
        return str(result[0].text).strip() if result else ""

    def _confirmed_text(self, session: AudioSession) -> str:
        source = " ".join(session.finalized)
        if self.punctuator is None or not source:
            return source
        if source == session.punctuated_source:
            return session.punctuated_text
        try:
            restored = self.punctuator.restore(source)
            if not isinstance(restored, str) or not restored.strip():
                raise ValueError("Punctuation restoration returned empty text")
        except Exception:
            # Formatting must never erase or prevent a successful audio result.
            session.punctuation_status = "unavailable"
            return source
        session.punctuated_source = source
        session.punctuated_text = restored
        session.punctuation_status = "formatted" if restored != source else "unchanged"
        return restored

    def snapshot(
        self, stream_id: str, sequence: int, samples: Any, final: bool,
        context: str | None = None,
    ) -> dict[str, Any]:
        import numpy as np

        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError("A stream id is required for Qwen3-ASR streaming")
        if isinstance(sequence, bool) or not isinstance(sequence, Integral) or sequence < 0:
            raise ValueError("The speech snapshot sequence must be a nonnegative integer")
        samples = np.asarray(samples)
        if samples.ndim != 1 or not np.issubdtype(samples.dtype, np.number) or not np.all(np.isfinite(samples)):
            raise ValueError("Speech snapshots must contain finite mono PCM samples")
        now = time.monotonic()
        self.sessions = {
            key: item for key, item in self.sessions.items()
            if now - item.updated_at < 180
        }
        effective_context = self.resolve_context(stream_id, context)
        session = self.sessions.get(stream_id)
        if session is None:
            if len(self.sessions) >= self.max_sessions:
                raise RuntimeError("Too many active speech recordings; please try again")
            session = AudioSession(updated_at=now, context=effective_context)
            self.sessions[stream_id] = session
        if sequence <= session.sequence:
            raise ValueError("The speech snapshot sequence must increase")
        if len(samples) + MAX_SNAPSHOT_SHRINK_SAMPLES < session.sample_count:
            raise ValueError("A speech snapshot must contain the complete recording")
        session.updated_at = now
        try:
            while True:
                tail = samples[session.offset:]
                boundary = pause_boundary(tail)
                if boundary is None:
                    break
                text = self._recognize(tail[:boundary], session.context)
                if not text:
                    # A failed/empty sentence recheck must not consume audio
                    # that the streaming draft may already have recognized.
                    # Keep it attached to the next speech and retry later.
                    break
                session.finalized.append(text)
                session.offset += boundary
                session.state = None
                session.fed = 0

            tail = samples[session.offset:]
            draft = ""
            if final:
                text = self._recognize(tail, session.context)
                if text:
                    session.finalized.append(text)
                elif session.state is not None and str(session.state.text).strip():
                    draft = str(session.state.text).strip()
                    raise FinalizationError(" ".join(part for part in (self._confirmed_text(session), draft) if part))
            elif len(tail) and float(np.max(np.abs(tail))) >= 0.0001:
                if session.state is None:
                    session.state = self.engine.init_streaming_state(
                        # A domain prompt can be copied into short, ambiguous
                        # drafts. Use it only for complete-sentence audio
                        # rechecks, where enough speech is available.
                        context="", chunk_size_sec=1.0,
                        unfixed_chunk_num=2, unfixed_token_num=8,
                    )
                if len(tail) > session.fed:
                    self.engine.streaming_transcribe(tail[session.fed:], session.state)
                    session.fed = len(tail)
                draft = str(session.state.text).strip()

            session.sequence = sequence
            session.sample_count = max(session.sample_count, len(samples))
            confirmed = self._confirmed_text(session)
            result = {
                "text": " ".join(part for part in (confirmed, draft) if part),
                "confirmed_text": confirmed,
                "draft_text": draft,
                "mode": "qwen3-native",
                "final": final,
                "asr_context": session.context,
            }
            if self.punctuator is not None:
                result["punctuation_status"] = session.punctuation_status
            return result
        except Exception:
            # An engine error can leave its KV/prefix state partially updated.
            # Discard it so a retry can rebuild from the complete audio.
            self.sessions.pop(stream_id, None)
            raise
        finally:
            if final:
                self.sessions.pop(stream_id, None)

