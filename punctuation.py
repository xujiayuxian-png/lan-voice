"""Local punctuation restoration that keeps the recognized words intact."""
from __future__ import annotations

from pathlib import Path
import re

try:
    from .correction import FORMAT_CHARS, PROTECTED, guarded_formatting
except ImportError:
    from correction import FORMAT_CHARS, PROTECTED, guarded_formatting


def punctuation_input(text: str) -> str:
    """Remove premature Chinese sentence marks, retaining literal spans."""
    locked = [match.span() for match in PROTECTED.finditer(text)]
    return "".join(
        char for index, char in enumerate(text)
        if char not in "，。；！？：、" or any(a <= index < b for a, b in locked)
    )


def repair_existing_breaks(original: str, candidate: str) -> str:
    """Repair ASR gaps without inserting new breaks inside continuous words.

    A punctuation classifier can mistake an embedded 'how/whether' clause for
    a question. Keep the original sentence force unless ASR already supplied
    a question mark at that same character boundary.
    """
    def split(text):
        characters, gaps = [], [""]
        for char in text:
            if char in FORMAT_CHARS:
                gaps[-1] += char
            else:
                characters.append(char)
                gaps.append("")
        return characters, gaps

    characters, gaps = split(original)
    proposed_characters, proposed_gaps = split(candidate)
    if characters != proposed_characters:
        return original
    pieces = []
    for index, (gap, proposed_gap) in enumerate(zip(gaps, proposed_gaps)):
        replacement = proposed_gap if gap else ""
        if any(mark in gap for mark in "?？"):
            # Do not swallow an existing question into the following clause.
            # A model may still repair a spurious question to a full stop.
            if not any(mark in replacement for mark in ".。?!？！"):
                replacement = gap
        else:
            replacement = replacement.replace("?", "。").replace("？", "。")
        pieces.append(replacement)
        if index < len(characters):
            pieces.append(characters[index])
    return "".join(pieces)


class TranscriptPunctuator:
    description = "CT-Transformer Chinese punctuation (CPU)"

    def __init__(self, model: Path, threads: int = 2) -> None:
        if not model.is_file():
            raise FileNotFoundError(f"Punctuation model not found: {model}")
        import sherpa_onnx

        self.model = model
        self.engine = sherpa_onnx.OfflinePunctuation(
            sherpa_onnx.OfflinePunctuationConfig(
                model=sherpa_onnx.OfflinePunctuationModelConfig(
                    ct_transformer=str(model), num_threads=threads, provider="cpu",
                )
            )
        )

    def restore(self, text: str) -> str:
        if not re.search(r"[\u4e00-\u9fff]", text):
            return text
        candidate = self.engine.add_punctuation(punctuation_input(text))
        if not isinstance(candidate, str):
            raise ValueError("Punctuation model returned invalid text")
        # Copy the original characters and protected English/code/numbers.
        # Only punctuation gaps between matching Chinese neighbours may move.
        return repair_existing_breaks(text, guarded_formatting(text, candidate))
