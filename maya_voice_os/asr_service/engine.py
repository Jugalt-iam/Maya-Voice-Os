"""
ASR — speech to text using faster-whisper.

Defaults to the "small" model on CPU with int8 quantization: light enough to
run on a laptop with no GPU, while still supporting full multilingual
transcription (English + Hindi + other Indic languages Whisper covers).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger("asr")

# ---------------------------------------------------------------------------
# Hallucination filter — ported from the original ASR service's main.py.
# Whisper (and faster-whisper) is known to "hallucinate" plausible-sounding
# but fabricated text on silence, noise, or very short/unclear audio —
# stock filler phrases, music-video sign-offs, or repeated garbage. This
# catches the common patterns before they ever reach the LLM.
# ---------------------------------------------------------------------------
HALLUCINATION_PATTERNS = [
    r"^thank you for watching\.?$",
    r"^thanks for watching\.?$",
    r"^please subscribe\.?$",
    r"^like and subscribe\.?$",
    r"^see you next time\.?$",
    r"^bye\.?$",
    r"^goodbye\.?$",
    r"^okay\.?$",
    r"^hmm\.?$",
    r"^uh\.?$",
    r"^um\.?$",
    r"^you\.?$",
    r"^i'm\.?$",
    r"^\.*$",  # Just periods
    r"^♪+$",   # Music symbols
    r"^♫+$",
    r"^[\u10A0-\u10FF]+\.?$",  # Georgian script
    r"^[\u4E00-\u9FFF]+\.?$",  # Chinese characters
    r"^[\u3040-\u309F]+\.?$",  # Japanese Hiragana
    r"^[\u30A0-\u30FF]+\.?$",  # Japanese Katakana
    r"^音楽\.?$",  # Japanese "music"
    r"^字幕\.?$",  # Chinese "subtitles"
    # Hindi gibberish patterns (only clear noise, not valid Hindi)
    r"^जागी\s*(सब|कोप्ता).*$",
]


def filter_hallucination(text: str, safe_mode: bool = False) -> Tuple[str, bool]:
    """
    Filter out common Whisper hallucinations.
    Returns (filtered_text, was_hallucination).
    In safe mode, we apply the stricter checks more aggressively to avoid
    weird public-demo outputs.
    """
    if not text:
        return "", True

    text_lower = text.lower().strip()
    # Normalize trailing punctuation before matching — the patterns above
    # only account for a trailing period; without this, "Thanks for
    # watching!" (exclamation instead of period) slips through unfiltered.
    text_lower_normalized = text_lower.rstrip(".!?")

    for pattern in HALLUCINATION_PATTERNS:
        if re.match(pattern, text_lower, re.IGNORECASE) or re.match(pattern, text_lower_normalized, re.IGNORECASE):
            return "", True

    # Repetitive text (e.g. the same word/phrase looping) is another
    # common hallucination signature.
    words = text.split()
    if len(words) >= 4:
        unique_words = set(words)
        repetition_ratio = len(words) / len(unique_words)
        if repetition_ratio > 2.5:
            return "", True

    if safe_mode and len(words) <= 3 and len(text) < 25:
        return "", True

    return text, False


@dataclass
class TranscriptResult:
    text: str
    language: str
    language_probability: float
    was_hallucination: bool = False


class ASREngine:
    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
        beam_size: Optional[int] = None,
        cpu_threads: Optional[int] = None,
        safe_mode: Optional[bool] = None,
    ):
        self.model_size = model_size or os.getenv("ASR_MODEL_SIZE", "small")
        self.device = device or os.getenv("ASR_DEVICE", "cpu")
        self.compute_type = compute_type or os.getenv("ASR_COMPUTE_TYPE", "int8")
        # Empty string / None => auto-detect language per utterance.
        self.language = language if language else (os.getenv("ASR_LANGUAGE") or None)
        # Beam search width: the single biggest speed/accuracy trade-off
        # knob for CPU transcription. faster-whisper's own default is 5,
        # which is noticeably slow for real-time conversational voice —
        # each extra beam roughly multiplies decode time. 1-2 is the
        # standard choice for live voice; raise it back toward 5 only if
        # you're seeing real accuracy problems and have CPU to spare.
        self.beam_size = beam_size if beam_size is not None else int(os.getenv("ASR_BEAM_SIZE", "1"))
        # None lets ctranslate2 pick automatically (usually all cores).
        # Set explicitly if you want to reserve cores for other processes.
        cpu_threads_env = os.getenv("ASR_CPU_THREADS")
        self.cpu_threads = cpu_threads if cpu_threads is not None else (int(cpu_threads_env) if cpu_threads_env else 0)
        self.safe_mode = safe_mode if safe_mode is not None else os.getenv("SAFE_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
        self.transcription_lock = asyncio.Lock()

        logger.info(
            f"Loading faster-whisper '{self.model_size}' "
            f"(device={self.device}, compute_type={self.compute_type}, "
            f"beam_size={self.beam_size}, cpu_threads={self.cpu_threads or 'auto'})..."
        )
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
        )
        logger.info("ASR model ready.")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptResult:
        """
        audio: mono float32 numpy array in [-1, 1], sampled at `sample_rate`.
        faster-whisper expects 16kHz; resample upstream if you're feeding it
        telephony audio (see src/telephony/twilio_stream.py).
        """
        if sample_rate != 16000:
            raise ValueError(
                f"ASREngine.transcribe expects 16kHz audio, got {sample_rate}Hz. "
                "Resample before calling (see audio_utils.resample_audio)."
            )

        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()

        filtered_text, was_hallucination = filter_hallucination(text, safe_mode=self.safe_mode)
        if was_hallucination and text:
            logger.info(f"Filtered likely hallucination: {text[:60]!r}")

        return TranscriptResult(
            text=filtered_text,
            language=info.language,
            language_probability=info.language_probability,
            was_hallucination=was_hallucination,
        )
