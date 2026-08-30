"""
Small, dependency-light audio helpers:
- resampling (numpy-only, no scipy/torchaudio dependency)
- mu-law <-> PCM16 conversion (for Twilio Media Streams, which use 8kHz mu-law)
"""

from __future__ import annotations

import audioop  # stdlib on <3.13; `audioop-lts` backport on 3.13+ (see requirements.txt)
import numpy as np


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear-interpolation resampler. Fine for speech; avoids heavy deps."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    target_len = int(round(duration * target_sr))
    orig_idx = np.linspace(0, len(audio) - 1, num=len(audio))
    target_idx = np.linspace(0, len(audio) - 1, num=target_len)
    return np.interp(target_idx, orig_idx, audio).astype(np.float32)


def pcm16_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    ints = np.frombuffer(pcm_bytes, dtype=np.int16)
    return (ints.astype(np.float32)) / 32768.0


def float32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    ints = (clipped * 32767.0).astype(np.int16)
    return ints.tobytes()


def mulaw_bytes_to_float32(mulaw_bytes: bytes, sample_rate: int = 8000) -> np.ndarray:
    """Twilio sends 8kHz mu-law audio over its Media Streams WebSocket."""
    pcm_bytes = audioop.ulaw2lin(mulaw_bytes, 2)  # -> 16-bit PCM
    return pcm16_bytes_to_float32(pcm_bytes)


def float32_to_mulaw_bytes(audio: np.ndarray) -> bytes:
    """Encode float32 PCM back to 8kHz mu-law for sending to Twilio."""
    pcm_bytes = float32_to_pcm16_bytes(audio)
    return audioop.lin2ulaw(pcm_bytes, 2)
