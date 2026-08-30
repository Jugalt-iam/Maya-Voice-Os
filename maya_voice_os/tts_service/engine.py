"""
TTS service.

Default provider: Edge TTS (free, keyless, cloud). No model download, no GPU.

Edge TTS's Python client always returns MP3-compressed audio (that's fixed
by the underlying library, not configurable), so it has to be decoded to
raw PCM before use. That decoding uses `av` (PyAV) rather than `pydub` +
a system `ffmpeg` binary: PyAV bundles its own compiled decoding libraries
directly inside the pip package, so no separate ffmpeg install/PATH setup
is needed on any OS. `av` is already installed anyway as a dependency of
`faster-whisper`, so this adds no extra footprint.

Also exposes a small provider interface (`TTSProvider`) so you can swap in
any other TTS API (ElevenLabs, PlayHT, Azure, a self-hosted model, etc.)
without touching orchestration code — just implement `synthesize_pcm16` for
your provider and set `TTS_PROVIDER` in .env.
"""

from __future__ import annotations

import io
import logging
import os
from abc import ABC, abstractmethod

import av
import numpy as np

logger = logging.getLogger("tts-service")


def _decode_audio_to_pcm16(audio_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Decodes any container/codec `av` understands (mp3, wav, ogg, ...) to
    mono float32 PCM at `sample_rate`. No system ffmpeg binary required."""
    container = av.open(io.BytesIO(audio_bytes))
    stream = container.streams.audio[0]
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    chunks = []
    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            chunks.append(resampled.to_ndarray())
    container.close()
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    pcm = np.concatenate(chunks, axis=1).flatten()
    return pcm.astype(np.float32) / 32768.0


class TTSProvider(ABC):
    """Implement this for any TTS backend you want to plug in."""

    @abstractmethod
    async def synthesize_pcm16(self, text: str, voice: str, sample_rate: int = 16000) -> np.ndarray:
        """Return mono float32 PCM audio in [-1, 1] at `sample_rate`."""
        raise NotImplementedError


class EdgeTTSProvider(TTSProvider):
    """Default provider. Free, keyless, requires internet."""

    async def synthesize_mp3(self, text: str, voice: str) -> bytes:
        import edge_tts

        communicate = edge_tts.Communicate(text, voice)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    async def synthesize_pcm16(self, text: str, voice: str, sample_rate: int = 16000) -> np.ndarray:
        mp3_bytes = await self.synthesize_mp3(text, voice)
        return _decode_audio_to_pcm16(mp3_bytes, sample_rate=sample_rate)


class CustomAPITTSProvider(TTSProvider):
    """
    TEMPLATE for wiring any other TTS REST API. Fill in `_call_api` for your
    provider's request/response shape, set:
        TTS_PROVIDER=custom_api
        TTS_API_URL=...
        TTS_API_KEY=...
    in .env, and everything else in the pipeline keeps working unchanged.
    """

    def __init__(self):
        self.api_url = os.getenv("TTS_API_URL", "")
        self.api_key = os.getenv("TTS_API_KEY", "")
        if not self.api_url:
            raise RuntimeError("TTS_PROVIDER=custom_api requires TTS_API_URL to be set in .env")

    async def _call_api(self, text: str, voice: str) -> bytes:
        """Return raw audio bytes (e.g. wav/mp3) from your provider. Example
        shown for a generic JSON POST -> binary-audio-response API; adjust
        to match the provider you're integrating."""
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self.api_url, json={"text": text, "voice": voice}, headers=headers)
            resp.raise_for_status()
            return resp.content

    async def synthesize_pcm16(self, text: str, voice: str, sample_rate: int = 16000) -> np.ndarray:
        audio_bytes = await self._call_api(text, voice)
        return _decode_audio_to_pcm16(audio_bytes, sample_rate=sample_rate)


def get_provider() -> TTSProvider:
    name = os.getenv("TTS_PROVIDER", "edge").lower()
    if name == "edge":
        return EdgeTTSProvider()
    if name == "custom_api":
        return CustomAPITTSProvider()
    raise ValueError(f"Unknown TTS_PROVIDER '{name}'. Use 'edge' or 'custom_api'.")
