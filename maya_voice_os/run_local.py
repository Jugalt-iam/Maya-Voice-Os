"""
Mic/speaker demo — talk to Maya with no servers running.
Invoked via `maya-voice-os local` (see cli.py) or `python -m maya_voice_os.run_local`.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

from maya_voice_os.orchestration_service.pipeline import ConversationState, VoicePipeline

RECORD_SR = 16000


def record_until_enter() -> np.ndarray:
    print("\n🎙️  Recording... press Enter to stop.")
    frames = []

    def callback(indata, frames_count, time_info, status):
        frames.append(indata.copy())

    stream = sd.InputStream(samplerate=RECORD_SR, channels=1, dtype="float32", callback=callback)
    with stream:
        try:
            input()
        except EOFError:
            # Some terminals send EOF (e.g. Ctrl+Z+Enter on Windows) instead
            # of a plain Enter keypress. Treat it as "stop recording" rather
            # than crashing the whole demo.
            print("(stdin closed — stopping recording; press Ctrl+C to quit next time instead)")
    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames, axis=0).flatten()


def play_audio(audio: np.ndarray, sample_rate: int) -> None:
    sd.play(audio, samplerate=sample_rate)
    sd.wait()


async def main():
    print("Loading pipeline (first run downloads the faster-whisper 'small' model)...")
    pipeline = VoicePipeline()
    await pipeline.async_init()  # wires up Redis for smart_memory, context_manager
    state = ConversationState(conversation_id="local-cli")

    print(f"\n🤖 {pipeline.identity.name}: {pipeline.identity.greeting}")
    greeting_audio = await pipeline.tts.synthesize_pcm16(pipeline.identity.greeting, voice=pipeline.identity.voice, sample_rate=16000)
    play_audio(greeting_audio, 16000)

    try:
        while True:
            audio = record_until_enter()
            if audio.size == 0:
                continue
            reply_text, transcript, reply_audio, timings = await pipeline.handle_utterance(audio, RECORD_SR, state)
            print(f"   (asr={timings['asr_ms']}ms, respond={timings['respond_ms']}ms, tts={timings['tts_ms']}ms)")
            if transcript:
                print(f"🧑 You: {transcript}")
            if reply_text:
                print(f"🤖 {pipeline.identity.name}: {reply_text}")
                play_audio(reply_audio, 16000)
    except KeyboardInterrupt:
        print("\nBye!")
        sys.exit(0)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
