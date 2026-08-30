"""
Command-line entry point, installed as `maya-voice-os` by pip.

    maya-voice-os local          -> mic/speaker demo, in-process
    maya-voice-os orchestrator   -> starts orchestration-service (:8004)
    maya-voice-os telephony      -> starts telephony-service (:8100)
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
import time
import wave
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

FIXED_TEST_SET = [
    "hello there",
    "good morning",
    "please book a meeting",
    "what is your pricing",
    "i need help",
    "thanks for calling",
]


def _percentile(samples: Sequence[float], pct: float) -> float:
    if not samples:
        return 0.0
    values = sorted(float(v) for v in samples)
    if len(values) == 1:
        return values[0]
    rank = max(1, math.ceil((pct / 100.0) * len(values)))
    return values[min(rank - 1, len(values) - 1)]


def _normalize_text(value: str) -> str:
    return " ".join((value or "").lower().strip().split())


def _score_fixed_test_set(expected: Sequence[str], actual: Sequence[str]) -> float:
    if not expected:
        return 1.0
    actual_norm = {_normalize_text(v): True for v in actual}
    matched = sum(1 for item in expected if _normalize_text(item) in actual_norm)
    return matched / len(expected)


def _read_wav_file(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise ValueError(f"Unsupported wav format in {path.name}: sample width {sample_width} bytes is not PCM16.")

    audio = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32) / 32768.0, sample_rate


def _measure_asr_latency(wav_files: Sequence[Path]) -> tuple[list[float], list[dict[str, object]]]:
    from maya_voice_os.asr_service.engine import ASREngine

    asr = ASREngine()
    latency_samples: list[float] = []
    rows: list[dict[str, object]] = []

    for wav_path in wav_files:
        audio, sample_rate = _read_wav_file(wav_path)
        start = time.perf_counter()
        result = asr.transcribe(audio, sample_rate=sample_rate)
        latency_ms = (time.perf_counter() - start) * 1000.0
        latency_samples.append(latency_ms)
        rows.append(
            {
                "filename": wav_path.name,
                "latency_ms": round(latency_ms, 2),
                "text": result.text,
                "hallucination_filtered": "Y" if result.was_hallucination else "N",
            }
        )

    return latency_samples, rows


def _run_eval() -> None:
    eval_dir = Path.cwd() / "eval_audio"
    if not eval_dir.exists() or not eval_dir.is_dir():
        print("No eval_audio/ directory found in the current working directory.")
        print("This benchmark is best-effort; use the more detailed latency suite below or add wavs under eval_audio/.")
        return

    wav_files = sorted(eval_dir.glob("*.wav"))
    if not wav_files:
        print("No .wav files found in eval_audio/.")
        return

    latency_samples, rows = _measure_asr_latency(wav_files)
    if not rows:
        return

    col_widths = {
        "filename": max(len("Filename"), max(len(str(r["filename"])) for r in rows)),
        "latency": max(len("ASR_Latency_ms"), max(len(f"{float(r['latency_ms']):.2f}") for r in rows)),
        "text": max(len("Transcribed_Text"), max(len(str(r["text"])) for r in rows)),
        "hallucination": max(len("Hallucination_Filtered (Y/N)"), max(len(str(r["hallucination_filtered"])) for r in rows)),
    }

    header = (
        f"{'Filename'.ljust(col_widths['filename'])} | "
        f"{'ASR_Latency_ms'.ljust(col_widths['latency'])} | "
        f"{'Transcribed_Text'.ljust(col_widths['text'])} | "
        f"{'Hallucination_Filtered (Y/N)'.ljust(col_widths['hallucination'])}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        text = str(row["text"] or "")
        print(
            f"{str(row['filename']).ljust(col_widths['filename'])} | "
            f"{float(row['latency_ms']):.2f}   ".ljust(col_widths['latency'] + 3) + "| "
            f"{text[: col_widths['text']].ljust(col_widths['text'])} | "
            f"{str(row['hallucination_filtered']).ljust(col_widths['hallucination'])}"
        )

    print("\nPercentiles:")
    print(f"P50: {_percentile(latency_samples, 50):.2f} ms")
    print(f"P95: {_percentile(latency_samples, 95):.2f} ms")
    print(f"P99: {_percentile(latency_samples, 99):.2f} ms")


def _run_latency_suite(calls: int = 30, concurrency: int = 3, threshold_ms: float = 2500.0) -> None:
    """
    Genuinely measures the real pipeline — LLM (llm_router.chat, real network
    calls to whichever providers are configured) and TTS (real edge-tts
    calls) — instead of the previous version, which cycled through hardcoded
    constant arrays and never touched the actual code.

    ASR timing is intentionally NOT included here: including it would mean
    silently loading and running faster-whisper on synthetic silence, which
    wouldn't measure anything real either. Use `--suite asr` with real .wav
    files under eval_audio/ for genuine ASR latency — that suite already
    calls the real ASREngine on real audio.

    Concurrency is real: `concurrency` calls actually run at once via
    asyncio.gather, not an algebraic formula estimating what concurrency
    "should" do.
    """
    import asyncio as _asyncio

    from maya_voice_os.llm_service.llm_router import LLMRouter
    from maya_voice_os.tts_service.engine import get_provider as get_tts_provider

    if calls <= 0:
        calls = 1
    if concurrency <= 0:
        concurrency = 1

    tts = get_tts_provider()
    test_inputs = [FIXED_TEST_SET[i % len(FIXED_TEST_SET)] for i in range(calls)]

    async def _one_call(text: str) -> dict[str, object]:
        # A fresh LLMRouter per call, not one shared instance — LLMRouter() is
        # cheap to construct (just reads provider config from env, no model
        # loading), and this avoids a real race condition: last_provider_used
        # is mutable instance state, and under real concurrency (which this
        # benchmark specifically exercises), two concurrent calls sharing one
        # router could each overwrite that field before the other reads it.
        local_router = LLMRouter()
        result: dict[str, object] = {"ok": False, "total_ms": None, "llm_ms": None, "tts_ms": None}
        try:
            t0 = time.perf_counter()
            reply = await _asyncio.to_thread(local_router.chat, [{"role": "user", "content": text}])
            llm_ms = (time.perf_counter() - t0) * 1000.0

            # chat() never raises — on total provider failure it returns a
            # graceful fallback string ("I'm having trouble thinking right
            # now...") by design, so a real call never just goes silent. A
            # non-empty reply alone does NOT prove a provider actually
            # answered; last_provider_used is None specifically when every
            # provider failed, which is the real signal to check.
            provider_answered = local_router.last_provider_used is not None
            if not provider_answered:
                result["error"] = "all providers failed (fallback message returned, not a real answer)"

            t1 = time.perf_counter()
            await tts.synthesize_pcm16(reply, voice="en-US-EmmaMultilingualNeural", sample_rate=16000)
            tts_ms = (time.perf_counter() - t1) * 1000.0

            result.update(
                ok=provider_answered,
                total_ms=llm_ms + tts_ms,
                llm_ms=llm_ms,
                tts_ms=tts_ms,
            )
        except Exception as exc:  # a failed call is real data too — record it as a failure, don't hide it
            result["error"] = str(exc)
        return result

    async def _run_at_concurrency(inputs: list[str], conc: int) -> list[dict[str, object]]:
        semaphore = _asyncio.Semaphore(conc)

        async def _bounded(text: str) -> dict[str, object]:
            async with semaphore:
                return await _one_call(text)

        return await _asyncio.gather(*(_bounded(t) for t in inputs))

    results = _asyncio.run(_run_at_concurrency(test_inputs, concurrency))

    ok_results = [r for r in results if r["ok"]]
    total_samples = [float(r["total_ms"]) for r in ok_results if r["total_ms"] is not None]
    llm_samples = [float(r["llm_ms"]) for r in ok_results if r["llm_ms"] is not None]
    tts_samples = [float(r["tts_ms"]) for r in ok_results if r["tts_ms"] is not None]
    pass_rate = len(ok_results) / len(results) if results else 0.0

    print("Latency benchmark suite (real LLM + TTS calls, ASR excluded — see --suite asr)")
    print(f"calls={calls} concurrency={concurrency} threshold_ms={threshold_ms}")
    if total_samples:
        print(f"P50 first-response latency: {_percentile(total_samples, 50):.2f} ms")
        print(f"P95 first-response latency: {_percentile(total_samples, 95):.2f} ms")
        print(f"P99 first-response latency: {_percentile(total_samples, 99):.2f} ms")
        print("Stage breakdown (measured, not estimated):")
        print(f"  llm_ms: {sum(llm_samples) / len(llm_samples):.2f} ms (avg over {len(llm_samples)} calls)")
        print(f"  tts_ms: {sum(tts_samples) / len(tts_samples):.2f} ms (avg over {len(tts_samples)} calls)")
    else:
        print("No successful calls — every request failed. Check LLM provider keys in .env.")
    failed = len(results) - len(ok_results)
    if failed:
        print(f"{failed}/{len(results)} calls FAILED (see errors below) — this is real failure data, not hidden.")
        for i, r in enumerate(results):
            if not r["ok"]:
                print(f"  call {i}: {r.get('error', 'empty response')}")
    print(f"Pass rate (non-empty response, no exception): {pass_rate * 100.0:.1f}%")
    if total_samples:
        over_threshold = sum(1 for v in total_samples if v > threshold_ms)
        print(f"Calls exceeding {threshold_ms:.0f}ms threshold: {over_threshold}/{len(total_samples)}")


def _run_local() -> None:
    from maya_voice_os.run_local import main
    asyncio.run(main())


def _run_orchestrator() -> None:
    import os
    import uvicorn
    from maya_voice_os.orchestration_service.server import app
    uvicorn.run(
        app,
        host=os.getenv("ORCHESTRATION_HOST", "0.0.0.0"),
        port=int(os.getenv("ORCHESTRATION_PORT", "8004")),
    )


def _run_telephony(barge_in_demo: bool = False, safe_mode: bool = False) -> None:
    import os
    import uvicorn
    from maya_voice_os.telephony_service.server import app
    if barge_in_demo:
        os.environ["BARGE_IN_DEMO_MODE"] = "true"
    if safe_mode:
        os.environ["SAFE_MODE"] = "true"
    uvicorn.run(
        app,
        host=os.getenv("TELEPHONY_HOST", "0.0.0.0"),
        port=int(os.getenv("TELEPHONY_PORT", "8100")),
    )


def _parse_eval_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="maya-eval", description="Run Maya Voice OS latency and ASR benchmark checks.")
    parser.add_argument("--suite", choices=["asr", "latency"], default="asr", help="Benchmark suite to run.")
    parser.add_argument("--calls", type=int, default=30, help="Number of latency calls to simulate in the latency suite.")
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrency assumed for the latency suite.")
    parser.add_argument("--threshold", type=float, default=2500.0, help="Latency threshold in milliseconds before the benchmark flags a concurrency limit.")
    parser.add_argument("--eval-dir", type=Path, default=Path.cwd() / "eval_audio", help="Directory containing WAV files for the ASR suite.")
    return parser.parse_args(argv)


def main_eval(argv: Sequence[str] | None = None) -> int:
    args = _parse_eval_args(argv)
    if args.suite == "latency":
        _run_latency_suite(calls=args.calls, concurrency=args.concurrency, threshold_ms=args.threshold)
        return 0

    eval_dir = args.eval_dir
    if not eval_dir.exists() or not eval_dir.is_dir():
        print(f"No eval_audio/ directory found at {eval_dir}.")
        return 1

    wav_files = sorted(eval_dir.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files found in {eval_dir}.")
        return 1

    _run_eval()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="maya-voice-os", description="Maya Voice OS — free, local voice AI pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)
    local_parser = sub.add_parser("local", help="Talk to Maya with your mic/speakers (no servers).")
    orchestrator_parser = sub.add_parser("orchestrator", help="Start orchestration-service (the /process HTTP API).")
    telephony_parser = sub.add_parser("telephony", help="Start telephony-service (talks to orchestration-service).")
    telephony_parser.add_argument("--barge-in-demo", action="store_true", help="Run Twilio in demo mode with intentional TTS gaps and visible interruption logging.")
    telephony_parser.add_argument("--safe-mode", action="store_true", help="Use a more conservative, public-demo-safe response profile.")
    eval_parser = sub.add_parser("eval", help="Run ASR or latency evaluations.")
    eval_parser.add_argument("--suite", choices=["asr", "latency"], default="asr", help="Benchmark suite to run.")
    eval_parser.add_argument("--calls", type=int, default=30, help="Number of latency calls to simulate in the latency suite.")
    eval_parser.add_argument("--concurrency", type=int, default=3, help="Concurrency assumed for the latency suite.")
    eval_parser.add_argument("--threshold", type=float, default=2500.0, help="Latency threshold in milliseconds before the benchmark flags a concurrency limit.")
    eval_parser.add_argument("--eval-dir", type=Path, default=Path.cwd() / "eval_audio", help="Directory containing WAV files for the ASR suite.")

    args = parser.parse_args()
    if args.command == "local":
        _run_local()
    elif args.command == "orchestrator":
        _run_orchestrator()
    elif args.command == "telephony":
        _run_telephony(barge_in_demo=args.barge_in_demo, safe_mode=args.safe_mode)
    elif args.command == "eval":
        if args.suite == "latency":
            _run_latency_suite(calls=args.calls, concurrency=args.concurrency, threshold_ms=args.threshold)
        else:
            eval_dir = args.eval_dir
            if not eval_dir.exists() or not eval_dir.is_dir():
                print(f"No eval_audio/ directory found at {eval_dir}.")
                raise SystemExit(1)
            wav_files = sorted(eval_dir.glob("*.wav"))
            if not wav_files:
                print(f"No .wav files found in {eval_dir}.")
                raise SystemExit(1)
            _run_eval()


if __name__ == "__main__":
    sys.exit(main())