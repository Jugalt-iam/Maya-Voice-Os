"""
Parallel LLM Processor — ported and made functional.

HONEST NOTE ON WHAT CHANGED: the uploaded version of this file was a stub
— `_process_request` did `await asyncio.sleep(0.1)` and returned nothing,
it never actually called an LLM. There was no real logic to preserve, so
this is a genuine implementation, not a port, built around the same
queue/worker shape the original sketched.

WHERE THIS ACTUALLY FITS: this repo's live call path (one utterance in,
one reply out per turn) doesn't need a manual worker pool — FastAPI is
already async and handles concurrent requests via the event loop for
free. A queue-and-workers pattern earns its keep for FAN-OUT scenarios:
firing off several independent LLM calls at once and collecting results
— e.g. generating a few candidate replies in parallel and picking the
best, or batch-testing a list of prompts against your configured
providers. It is NOT wired into the live /process path; use it directly
for anything that needs that fan-out shape.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("parallel-processor")


@dataclass
class ParallelRequest:
    request_id: str
    text: str
    system_prompt: Optional[str] = None
    priority: int = 0


@dataclass
class ParallelResult:
    request_id: str
    text: str
    reply: str = ""
    error: Optional[str] = None


class ParallelLLMProcessor:
    """
    Fans a batch of independent prompts out across a small worker pool and
    collects results. `llm_call` is any sync callable(messages, system_prompt)
    -> str — pass `llm_router.chat` (from llm-service/llm_router.py) to
    actually hit your configured free providers, or any test double for
    dry runs.

    Usage:
        processor = ParallelLLMProcessor(llm_router.chat, num_workers=3)
        results = await processor.run_batch([
            ParallelRequest(request_id="1", text="Summarize X"),
            ParallelRequest(request_id="2", text="Summarize Y"),
        ])
    """

    def __init__(self, llm_call: Callable[..., str], num_workers: int = 2):
        self.llm_call = llm_call
        self.num_workers = num_workers

    async def run_batch(self, requests: List[ParallelRequest]) -> List[ParallelResult]:
        queue: asyncio.Queue = asyncio.Queue()
        for req in sorted(requests, key=lambda r: -r.priority):
            queue.put_nowait(req)

        results: Dict[str, ParallelResult] = {}
        workers = [asyncio.create_task(self._worker(f"worker_{i}", queue, results)) for i in range(self.num_workers)]

        await queue.join()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        # Preserve the caller's original request order in the output.
        return [results[r.request_id] for r in requests]

    async def _worker(self, worker_id: str, queue: asyncio.Queue, results: Dict[str, ParallelResult]) -> None:
        while True:
            try:
                request: ParallelRequest = await queue.get()
            except asyncio.CancelledError:
                return

            try:
                # llm_router.chat is synchronous/blocking (see its own
                # docstring on why) — run it in a thread so one slow
                # provider call doesn't stall the whole event loop.
                reply = await asyncio.to_thread(
                    self.llm_call,
                    [{"role": "user", "content": request.text}],
                    request.system_prompt,
                )
                results[request.request_id] = ParallelResult(
                    request_id=request.request_id, text=request.text, reply=reply
                )
            except Exception as e:
                logger.error(f"[{worker_id}] request {request.request_id} failed: {e}")
                results[request.request_id] = ParallelResult(
                    request_id=request.request_id, text=request.text, error=str(e)
                )
            finally:
                queue.task_done()
