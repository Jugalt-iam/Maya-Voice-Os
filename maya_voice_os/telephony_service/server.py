"""
Telephony-service — provider-agnostic gateway. Both bundled adapters are
mounted; you only ever configure the ONE your chosen provider needs
(Twilio webhook URL, or Exotel webhook URL + secret) — the other simply
never receives traffic.

To add a new provider: write a new file in adapters/ that talks to your
provider's wire protocol on one side and calls `orchestration_client` on
the other (see adapters/twilio_adapter.py for a streaming example, or
adapters/exotel_adapter.py for a turn-based example), then add one line
mounting its router below.

This service ONLY talks to:
  - the telephony provider (Twilio/Exotel/whatever you wire up)
  - this repo's own orchestration-service (ORCHESTRATION_URL)
No calls anywhere else, no dependency on any external/private project.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from maya_voice_os.telephony_service.adapters.twilio_adapter import router as twilio_router
from maya_voice_os.telephony_service.adapters.exotel_adapter import router as exotel_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
logger = logging.getLogger("telephony-service")

app = FastAPI(title="telephony-service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("TELEPHONY_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(twilio_router)
app.include_router(exotel_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("TELEPHONY_HOST", "0.0.0.0"),
        port=int(os.getenv("TELEPHONY_PORT", "8100")),
    )
