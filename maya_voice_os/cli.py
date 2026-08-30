"""
Command-line entry point, installed as `maya-voice-os` by pip.

    maya-voice-os local          -> mic/speaker demo, in-process
    maya-voice-os orchestrator   -> starts orchestration-service (:8004)
    maya-voice-os telephony      -> starts telephony-service (:8100)
"""

from __future__ import annotations

import argparse
import asyncio
import sys


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


def _run_telephony() -> None:
    import os
    import uvicorn
    from maya_voice_os.telephony_service.server import app
    uvicorn.run(
        app,
        host=os.getenv("TELEPHONY_HOST", "0.0.0.0"),
        port=int(os.getenv("TELEPHONY_PORT", "8100")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="maya-voice-os", description="Maya Voice OS — free, local voice AI pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("local", help="Talk to Maya with your mic/speakers (no servers).")
    sub.add_parser("orchestrator", help="Start orchestration-service (the /process HTTP API).")
    sub.add_parser("telephony", help="Start telephony-service (talks to orchestration-service).")

    args = parser.parse_args()
    if args.command == "local":
        _run_local()
    elif args.command == "orchestrator":
        _run_orchestrator()
    elif args.command == "telephony":
        _run_telephony()


if __name__ == "__main__":
    sys.exit(main())
