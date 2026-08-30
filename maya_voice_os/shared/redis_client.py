"""
Shared Redis client. Redis is now a real, configured dependency (see
requirements.txt and .env.sample) used by llm-service/smart_memory.py for
cross-turn sales-conversation state (objections, buying signals, emotional
trend, stage).

If REDIS_HOST is unset, this returns None and callers fall back to
in-memory state for that process's lifetime — but unlike before, that's a
loud, logged fallback, not a silent one, since Redis is expected to be
configured for local runs now.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("redis-client")

_redis_client = None
_warned_missing = False


async def get_redis_client():
    """Returns an async redis client, or None if Redis isn't configured/reachable.
    Cached after first successful connection attempt."""
    global _redis_client, _warned_missing

    if _redis_client is not None:
        return _redis_client

    host = os.getenv("REDIS_HOST")
    if not host:
        if not _warned_missing:
            logger.warning(
                "REDIS_HOST not set — smart_memory will use in-memory state only "
                "(lost on restart, not shared across processes). Set REDIS_HOST in "
                ".env and run a local Redis server to enable persistent conversation "
                "memory. See README for setup."
            )
            _warned_missing = True
        return None

    try:
        import redis.asyncio as redis

        port = int(os.getenv("REDIS_PORT", "6379"))
        password = os.getenv("REDIS_PASSWORD") or None
        client = redis.Redis(host=host, port=port, password=password, decode_responses=True)
        await client.ping()
        logger.info(f"Connected to Redis at {host}:{port}")
        _redis_client = client
        return _redis_client
    except Exception as e:
        if not _warned_missing:
            logger.warning(
                f"Could not connect to Redis at {host} ({e}) — falling back to "
                "in-memory state. Check REDIS_HOST/REDIS_PORT/REDIS_PASSWORD in .env "
                "and that a Redis server is actually running."
            )
            _warned_missing = True
        return None
