"""Retry Logic with Exponential Backoff

Implements retry logic for transient failures in inter-service communication.
"""

import asyncio
import logging
import random
from functools import wraps
from typing import Callable, Tuple, Type, Union, Optional

import httpx

logger = logging.getLogger(__name__)


# Exceptions that should trigger retry
RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    asyncio.TimeoutError,
    ConnectionError,
    ConnectionResetError,
)

# HTTP status codes that should trigger retry
RETRYABLE_STATUS_CODES = {502, 503, 504, 429}


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: float = 0.1,
    ):
        """
        Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Random jitter factor (0.1 = ±10%)
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt with jitter."""
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )

        # Add jitter
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)

        return max(0.1, delay)  # Minimum 100ms


class RetryExhausted(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.last_exception = last_exception


async def retry_with_backoff(
    func: Callable,
    config: RetryConfig = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
):
    """
    Execute a function with retry logic and exponential backoff.

    Args:
        func: Async function to execute
        config: Retry configuration
        on_retry: Callback called on each retry (attempt, exception)

    Returns:
        Result of the function

    Raises:
        RetryExhausted: When all retries are exhausted
    """
    config = config or RetryConfig()
    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            return func()

        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e

            if attempt < config.max_retries:
                delay = config.get_delay(attempt)
                logger.warning(
                    f"Retryable error (attempt {attempt + 1}/{config.max_retries + 1}): "
                    f"{type(e).__name__}: {str(e)}. Retrying in {delay:.2f}s"
                )

                if on_retry:
                    on_retry(attempt, e)

                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"All {config.max_retries + 1} attempts exhausted: "
                    f"{type(e).__name__}: {str(e)}"
                )

        except httpx.HTTPStatusError as e:
            if e.response.status_code in RETRYABLE_STATUS_CODES:
                last_exception = e

                if attempt < config.max_retries:
                    # Check for Retry-After header
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = config.get_delay(attempt)
                    else:
                        delay = config.get_delay(attempt)

                    logger.warning(
                        f"Retryable HTTP {e.response.status_code} (attempt {attempt + 1}/{config.max_retries + 1}). "
                        f"Retrying in {delay:.2f}s"
                    )

                    if on_retry:
                        on_retry(attempt, e)

                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"All {config.max_retries + 1} attempts exhausted with HTTP {e.response.status_code}"
                    )
            else:
                # Non-retryable HTTP error
                raise

    raise RetryExhausted(
        f"All {config.max_retries + 1} retry attempts exhausted",
        last_exception
    )


def with_retry(
    config: RetryConfig = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
):
    """
    Decorator to add retry logic to async functions.

    Usage:
        @with_retry(RetryConfig(max_retries=3))
        async def call_service():
            ...
    """
    config = config or RetryConfig()

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async def call():
                return await func(*args, **kwargs)

            return await retry_with_backoff(call, config, on_retry)

        return wrapper

    return decorator


# Pre-configured retry configs for different services
ASR_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    max_delay=10.0,
)

LLM_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    initial_delay=2.0,
    max_delay=16.0,
)

TTS_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    max_delay=10.0,
)
