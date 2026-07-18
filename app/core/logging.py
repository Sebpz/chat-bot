import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

from app.core.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def log_duration(logger: logging.Logger, operation: str, **context: object) -> Iterator[None]:
    """Times a block and logs its duration in milliseconds.

    Used throughout the ingestion and chat pipelines to record per-stage
    latency (parsing, chunking, embedding, retrieval, reranking, LLM, total)
    as required by the spec's logging section.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        extra = " ".join(f"{k}={v}" for k, v in context.items())
        logger.info("%s completed duration_ms=%.2f %s", operation, duration_ms, extra)
