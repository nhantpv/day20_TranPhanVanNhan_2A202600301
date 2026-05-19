"""Tracing hooks.

Provides a minimal in-process span context manager that every agent uses,
plus an optional LangSmith pass-through so spans show up in the LangSmith
project when ``LANGSMITH_API_KEY`` is configured.

The local span is always recorded — the LangSmith integration is best-effort
so the workflow keeps running if the tracing backend is unavailable.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)


def _langsmith_tracer() -> Any | None:
    """Return a LangSmith client if configured, else ``None``."""

    if not os.environ.get("LANGSMITH_API_KEY"):
        return None
    try:
        from langsmith import Client  # type: ignore

        return Client()
    except Exception as exc:  # noqa: BLE001
        logger.debug("LangSmith not available: %s", exc)
        return None


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Record a named span with timing and arbitrary attributes.

    The yielded dict can be mutated by the caller to add attributes — useful
    for recording counts or routes discovered mid-span.
    """

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    client = _langsmith_tracer()
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
        if client is not None:
            try:
                client.create_run(
                    name=name,
                    run_type="chain",
                    inputs=span["attributes"],
                    outputs={"duration_seconds": span["duration_seconds"]},
                    project_name=os.environ.get("LANGSMITH_PROJECT", "multi-agent-research-lab"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("LangSmith create_run failed: %s", exc)
