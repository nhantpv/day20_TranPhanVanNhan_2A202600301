"""Benchmark for single-agent vs multi-agent.

Adds:

* Quality scoring (heuristic: length, citation count, has-analysis flag).
* Estimated USD cost summed from per-agent metadata.
* Error count surfaced into ``BenchmarkMetrics.notes``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import AgentName, BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


_CITATION_RE = re.compile(r"\[S\d+\]")


def _score_quality(state: ResearchState) -> float:
    """Heuristic 0-10 quality score.

    The score is intentionally simple and well-known so it cannot be gamed by
    just making the answer longer — the components are independently capped.
    """

    answer = state.final_answer or ""
    if not answer:
        return 0.0

    # Length: reward roughly 200-800 word answers (up to 4.0).
    word_count = len(answer.split())
    if word_count < 50:
        length_score = word_count / 50 * 2.0
    elif word_count <= 800:
        length_score = 4.0
    else:
        length_score = max(0.0, 4.0 - (word_count - 800) / 400)

    # Citations: reward each unique citation up to 3.0.
    unique_citations = len(set(_CITATION_RE.findall(answer)))
    citation_score = min(3.0, unique_citations)

    # Multi-step credit: reward presence of separate research + analysis stages.
    structure_score = 0.0
    if state.research_notes:
        structure_score += 1.0
    if state.analysis_notes:
        structure_score += 1.0

    # Penalty for errors.
    error_penalty = min(2.0, 0.5 * len(state.errors))

    score = length_score + citation_score + structure_score - error_penalty
    return max(0.0, min(10.0, score))


def _sum_cost(state: ResearchState) -> float:
    total = 0.0
    for result in state.agent_results:
        cost = result.metadata.get("cost_usd")
        if isinstance(cost, (int, float)):
            total += float(cost)
    return total


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run a single benchmark cell and return both the final state and metrics."""

    with trace_span("benchmark.run", {"run": run_name, "query": query}):
        started = perf_counter()
        state = runner(query)
        latency = perf_counter() - started

    cost = _sum_cost(state)
    quality = _score_quality(state)
    num_agent_steps = sum(1 for r in state.agent_results if r.agent != AgentName.SUPERVISOR)
    notes_parts = [
        f"agent_steps={num_agent_steps}",
        f"sources={len(state.sources)}",
        f"errors={len(state.errors)}",
    ]
    if state.errors:
        notes_parts.append("err=" + "; ".join(state.errors[:2]))
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost if cost > 0 else None,
        quality_score=quality,
        notes=" | ".join(notes_parts),
    )
    return state, metrics
