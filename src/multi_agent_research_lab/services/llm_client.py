"""LLM client abstraction.

Provides a provider-agnostic ``LLMClient``. Agents should depend on this
interface instead of importing an SDK directly.

The implementation here has two modes:

* If ``OPENAI_API_KEY`` is set and the ``openai`` package is installed, the
  client routes requests to OpenAI's Chat Completions API.
* Otherwise, it falls back to a deterministic local "mock" LLM that is good
  enough for the lab smoke test, unit tests, and offline benchmarking.

Token/cost accounting is approximate but consistent across modes so the
benchmark report can compare apples to apples.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


# Rough OpenAI gpt-4o-mini pricing as of 2025-Q4 (USD per 1K tokens).
_PRICE_PER_1K_IN = 0.00015
_PRICE_PER_1K_OUT = 0.00060


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def _estimate_tokens(text: str) -> int:
    # ~4 chars per token is the usual rule of thumb for English.
    return max(1, len(text) // 4)


def _estimate_cost(in_tok: int, out_tok: int) -> float:
    return (in_tok / 1000.0) * _PRICE_PER_1K_IN + (out_tok / 1000.0) * _PRICE_PER_1K_OUT


class LLMClient:
    """Provider-agnostic LLM client.

    Real provider calls go through ``_complete_openai``. When no key is
    configured, ``_complete_mock`` produces deterministic, structured output
    that the downstream agents can still parse.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._openai_client = None
        if self.settings.openai_api_key:
            try:
                from openai import OpenAI  # type: ignore

                self._openai_client = OpenAI(api_key=self.settings.openai_api_key)
            except Exception as exc:  # noqa: BLE001 - we want to fall back
                logger.warning("openai SDK unavailable, falling back to mock LLM: %s", exc)
                self._openai_client = None

    @retry(wait=wait_exponential(multiplier=0.5, max=4), stop=stop_after_attempt(3), reraise=True)
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion. Retries with exponential backoff."""

        started = time.perf_counter()
        if self._openai_client is not None:
            response = self._complete_openai(system_prompt, user_prompt)
        else:
            response = self._complete_mock(system_prompt, user_prompt)
        logger.debug(
            "llm.complete done in %.3fs (in=%s,out=%s)",
            time.perf_counter() - started,
            response.input_tokens,
            response.output_tokens,
        )
        return response

    def _complete_openai(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        assert self._openai_client is not None
        completion = self._openai_client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=self.settings.timeout_seconds,
        )
        choice = completion.choices[0].message.content or ""
        usage = getattr(completion, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", _estimate_tokens(system_prompt + user_prompt))
        out_tok = getattr(usage, "completion_tokens", _estimate_tokens(choice))
        return LLMResponse(
            content=choice,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=_estimate_cost(in_tok, out_tok),
        )

    def _complete_mock(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Deterministic template-based response.

        The mock detects the agent role from a tag in the system prompt
        (``[role=...]``) and returns content the next agent can consume.
        This keeps the workflow exercised end-to-end without external calls.
        """

        role = "writer"
        marker = "[role="
        if marker in system_prompt:
            start = system_prompt.index(marker) + len(marker)
            end = system_prompt.index("]", start)
            role = system_prompt[start:end].strip().lower()

        query = user_prompt.strip().splitlines()[0][:200]

        if role == "researcher":
            content = (
                f"Research notes for: {query}\n"
                "- Definition: A multi-agent system coordinates several LLM-backed agents "
                "with distinct roles to solve a task collaboratively.\n"
                "- Pattern: Supervisor + workers (researcher, analyst, writer) with shared state.\n"
                "- Tradeoffs: higher quality + traceability vs. higher latency and cost.\n"
                "- Sources cited inline as [S1], [S2], ..."
            )
        elif role == "analyst":
            content = (
                "Analysis:\n"
                "1. Key claims:\n"
                "   - Specialised agents reduce instruction overload per call.\n"
                "   - Supervisor routing improves reliability over a single mega-prompt.\n"
                "2. Counter-points / risks:\n"
                "   - Multi-agent setups add latency and cost; only useful when the task "
                "is decomposable.\n"
                "   - Coordination errors (loops, drift) require guardrails: max_iterations, "
                "timeouts, validation.\n"
                "3. Evidence quality: medium — relies on synthesis of [S1], [S2], [S3]."
            )
        elif role == "writer":
            content = (
                f"# Answer\n\n"
                f"**Query:** {query}\n\n"
                "Multi-agent research systems split a complex query across specialised "
                "agents — a researcher gathers sources, an analyst extracts claims, and a "
                "writer composes the final answer — coordinated by a supervisor that decides "
                "what runs next and when to stop. This trades extra latency and cost for "
                "better traceability and answer quality versus a single-agent baseline.\n\n"
                "Sources: [S1], [S2], [S3]."
            )
        elif role == "critic":
            content = (
                "Critic review:\n"
                "- Citations present: yes\n"
                "- Hallucination risk: low (claims align with provided notes)\n"
                "- Recommended action: accept"
            )
        elif role == "baseline":
            content = (
                f"# Single-agent answer\n\n"
                f"**Query:** {query}\n\n"
                "A single LLM call handles search + analysis + writing in one shot. "
                "It is faster and cheaper but tends to skip citations and conflate steps, "
                "which is exactly the gap multi-agent systems address."
            )
        else:
            content = f"[mock-{role}] {query}"

        in_tok = _estimate_tokens(system_prompt + user_prompt)
        out_tok = _estimate_tokens(content)
        return LLMResponse(
            content=content,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=_estimate_cost(in_tok, out_tok),
        )


def is_live_mode() -> bool:
    """Return True when the LLM client will hit a real provider."""

    return bool(os.environ.get("OPENAI_API_KEY") or get_settings().openai_api_key)
