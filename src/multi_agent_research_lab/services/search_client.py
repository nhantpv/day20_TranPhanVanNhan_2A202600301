"""Search client abstraction for ResearcherAgent.

If ``TAVILY_API_KEY`` is set and the ``tavily`` package is installed, the
client routes to Tavily. Otherwise it returns a deterministic mock corpus so
the workflow runs offline. The mock filters results by simple keyword overlap
with the query so tests can assert that ranking actually depends on input.
"""

from __future__ import annotations

import logging
import re

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


_MOCK_CORPUS: list[SourceDocument] = [
    SourceDocument(
        title="Building effective agents (Anthropic)",
        url="https://www.anthropic.com/engineering/building-effective-agents",
        snippet=(
            "Practical patterns for building agents: prompt chaining, routing, parallelisation, "
            "orchestrator-workers, evaluator-optimizer. Multi-agent is recommended only when the "
            "task is decomposable; otherwise a single-agent loop is cheaper and more reliable."
        ),
        metadata={"source": "anthropic", "year": 2024, "id": "S1"},
    ),
    SourceDocument(
        title="LangGraph concepts: stateful multi-agent graphs",
        url="https://langchain-ai.github.io/langgraph/concepts/",
        snippet=(
            "LangGraph models multi-agent systems as a directed graph over a shared state. "
            "Nodes are agents/functions; edges express handoffs; conditional edges implement "
            "supervisor routing. Stop conditions and max iterations are first-class."
        ),
        metadata={"source": "langgraph", "year": 2025, "id": "S2"},
    ),
    SourceDocument(
        title="OpenAI Agents SDK: orchestration & handoffs",
        url="https://developers.openai.com/api/docs/guides/agents/orchestration",
        snippet=(
            "Agents can hand off to other agents and share tool results through a managed run "
            "context. The SDK enforces guardrails for cost, latency, and tool budgets, and "
            "exposes traces for every agent step."
        ),
        metadata={"source": "openai", "year": 2025, "id": "S3"},
    ),
    SourceDocument(
        title="GraphRAG: graph-based retrieval-augmented generation",
        url="https://arxiv.org/abs/2404.16130",
        snippet=(
            "GraphRAG builds a knowledge graph over documents and queries it during generation. "
            "Compared to vanilla RAG it improves multi-hop reasoning and source attribution at "
            "the cost of an offline indexing pass."
        ),
        metadata={"source": "arxiv", "year": 2024, "id": "S4"},
    ),
    SourceDocument(
        title="LangSmith tracing for multi-agent systems",
        url="https://docs.smith.langchain.com/",
        snippet=(
            "LangSmith captures per-agent spans, inputs, outputs, and token usage. Useful for "
            "comparing single-agent vs multi-agent runs side by side on the same query."
        ),
        metadata={"source": "langsmith", "year": 2025, "id": "S5"},
    ),
    SourceDocument(
        title="Multi-agent failure modes: loops, drift, and over-coordination",
        url="https://arxiv.org/abs/2402.01030",
        snippet=(
            "Common failure modes in multi-agent LLM systems: routing loops, instruction drift, "
            "context window blow-up. Mitigations: max_iterations guardrail, structured handoff "
            "schemas, and a critic agent for final validation."
        ),
        metadata={"source": "arxiv", "year": 2024, "id": "S6"},
    ),
]


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 2}


class SearchClient:
    """Provider-agnostic search client."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._tavily = None
        if self.settings.tavily_api_key:
            try:
                from tavily import TavilyClient  # type: ignore

                self._tavily = TavilyClient(api_key=self.settings.tavily_api_key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("tavily SDK unavailable, falling back to mock search: %s", exc)
                self._tavily = None

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Return up to ``max_results`` documents ranked by relevance."""

        if self._tavily is not None:
            return self._search_tavily(query, max_results)
        return self._search_mock(query, max_results)

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        assert self._tavily is not None
        raw = self._tavily.search(query=query, max_results=max_results)
        docs: list[SourceDocument] = []
        for item in raw.get("results", []):
            docs.append(
                SourceDocument(
                    title=item.get("title", "untitled"),
                    url=item.get("url"),
                    snippet=item.get("content", ""),
                    metadata={"source": "tavily", "score": item.get("score")},
                )
            )
        return docs

    def _search_mock(self, query: str, max_results: int) -> list[SourceDocument]:
        q_tokens = _tokenize(query)
        scored: list[tuple[int, SourceDocument]] = []
        for doc in _MOCK_CORPUS:
            doc_tokens = _tokenize(doc.title + " " + doc.snippet)
            score = len(q_tokens & doc_tokens)
            scored.append((score, doc))
        # Stable sort: highest overlap first, then original order.
        scored.sort(key=lambda x: x[0], reverse=True)
        # Always return at least one doc so downstream agents have something to work with.
        results = [doc for score, doc in scored if score > 0][:max_results]
        if not results:
            results = [doc for _, doc in scored[:max_results]]
        return results
