"""Researcher agent.

Collects sources via the ``SearchClient`` and summarises them into
``state.research_notes`` using the ``LLMClient``. The agent records its own
``AgentResult`` and emits a trace event so the supervisor can see what it did.
"""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "[role=researcher] You are a meticulous research assistant. Read the user query and the "
    "candidate sources, then produce concise notes (5-8 bullet points) that cover: definitions, "
    "key claims, contrasting viewpoints, and gaps. Reference sources inline as [S1], [S2]. "
    "Do not invent sources."
)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
    ) -> None:
        self.llm = llm or LLMClient()
        self.search = search or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("agent.researcher", {"query": state.request.query}) as span:
            try:
                docs = self.search.search(state.request.query, max_results=state.request.max_sources)
            except Exception as exc:  # noqa: BLE001
                state.errors.append(f"researcher.search_failed: {exc}")
                raise AgentExecutionError("researcher search failed") from exc

            state.sources = docs
            corpus = "\n".join(
                f"[S{i + 1}] {d.title}: {d.snippet}" for i, d in enumerate(docs)
            )
            user_prompt = (
                f"Query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Candidate sources:\n{corpus}\n\n"
                "Produce research notes."
            )
            response = self.llm.complete(SYSTEM_PROMPT, user_prompt)
            state.research_notes = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                        "num_sources": len(docs),
                    },
                )
            )
            span["attributes"]["num_sources"] = len(docs)
            state.add_trace_event("researcher.done", {"num_sources": len(docs)})
        return state
