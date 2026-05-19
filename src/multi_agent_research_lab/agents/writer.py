"""Writer agent.

Synthesises research + analysis notes into ``state.final_answer`` with inline
citations referencing the documents in ``state.sources``.
"""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "[role=writer] You are a technical writer. Given research notes and analysis, write a "
    "clear, well-structured answer for the requested audience. Use inline citations [S1], [S2] "
    "that match the provided source list. Aim for ~400-600 words unless the user asked otherwise."
)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        if not state.research_notes:
            raise ValidationError("writer requires research_notes before running")
        with trace_span("agent.writer") as span:
            sources_block = "\n".join(
                f"[S{i + 1}] {d.title} — {d.url or 'n/a'}"
                for i, d in enumerate(state.sources)
            )
            user_prompt = (
                f"Query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Research notes:\n{state.research_notes}\n\n"
                f"Analysis:\n{state.analysis_notes or '(none)'}\n\n"
                f"Sources:\n{sources_block}\n\n"
                "Write the final answer."
            )
            response = self.llm.complete(SYSTEM_PROMPT, user_prompt)
            state.final_answer = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                        "num_citations": response.content.count("[S"),
                    },
                )
            )
            span["attributes"]["chars"] = len(response.content)
            state.add_trace_event("writer.done", {"chars": len(response.content)})
        return state
