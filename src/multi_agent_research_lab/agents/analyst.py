"""Analyst agent.

Reads ``state.research_notes`` and produces ``state.analysis_notes`` with
structured insights: key claims, contrasting viewpoints, evidence quality.
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
    "[role=analyst] You are a critical analyst. Given research notes, extract: "
    "(1) the strongest claims, (2) any contradictions or counter-points, "
    "(3) a one-line evidence-quality verdict (low/medium/high). Be concise."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        if not state.research_notes:
            raise ValidationError("analyst requires research_notes before running")
        with trace_span("agent.analyst") as span:
            user_prompt = (
                f"Query: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes}\n\n"
                "Produce the structured analysis."
            )
            response = self.llm.complete(SYSTEM_PROMPT, user_prompt)
            state.analysis_notes = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            span["attributes"]["chars"] = len(response.content)
            state.add_trace_event("analyst.done", {"chars": len(response.content)})
        return state
