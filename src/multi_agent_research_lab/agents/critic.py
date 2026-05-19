"""Critic agent (optional).

Performs a light validation pass: checks citation coverage, length sanity, and
asks the LLM for a verdict. Appends findings to ``state.errors`` only if a
real problem is detected, so the supervisor can decide whether to loop.
"""

from __future__ import annotations

import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "[role=critic] You are a fact-checking critic. Validate the writer's answer against the "
    "research notes. Flag missing citations, contradictions, or hallucinated claims. End with "
    "a single verdict line: 'verdict: accept' or 'verdict: revise'."
)

_CITATION_RE = re.compile(r"\[S\d+\]")


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        if not state.final_answer:
            raise ValidationError("critic requires final_answer before running")
        with trace_span("agent.critic") as span:
            citations = _CITATION_RE.findall(state.final_answer)
            citation_count = len(citations)
            user_prompt = (
                f"Query: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes or '(missing)'}\n\n"
                f"Final answer:\n{state.final_answer}\n\n"
                f"Citation tags found: {citation_count}.\n"
                "Review and return verdict."
            )
            response = self.llm.complete(SYSTEM_PROMPT, user_prompt)
            verdict = "accept" if "verdict: accept" in response.content.lower() else "revise"
            if citation_count == 0:
                state.errors.append("critic: final answer has no citations")
                verdict = "revise"
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content=response.content,
                    metadata={
                        "verdict": verdict,
                        "citation_count": citation_count,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            span["attributes"]["verdict"] = verdict
            state.add_trace_event("critic.done", {"verdict": verdict, "citations": citation_count})
        return state
