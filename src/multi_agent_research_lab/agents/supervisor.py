"""Supervisor / router.

Implements a deterministic routing policy over ``ResearchState``:

1. If ``research_notes`` is missing -> route to ``researcher``.
2. Else if ``analysis_notes`` is missing -> route to ``analyst``.
3. Else if ``final_answer`` is missing -> route to ``writer``.
4. Else, route to ``critic`` once for validation; ``done`` afterwards.

The supervisor enforces ``settings.max_iterations`` so a stuck loop will halt
with an ``errors`` entry instead of burning tokens forever.
"""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


ROUTE_DONE = "done"
VALID_ROUTES = {"researcher", "analyst", "writer", "critic", ROUTE_DONE}


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, enable_critic: bool = True) -> None:
        self.enable_critic = enable_critic
        self.settings = get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("agent.supervisor", {"iteration": state.iteration}) as span:
            route = self._decide(state)
            span["attributes"]["route"] = route
            state.record_route(route)
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.SUPERVISOR,
                    content=f"route -> {route}",
                    metadata={"iteration": state.iteration, "route": route},
                )
            )
            state.add_trace_event("supervisor.decision", {"route": route})
        return state

    def _decide(self, state: ResearchState) -> str:
        if state.iteration >= self.settings.max_iterations:
            state.errors.append(
                f"supervisor: max_iterations ({self.settings.max_iterations}) reached"
            )
            return ROUTE_DONE

        if not state.research_notes:
            return "researcher"
        if not state.analysis_notes:
            return "analyst"
        if not state.final_answer:
            return "writer"
        if self.enable_critic and "critic" not in state.route_history:
            return "critic"
        return ROUTE_DONE
