"""Multi-agent workflow.

A LangGraph-style state machine over ``ResearchState``. We don't pull in the
``langgraph`` runtime by default (it is an optional extra), but we mirror the
same shape: a graph of node functions keyed by name, with conditional edges
driven by the supervisor's routing decision.

If ``langgraph`` is installed the same workflow could be expressed there; the
code is intentionally simple so students can swap in ``StateGraph`` if they
want to extend the lab.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.supervisor import ROUTE_DONE
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


NodeFn = Callable[[ResearchState], ResearchState]


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Orchestration lives here; agent internals live in ``agents/``.
    """

    def __init__(self, enable_critic: bool = True) -> None:
        self.enable_critic = enable_critic
        self.settings = get_settings()
        self.supervisor = SupervisorAgent(enable_critic=enable_critic)
        self.nodes: dict[str, BaseAgent] = {}

    def build(self) -> dict[str, BaseAgent]:
        """Create the node table. Returns the dict so tests can introspect it."""

        self.nodes = {
            "researcher": ResearcherAgent(),
            "analyst": AnalystAgent(),
            "writer": WriterAgent(),
        }
        if self.enable_critic:
            self.nodes["critic"] = CriticAgent()
        return self.nodes

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow until the supervisor returns ``done``.

        Respects ``settings.max_iterations`` and ``settings.timeout_seconds``
        as hard guardrails so a misbehaving agent cannot loop forever.
        """

        if not self.nodes:
            self.build()

        deadline = time.monotonic() + self.settings.timeout_seconds
        while True:
            if time.monotonic() > deadline:
                state.errors.append(
                    f"workflow: timeout after {self.settings.timeout_seconds}s"
                )
                break

            self.supervisor.run(state)
            next_route = state.route_history[-1]
            logger.info("workflow route -> %s (iter=%d)", next_route, state.iteration)

            if next_route == ROUTE_DONE:
                break

            agent = self.nodes.get(next_route)
            if agent is None:
                state.errors.append(f"workflow: unknown route '{next_route}'")
                break

            try:
                agent.run(state)
            except AgentExecutionError as exc:
                state.errors.append(f"{next_route}: {exc}")
                # Fallback: bail out gracefully instead of looping forever.
                break
            except Exception as exc:  # noqa: BLE001 - keep workflow alive for the report
                state.errors.append(f"{next_route}: unhandled {type(exc).__name__}: {exc}")
                break

            if state.iteration >= self.settings.max_iterations:
                break

        return state
