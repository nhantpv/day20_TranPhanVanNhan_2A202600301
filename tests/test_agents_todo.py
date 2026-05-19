"""Tests for the implemented agents and workflow.

The original skeleton expected ``SupervisorAgent`` to raise ``StudentTodoError``
until students filled it in. Now that the agents are implemented, the tests
verify real routing + workflow behaviour instead.
"""

from __future__ import annotations

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.supervisor import ROUTE_DONE
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def _fresh_state(query: str = "Explain multi-agent systems") -> ResearchState:
    return ResearchState(request=ResearchQuery(query=query))


def test_supervisor_routes_to_researcher_first() -> None:
    state = _fresh_state()
    SupervisorAgent().run(state)
    assert state.route_history == ["researcher"]


def test_supervisor_routes_to_analyst_after_research() -> None:
    state = _fresh_state()
    state.research_notes = "some notes"
    SupervisorAgent().run(state)
    assert state.route_history[-1] == "analyst"


def test_supervisor_finishes_after_critic() -> None:
    state = _fresh_state()
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "answer [S1]"
    state.route_history = ["researcher", "analyst", "writer", "critic"]
    SupervisorAgent().run(state)
    assert state.route_history[-1] == ROUTE_DONE


def test_researcher_populates_sources_and_notes() -> None:
    state = _fresh_state("Explain LangGraph multi-agent routing")
    ResearcherAgent().run(state)
    assert len(state.sources) > 0
    assert state.research_notes is not None


def test_analyst_requires_research_notes() -> None:
    state = _fresh_state()
    import pytest

    from multi_agent_research_lab.core.errors import ValidationError

    with pytest.raises(ValidationError):
        AnalystAgent().run(state)


def test_writer_emits_final_answer() -> None:
    state = _fresh_state()
    ResearcherAgent().run(state)
    AnalystAgent().run(state)
    WriterAgent().run(state)
    assert state.final_answer
    assert "[S" in state.final_answer  # citations included


def test_critic_marks_verdict() -> None:
    state = _fresh_state()
    ResearcherAgent().run(state)
    AnalystAgent().run(state)
    WriterAgent().run(state)
    CriticAgent().run(state)
    verdicts = [r.metadata.get("verdict") for r in state.agent_results if r.agent == "critic"]
    assert verdicts and verdicts[0] in {"accept", "revise"}


def test_workflow_runs_end_to_end() -> None:
    state = _fresh_state("Research GraphRAG state-of-the-art")
    workflow = MultiAgentWorkflow()
    workflow.run(state)
    assert state.final_answer
    assert ROUTE_DONE in state.route_history
    # supervisor + at least researcher/analyst/writer
    assert {"researcher", "analyst", "writer"}.issubset(set(state.route_history))
