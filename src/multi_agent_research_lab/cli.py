"""Command-line entrypoint for the multi-agent research lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    BenchmarkMetrics,
    ResearchQuery,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient, is_live_mode
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


BASELINE_SYSTEM_PROMPT = (
    "[role=baseline] You are a single research assistant who must research, analyse, and write "
    "the answer in one shot. Cite sources as [S1], [S2] if available. Be thorough but concise."
)


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def run_baseline(query: str) -> ResearchState:
    """Single-agent baseline: one LLM call, no search, no router."""

    state = ResearchState(request=ResearchQuery(query=query))
    llm = LLMClient()
    user_prompt = (
        f"Query: {query}\n"
        f"Audience: {state.request.audience}\n"
        "Write the answer."
    )
    response = llm.complete(BASELINE_SYSTEM_PROMPT, user_prompt)
    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "mode": "baseline",
            },
        )
    )
    state.add_trace_event("baseline.done", {"chars": len(response.content)})
    return state


def run_multi_agent(query: str, max_sources: int = 5) -> ResearchState:
    """Multi-agent workflow: supervisor + researcher + analyst + writer + critic."""

    state = ResearchState(request=ResearchQuery(query=query, max_sources=max_sources))
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


def _print_state(state: ResearchState, title: str) -> None:
    console.print(Panel.fit(state.final_answer or "(no answer)", title=title))
    table = Table(title="Route history")
    table.add_column("#")
    table.add_column("route")
    for i, route in enumerate(state.route_history, start=1):
        table.add_row(str(i), route)
    if state.route_history:
        console.print(table)
    if state.errors:
        console.print(Panel.fit("\n".join(state.errors), title="Errors", style="red"))


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline."""

    _init()
    state = run_baseline(query)
    _print_state(state, "Single-Agent Baseline")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    max_sources: Annotated[int, typer.Option("--max-sources", "-n")] = 5,
    json_out: Annotated[bool, typer.Option("--json", help="Print full state as JSON")] = False,
) -> None:
    """Run the multi-agent workflow."""

    _init()
    state = run_multi_agent(query, max_sources=max_sources)
    if json_out:
        console.print(state.model_dump_json(indent=2))
    else:
        _print_state(state, "Multi-Agent Answer")


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to write the markdown report")
    ] = Path("reports/benchmark_report.md"),
) -> None:
    """Run baseline + multi-agent and emit a comparison report."""

    _init()
    console.print(
        Panel.fit(
            f"mode: {'live (OpenAI)' if is_live_mode() else 'mock (deterministic)'}",
            title="LLM mode",
        )
    )

    baseline_state, baseline_metrics = run_benchmark("single-agent", query, run_baseline)
    multi_state, multi_metrics = run_benchmark("multi-agent", query, run_multi_agent)

    extra = {
        "Single-Agent Output (excerpt)": (baseline_state.final_answer or "(none)")[:1500],
        "Multi-Agent Output (excerpt)": (multi_state.final_answer or "(none)")[:1500],
        "Multi-Agent Route History": " -> ".join(multi_state.route_history) or "(none)",
    }
    if multi_state.errors:
        extra["Multi-Agent Errors"] = "\n".join(f"- {e}" for e in multi_state.errors)

    report = render_markdown_report(
        [baseline_metrics, multi_metrics], query=query, extra_sections=extra
    )
    store = LocalArtifactStore(root=output.parent if output.parent.parts else Path("reports"))
    path = store.write_text(output.name, report)

    table = Table(title="Benchmark summary")
    for col in ("run", "latency (s)", "cost (USD)", "quality", "notes"):
        table.add_column(col)
    for m in (baseline_metrics, multi_metrics):
        table.add_row(
            m.run_name,
            f"{m.latency_seconds:.2f}",
            "-" if m.estimated_cost_usd is None else f"{m.estimated_cost_usd:.4f}",
            "-" if m.quality_score is None else f"{m.quality_score:.1f}",
            m.notes,
        )
    console.print(table)
    console.print(f"Report written to [bold]{path}[/bold]")

    # Also drop the raw multi-agent trace next to the report for inspection.
    trace_path = store.write_text(
        Path(output.stem + "_trace.json").name,
        json.dumps(multi_state.trace, indent=2, default=str),
    )
    console.print(f"Trace written to [bold]{trace_path}[/bold]")


if __name__ == "__main__":
    app()
