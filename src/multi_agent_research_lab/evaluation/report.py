"""Benchmark report rendering."""

from __future__ import annotations

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    query: str | None = None,
    extra_sections: dict[str, str] | None = None,
) -> str:
    """Render benchmark metrics to a markdown report.

    Args:
        metrics: One row per run (e.g. baseline + multi-agent).
        query: Optional query string to include in the header.
        extra_sections: Optional ``{title: body}`` sections appended at the end
            (e.g. failure mode analysis, sample outputs).
    """

    lines: list[str] = ["# Benchmark Report", ""]
    if query:
        lines += [f"**Query:** {query}", ""]
    lines += [
        "| Run | Latency (s) | Cost (USD) | Quality (0-10) | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "-" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "-" if item.quality_score is None else f"{item.quality_score:.1f}"
        notes = item.notes.replace("|", "\\|")
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {notes} |"
        )

    if len(metrics) >= 2:
        a, b = metrics[0], metrics[1]
        lines += ["", "## Headline comparison", ""]
        if a.quality_score is not None and b.quality_score is not None:
            delta_q = b.quality_score - a.quality_score
            lines.append(
                f"- Quality: **{b.run_name}** {b.quality_score:.1f} vs "
                f"**{a.run_name}** {a.quality_score:.1f} (Δ {delta_q:+.1f})"
            )
        lines.append(
            f"- Latency: **{b.run_name}** {b.latency_seconds:.2f}s vs "
            f"**{a.run_name}** {a.latency_seconds:.2f}s "
            f"(Δ {b.latency_seconds - a.latency_seconds:+.2f}s)"
        )
        if a.estimated_cost_usd and b.estimated_cost_usd:
            lines.append(
                f"- Cost: **{b.run_name}** ${b.estimated_cost_usd:.4f} vs "
                f"**{a.run_name}** ${a.estimated_cost_usd:.4f} "
                f"(Δ ${b.estimated_cost_usd - a.estimated_cost_usd:+.4f})"
            )

    if extra_sections:
        for title, body in extra_sections.items():
            lines += ["", f"## {title}", "", body.strip()]

    return "\n".join(lines) + "\n"
