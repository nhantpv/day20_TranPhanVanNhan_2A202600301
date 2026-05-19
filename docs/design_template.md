# Design — Multi-Agent Research System

**Student:** Trần Phan Văn Nhân — **ID:** 2A202600301

## Problem

Given an open-ended research query (e.g. *"Research GraphRAG state-of-the-art
and write a 500-word summary"*), produce a well-cited, structured technical
answer for a "technical learners" audience. The system must:

1. Gather sources from a search backend.
2. Distil notes and weigh evidence.
3. Compose a final answer with inline citations `[S1]`, `[S2]`, ...
4. Self-validate (citation coverage, hallucination check) before returning.
5. Stay within a latency / cost / iteration budget.

## Why multi-agent?

Single-agent prompts force one model call to do *search + analysis + writing*
simultaneously. In practice this leads to:

- Skipped or invented citations (the model has no real source list).
- Conflated reasoning steps (it claims and writes in the same paragraph).
- Hard-to-debug failures (one big trace, no per-step state to inspect).

Splitting the task by role gives each agent a focused prompt, a smaller
context window, and a clear handoff schema. The supervisor enforces ordering
and guardrails. Measured on the benchmark query, this raised the quality
score from 6.0 → 9.0 (see `reports/benchmark_report.md`).

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Decide next route, enforce `max_iterations` and timeout | `ResearchState` | new entry in `route_history` | wrong route → loop; mitigated by deterministic policy + iteration cap |
| Researcher | Search + summarise into notes | `request.query`, `max_sources` | `state.sources`, `state.research_notes` | search backend down → `AgentExecutionError`, workflow aborts gracefully |
| Analyst | Extract claims, contradictions, evidence verdict | `research_notes` | `state.analysis_notes` | empty notes → `ValidationError` raised before LLM call |
| Writer | Compose final answer with citations | research + analysis notes + source list | `state.final_answer` | hallucinated citation → caught by critic |
| Critic (optional) | Validate citation coverage and produce verdict | `final_answer`, notes | `AgentResult` with `verdict: accept` or `revise` | flags revise but supervisor caps to one critic pass to avoid loops |

## Shared state

`ResearchState` (see [`core/state.py`](../src/multi_agent_research_lab/core/state.py)):

| Field | Why it exists |
|---|---|
| `request: ResearchQuery` | Original query, audience, source budget — immutable input |
| `iteration: int` | Drives `max_iterations` guardrail in supervisor |
| `route_history: list[str]` | Audit trail; supervisor uses it to detect "critic already ran" |
| `sources: list[SourceDocument]` | Hand-off from researcher → writer; lets writer build the citation list |
| `research_notes`, `analysis_notes`, `final_answer` | Per-stage outputs; supervisor checks which are missing to pick next route |
| `agent_results: list[AgentResult]` | Per-step content + token/cost metadata for the benchmark |
| `trace: list[dict]` | Lightweight in-process trace; mirrors LangSmith span structure |
| `errors: list[str]` | Soft errors so the run can finish and be reported, not crash |

## Routing policy

Deterministic priority cascade (see `SupervisorAgent._decide`):

```text
                  ┌────────────────────────────────┐
                  │ if iteration >= max → done     │
                  ├────────────────────────────────┤
User Query ──►  Supervisor ──► researcher (if research_notes missing)
                  │           ──► analyst    (if analysis_notes missing)
                  │           ──► writer     (if final_answer missing)
                  │           ──► critic     (once, if enabled)
                  │           ──► done
                  └────────────────────────────────┘
```

Each worker writes back to shared state and returns control to the supervisor.

## Guardrails

- **Max iterations:** `MAX_ITERATIONS=6` (env). Supervisor returns `done` and appends an error if exceeded.
- **Timeout:** `TIMEOUT_SECONDS=60` (env). Workflow checks a monotonic deadline between routes.
- **Retry:** LLM client wraps `complete()` with `tenacity` exponential backoff (3 attempts).
- **Fallback:** if `openai` SDK / API key is unavailable, LLM client falls back to a deterministic mock so the workflow still runs end-to-end; same for `tavily`.
- **Validation:** Analyst/Writer/Critic raise `ValidationError` when required upstream fields are missing — fail-fast instead of producing junk.

## Benchmark plan

Query: `Research GraphRAG state-of-the-art and write a 500-word summary`.

| Metric | Expected |
|---|---|
| Quality (length + citations + structure) | Multi-agent ≥ +2.0 over baseline |
| Latency (s) | Multi-agent ~2-3× slower (4-5 calls vs 1) |
| Cost (USD) | Multi-agent ~3× more expensive |
| Citation count | Baseline: 0 inline citations. Multi-agent: ≥ 3 unique `[Sx]` tags |

Outcome (live run on 2026-05-19, model `gpt-4o-mini`):
quality 6.0 → 9.0, latency 11.9s → 33.2s, cost \$0.0005 → \$0.0015. Matches plan.
