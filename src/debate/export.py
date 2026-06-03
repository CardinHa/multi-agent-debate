"""Markdown export for DebateResult."""
from __future__ import annotations
from .schemas import DebateResult


def debate_to_markdown(result: DebateResult) -> str:
    lines: list[str] = []

    # Title
    lines.append(f"# Debate: {result.question}")
    lines.append("")
    lines.append(f"**Converged:** {'Yes' if result.converged else 'No'}")
    if result.convergence_reason:
        lines.append(f"**Convergence reason:** {result.convergence_reason.value}")
    lines.append(f"**Rounds used:** {result.rounds_used}")
    lines.append("")

    # Transcript
    lines.append("## Transcript")
    lines.append("")
    for turn in result.transcript.turns:
        lines.append(f"### Turn {turn.round_num} — {turn.role.value.capitalize()}")
        lines.append("")
        quoted = "\n".join(f"> {line}" for line in turn.content.splitlines())
        lines.append(quoted)
        lines.append("")

    # Judge verdict
    j = result.judge_output
    lines.append("## Judge Verdict")
    lines.append("")
    lines.append(f"**Verdict:** {j.verdict.value.upper()}")
    lines.append(f"**Confidence:** {j.confidence:.0%}")
    lines.append("")
    if j.final_answer:
        lines.append(f"**Final answer:** {j.final_answer}")
        lines.append("")
    lines.append("**Key reasons:**")
    for reason in j.key_reasons:
        lines.append(f"- {reason}")
    lines.append("")
    if j.unresolved_uncertainties:
        lines.append("**Unresolved uncertainties:**")
        for u in j.unresolved_uncertainties:
            lines.append(f"- {u}")
        lines.append("")
    lines.append(f"**Recency bias check:** {j.recency_bias_check}")
    lines.append("")

    # Graph metrics
    if result.graph_analysis:
        g = result.graph_analysis
        lines.append("## Graph Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Turns | {g.num_turns} |")
        lines.append(f"| Claims | {g.num_claims} |")
        lines.append(f"| Rebuttals | {g.num_rebuttals} |")
        lines.append(f"| Concessions | {g.num_concessions} |")
        lines.append(f"| Revisions | {g.num_revisions} |")
        lines.append(f"| Has cycles | {g.has_cycles} |")
        lines.append(f"| Argument depth | {g.argument_depth} |")
        lines.append("")

    # Token footer
    lines.append("---")
    lines.append("")
    lines.append(
        f"*Tokens — input: {result.total_input_tokens:,} | "
        f"output: {result.total_output_tokens:,}*"
    )
    lines.append("")

    return "\n".join(lines)
