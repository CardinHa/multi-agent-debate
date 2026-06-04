"""Self-contained HTML export for DebateResult."""
from __future__ import annotations
import html
from .schemas import DebateResult

_CSS = """
body { font-family: system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #1e293b; background: #f8fafc; }
h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
h2 { font-size: 1.1rem; color: #475569; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; margin-top: 2rem; }
.meta { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1.5rem; }
.badge { padding: 3px 10px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; }
.badge-gray { background: #e2e8f0; color: #475569; }
.badge-green { background: #dcfce7; color: #166534; }
.badge-yellow { background: #fef9c3; color: #854d0e; }
.turn { border-radius: 8px; margin: 12px 0; overflow: hidden; border: 1px solid #e2e8f0; }
.turn-header { padding: 8px 14px; font-weight: 700; font-size: 0.85rem; color: #fff; }
.turn-body { padding: 14px; background: #fff; white-space: pre-wrap; font-size: 0.95rem; line-height: 1.6; }
.turn-proposer .turn-header { background: #22c55e; }
.turn-skeptic  .turn-header { background: #ef4444; }
.turn-judge    .turn-header { background: #3b82f6; }
.verdict-badge { display: inline-block; padding: 4px 14px; border-radius: 6px; font-weight: 700; color: #fff; }
.verdict-supported { background: #22c55e; }
.verdict-refuted   { background: #ef4444; }
.verdict-uncertain { background: #f59e0b; }
.confidence-wrap { background: #e2e8f0; border-radius: 4px; height: 12px; margin: 8px 0 16px; }
.confidence-bar  { background: #3b82f6; height: 100%; border-radius: 4px; }
table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
td, th { padding: 8px 12px; border: 1px solid #e2e8f0; text-align: left; font-size: 0.9rem; }
th { background: #f1f5f9; font-weight: 600; }
footer { margin-top: 2rem; font-size: 0.8rem; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 1rem; }
ul { margin: 0.5rem 0; padding-left: 1.5rem; }
li { margin: 4px 0; }
""".strip()


def debate_to_html(result: DebateResult) -> str:
    """Render a DebateResult as a self-contained HTML string."""
    q = html.escape(result.question)

    # --- metadata badges ---
    converged_badge = (
        '<span class="badge badge-green">Converged</span>'
        if result.converged
        else '<span class="badge badge-gray">Not Converged</span>'
    )
    rounds_badge = f'<span class="badge badge-gray">Rounds: {result.rounds_used}</span>'
    panel_badge = '<span class="badge badge-gray">Panel Mode</span>' if result.panel_mode else ""
    if result.convergence_reason:
        conv_reason = (
            f'<span class="badge badge-gray">{html.escape(result.convergence_reason.value)}</span>'
        )
    else:
        conv_reason = ""

    meta_parts = [converged_badge, rounds_badge]
    if panel_badge:
        meta_parts.append(panel_badge)
    if conv_reason:
        meta_parts.append(conv_reason)
    meta_html = "\n    ".join(meta_parts)

    # --- transcript turns ---
    turns_html_parts: list[str] = []
    for turn in result.transcript.turns:
        role_val = turn.role.value  # "proposer" | "skeptic" | "judge"
        role_label = html.escape(role_val.capitalize())
        content_escaped = html.escape(turn.content)
        token_info = f" ({turn.token_count} tokens)" if turn.token_count is not None else ""
        turns_html_parts.append(
            f'<div class="turn turn-{role_val}">\n'
            f'  <div class="turn-header">{role_label} — Round {turn.round_num}{html.escape(token_info)}</div>\n'
            f'  <div class="turn-body">{content_escaped}</div>\n'
            f'</div>'
        )
    turns_html = "\n".join(turns_html_parts)

    # --- judge verdict ---
    j = result.judge_output
    verdict_val = j.verdict.value if hasattr(j.verdict, "value") else str(j.verdict)
    verdict_class = f"verdict-{verdict_val}"
    verdict_label = html.escape(verdict_val.upper())
    confidence_pct = int(j.confidence * 100)
    final_answer_escaped = html.escape(j.final_answer)

    reasons_html = ""
    if j.key_reasons:
        items = "\n".join(f"  <li>{html.escape(r)}</li>" for r in j.key_reasons)
        reasons_html = f"<p><strong>Key Reasons:</strong></p>\n<ul>\n{items}\n</ul>"

    uncertainties_html = ""
    if j.unresolved_uncertainties:
        items = "\n".join(f"  <li>{html.escape(u)}</li>" for u in j.unresolved_uncertainties)
        uncertainties_html = f"<p><strong>Unresolved Uncertainties:</strong></p>\n<ul>\n{items}\n</ul>"

    recency_escaped = html.escape(j.recency_bias_check)

    verdict_html = f"""\
<p>
  <span class="verdict-badge {verdict_class}">{verdict_label}</span>
  &nbsp; Confidence: <strong>{confidence_pct}%</strong>
</p>
<div class="confidence-wrap">
  <div class="confidence-bar" style="width: {confidence_pct}%"></div>
</div>
<p><strong>Final Answer:</strong> {final_answer_escaped}</p>
{reasons_html}
{uncertainties_html}
<p><em>Recency bias check:</em> {recency_escaped}</p>"""

    # --- graph metrics table ---
    graph_section = ""
    if result.graph_analysis:
        g = result.graph_analysis
        rows = [
            ("Num Turns", g.num_turns),
            ("Num Claims", g.num_claims),
            ("Num Rebuttals", g.num_rebuttals),
            ("Num Concessions", g.num_concessions),
            ("Num Revisions", g.num_revisions),
            ("Argument Depth", g.argument_depth),
            ("Has Cycles", "Yes" if g.has_cycles else "No"),
        ]
        row_html = "\n".join(
            f"  <tr><td>{html.escape(str(label))}</td><td>{html.escape(str(value))}</td></tr>"
            for label, value in rows
        )
        graph_section = f"""\
<section id="graph">
  <h2>Graph Metrics</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>
{row_html}
    </tbody>
  </table>
</section>"""

    # --- assemble ---
    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Debate: {q}</title>
  <style>
{_CSS}
  </style>
</head>
<body>
  <h1>Debate: {q}</h1>
  <div class="meta">
    {meta_html}
  </div>

  <section id="transcript">
    <h2>Transcript</h2>
    {turns_html}
  </section>

  <section id="verdict">
    <h2>Judge Verdict</h2>
    {verdict_html}
  </section>

  {graph_section}

  <footer>Tokens: input {result.total_input_tokens:,} | output {result.total_output_tokens:,}</footer>
</body>
</html>"""
