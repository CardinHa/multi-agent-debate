"""A/B debate comparison — run same question under two configs and diff results."""
from __future__ import annotations
from src.debate.schemas import DebateComparison, DebateResult
from src.debate.orchestrator import DebateOrchestrator
from src.debate.utils import BaseLLMClient


def run_comparison(
    question: str,
    client: BaseLLMClient,
    config_a: dict,
    config_b: dict,
    max_rounds: int = 3,
) -> DebateComparison:
    """Run the same question under two orchestrator configurations and return a comparison."""

    def _build_label(cfg: dict) -> str:
        parts = []
        if "skeptic_mode" in cfg and cfg["skeptic_mode"] != "general":
            parts.append(f"skeptic={cfg['skeptic_mode']}")
        if "skeptic_modes" in cfg:
            parts.append(f"panel={','.join(cfg['skeptic_modes'])}")
        if not parts:
            parts.append("general")
        return " | ".join(parts)

    def _run(cfg: dict) -> DebateResult:
        orch = DebateOrchestrator(
            client=client,
            max_rounds=max_rounds,
            save_results=False,
            enable_graph_analysis=False,
            skeptic_mode=cfg.get("skeptic_mode", "general"),
            skeptic_modes=cfg.get("skeptic_modes"),
        )
        return orch.run(question)

    result_a = _run(config_a)
    result_b = _run(config_b)

    return DebateComparison(
        question=question,
        config_a_label=_build_label(config_a),
        config_b_label=_build_label(config_b),
        result_a=result_a,
        result_b=result_b,
        verdict_match=result_a.judge_output.verdict == result_b.judge_output.verdict,
        confidence_delta=result_b.judge_output.confidence - result_a.judge_output.confidence,
        convergence_match=result_a.converged == result_b.converged,
    )


def comparison_to_markdown(comp: DebateComparison) -> str:
    """Render a DebateComparison as Markdown."""
    lines: list[str] = []
    lines += [
        f"# Debate Comparison: {comp.question}",
        "",
        f"| | Config A ({comp.config_a_label}) | Config B ({comp.config_b_label}) |",
        "| --- | --- | --- |",
        f"| Verdict | {comp.result_a.judge_output.verdict.value.upper()} | {comp.result_b.judge_output.verdict.value.upper()} |",
        f"| Confidence | {comp.result_a.judge_output.confidence:.0%} | {comp.result_b.judge_output.confidence:.0%} |",
        f"| Rounds used | {comp.result_a.rounds_used} | {comp.result_b.rounds_used} |",
        f"| Converged | {'Yes' if comp.result_a.converged else 'No'} | {'Yes' if comp.result_b.converged else 'No'} |",
        "",
        "## Analysis",
        "",
        f"**Verdict match:** {'Yes' if comp.verdict_match else 'No — configurations produced different verdicts'}",
        f"**Confidence delta:** {comp.confidence_delta:+.0%} (Config B vs Config A)",
        f"**Convergence match:** {'Yes' if comp.convergence_match else 'No'}",
        "",
        "## Config A Final Answer",
        "",
        f"> {comp.result_a.judge_output.final_answer}",
        "",
        "## Config B Final Answer",
        "",
        f"> {comp.result_b.judge_output.final_answer}",
        "",
    ]
    return "\n".join(lines)
