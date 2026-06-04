"""CLI entry point for the Multi-Agent Debate System."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(
    name="debate",
    help="Multi-Agent Debate System — scalable oversight via adversarial LLM debate.",
    add_completion=False,
)
console = Console()


def _print_turn(role: str, round_num: int, content: str) -> None:
    colors = {"proposer": "green", "skeptic": "red", "judge": "blue"}
    color = colors.get(role.lower(), "white")
    title = f"[bold {color}]{role.upper()}[/] — Round {round_num}"
    console.print(Panel(content, title=title, border_style=color, padding=(1, 2)))


@app.command()
def debate(
    question: str = typer.Argument(..., help="Question or claim to debate."),
    rounds: int = typer.Option(3, "--rounds", "-r", help="Maximum debate rounds."),
    model: str = typer.Option(
        "claude-3-5-sonnet-latest", "--model", "-m", help="Claude model name."
    ),
    temperature: float = typer.Option(0.7, "--temperature", "-t"),
    graph: bool = typer.Option(False, "--graph", "-g", help="Enable graph analysis."),
    graph_viz: bool = typer.Option(False, "--graph-viz", help="Save graph PNG."),
    save: bool = typer.Option(True, "--save/--no-save", help="Save result JSON."),
    mock: bool = typer.Option(False, "--mock", help="Use mock LLM (no API calls)."),
    skeptic_mode: str = typer.Option(
        "general", "--skeptic-mode", "-s",
        help="Skeptic personality: general | factual | logic | evidence | safety",
    ),
    export: str = typer.Option("", "--export", "-e", help="Export debate result to Markdown file"),
    panel: str = typer.Option(
        "", "--panel",
        help="Comma-separated skeptic modes for panel debate, e.g. 'logic,evidence'",
    ),
) -> None:
    """Run a single multi-agent debate for a question or claim."""
    from src.debate.orchestrator import DebateOrchestrator
    from src.debate.utils import MockLLMClient

    console.print()
    console.rule("[bold cyan]Multi-Agent Debate System[/]")
    console.print(Panel(question, title="[bold]Question / Claim[/]", border_style="cyan"))
    if skeptic_mode != "general":
        console.print(f"[dim]Skeptic mode: [yellow]{skeptic_mode}[/][/]")

    client = MockLLMClient() if mock else None

    skeptic_modes_list = [m.strip() for m in panel.split(",") if m.strip()] if panel else None

    orchestrator = DebateOrchestrator(
        client=client,
        model=model,
        temperature=temperature,
        max_rounds=rounds,
        enable_graph_analysis=graph,
        save_results=save,
        skeptic_mode=skeptic_mode,
        skeptic_modes=skeptic_modes_list,
    )

    with console.status("[cyan]Running debate...[/]"):
        result = orchestrator.run(question)

    # Print transcript
    console.print("\n[bold]Debate Transcript[/]\n")
    for turn in result.transcript.turns:
        _print_turn(turn.role.value, turn.round_num, turn.content)

    # Convergence
    if result.converged:
        console.print(
            f"\n[yellow]Debate converged after round {result.rounds_used}[/] "
            f"([dim]{result.convergence_reason.value}[/])"
        )
    else:
        console.print(f"\n[dim]Ran full {result.rounds_used} round(s) (max rounds reached).[/]")

    if result.panel_mode:
        console.print(f"[dim]Panel mode: {', '.join(m for m in skeptic_modes_list)}[/]")

    # Judge verdict
    j = result.judge_output
    verdict_color = {"supported": "green", "refuted": "red", "uncertain": "yellow"}.get(
        j.verdict.value if hasattr(j.verdict, 'value') else j.verdict, "white"
    )
    verdict_panel = (
        f"[bold]Final Answer:[/] {j.final_answer}\n\n"
        f"[bold]Verdict:[/] [{verdict_color}]{j.verdict.value if hasattr(j.verdict, 'value') else j.verdict}[/]   "
        f"[bold]Confidence:[/] {j.confidence:.0%}\n\n"
        f"[bold]Key Reasons:[/]\n" + "\n".join(f"  • {r}" for r in j.key_reasons) +
        (
            f"\n\n[bold]Unresolved Uncertainties:[/]\n" +
            "\n".join(f"  • {u}" for u in j.unresolved_uncertainties)
            if j.unresolved_uncertainties else ""
        ) +
        f"\n\n[dim]Recency bias check:[/] {j.recency_bias_check}"
    )
    console.print(Panel(verdict_panel, title="[bold blue]Judge Verdict[/]", border_style="blue"))

    # Graph analysis
    if result.graph_analysis:
        ga = result.graph_analysis
        table = Table(title="Debate Graph Metrics", box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        for field, value in ga.model_dump().items():
            if field not in ("centrality_scores", "edge_type_counts"):
                table.add_row(field.replace("_", " ").title(), str(value))
        console.print(table)

        if graph_viz:
            from src.debate.graph import DebateGraphBuilder, GraphAnalyzer
            builder = DebateGraphBuilder()
            nx_graph = builder.build(result.transcript)
            analyzer = GraphAnalyzer(nx_graph, result.transcript)
            viz_path = "results/debate_graph.png"
            analyzer.visualize(viz_path)
            console.print(f"[green]Graph saved to {viz_path}[/]")

    # Token summary
    console.print(
        f"\n[dim]Tokens — input: {result.total_input_tokens:,} | "
        f"output: {result.total_output_tokens:,}[/]"
    )

    if export:
        from src.debate.export import debate_to_markdown
        export_path = Path(export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(debate_to_markdown(result), encoding="utf-8")
        console.print(f"[green]Exported to {export}[/green]")


@app.command()
def benchmark(
    dataset: str = typer.Option(
        "data/sample_claims.jsonl", "--dataset", "-d", help="Path to JSONL dataset."
    ),
    model: str = typer.Option("claude-3-5-sonnet-latest", "--model", "-m"),
    rounds: int = typer.Option(3, "--rounds", "-r"),
    mock: bool = typer.Option(False, "--mock", help="Use mock LLM (no API calls)."),
    report: bool = typer.Option(False, "--report", help="Save Markdown calibration report."),
) -> None:
    """Run benchmark comparing single-agent baseline vs debate system."""
    from src.debate.benchmark import BenchmarkRunner
    from src.debate.utils import MockLLMClient

    console.print()
    console.rule("[bold cyan]Benchmark: Debate vs Baseline[/]")
    console.print(f"Dataset: [cyan]{dataset}[/]  Model: [cyan]{model}[/]  Max rounds: [cyan]{rounds}[/]\n")

    client = MockLLMClient() if mock else None
    runner = BenchmarkRunner(client=client, model=model, max_rounds=rounds)

    with console.status("[cyan]Running benchmark...[/]"):
        results = runner.run(dataset)

    summary = runner.summary(results)
    base_path = runner.save_results(results)

    # Summary table
    table = Table(title="Benchmark Summary", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold white")
    for key, value in summary.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    console.print(table)
    console.print(f"\n[green]Results saved to {base_path}[/]")

    if report:
        from datetime import datetime
        cal = runner.calibration_report(results)
        lines = [
            "# Calibration Report", "",
            f"**Total examples:** {cal.total_examples}",
            f"**Overall baseline accuracy:** {cal.overall_baseline_accuracy:.1%}",
            f"**Overall debate accuracy:** {cal.overall_debate_accuracy:.1%}",
            f"**Overall improvement rate:** {cal.overall_improvement_rate:.1%}",
            "",
            "## Per-Category Breakdown", "",
            "| Category | Total | Baseline Acc | Debate Acc | Improvement |",
            "| --- | --- | --- | --- | --- |",
        ]
        for s in cal.per_category:
            lines.append(
                f"| {s.category} | {s.total} | {s.baseline_accuracy:.1%} "
                f"| {s.debate_accuracy:.1%} | {s.improvement_rate:.1%} |"
            )
        lines += [
            "",
            "## Confidence Calibration", "",
            "| Confidence Range | Count | Actual Accuracy |",
            "| --- | --- | --- |",
        ]
        for b in cal.calibration_bins:
            lines.append(
                f"| {b.confidence_low:.1f}–{b.confidence_high:.1f} "
                f"| {b.count} | {b.actual_accuracy:.1%} |"
            )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cal_path = Path(f"results/calibration_{ts}.md")
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        cal_path.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"[green]Calibration report saved to {cal_path}[/]")


if __name__ == "__main__":
    app()
