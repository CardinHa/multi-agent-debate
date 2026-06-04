"""BenchmarkRunner — compares single-agent baseline vs debate system."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from src.debate.schemas import BenchmarkExample, BenchmarkResult, AgentRole
from src.debate.orchestrator import DebateOrchestrator
from src.debate.utils import BaseLLMClient, AnthropicClient
from src.debate.prompts import PROPOSER_SYSTEM_PROMPT


def _load_examples(path: str) -> list[BenchmarkExample]:
    examples = []
    for line in Path(path).read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if line:
            data = json.loads(line)
            examples.append(BenchmarkExample(**data))
    return examples


def _baseline_answer(client: BaseLLMClient, question: str) -> tuple[str, int]:
    """Get a single-agent answer using the Proposer system prompt."""
    text, inp, out = client.call(
        system=PROPOSER_SYSTEM_PROMPT,
        user=f"Question: {question}\n\nProvide your best answer.",
    )
    return text, inp + out


def _answers_match(answer: str, ground_truth: str) -> bool:
    """
    Simple keyword-overlap correctness check.
    Returns True if the answer overlaps substantially with ground truth key terms.
    """
    a_tokens = set(answer.lower().split())
    g_tokens = set(ground_truth.lower().split())
    overlap = len(a_tokens & g_tokens) / max(len(g_tokens), 1)
    return overlap > 0.35


class BenchmarkRunner:
    """Runs baseline and debate for each benchmark example and compares results."""

    def __init__(
        self,
        client: BaseLLMClient | None = None,
        model: str = "claude-3-5-sonnet-latest",
        max_rounds: int = 3,
        results_dir: str = "results",
    ) -> None:
        if client is None:
            client = AnthropicClient(model=model)
        self._client = client
        self._orchestrator = DebateOrchestrator(
            client=client,
            max_rounds=max_rounds,
            save_results=False,
            enable_graph_analysis=False,
            results_dir=results_dir,
        )
        self.results_dir = Path(results_dir)

    def run(self, dataset_path: str) -> list[BenchmarkResult]:
        examples = _load_examples(dataset_path)
        results: list[BenchmarkResult] = []

        for ex in examples:
            print(f"  [{ex.id}] {ex.question[:60]}...")

            # Baseline
            baseline_answer, baseline_tokens = _baseline_answer(self._client, ex.question)

            # Debate
            debate_result = self._orchestrator.run(ex.question)
            debate_answer = debate_result.judge_output.final_answer
            total_tokens = (
                debate_result.total_input_tokens + debate_result.total_output_tokens
                + baseline_tokens
            )

            baseline_correct = _answers_match(baseline_answer, ex.ground_truth)
            debate_correct = _answers_match(debate_answer, ex.ground_truth)
            debate_improved = debate_correct and not baseline_correct

            results.append(BenchmarkResult(
                example_id=ex.id,
                question=ex.question,
                category=ex.category,
                ground_truth=ex.ground_truth,
                baseline_answer=baseline_answer[:400],
                debate_final_answer=debate_answer[:400],
                baseline_correct=baseline_correct,
                debate_correct=debate_correct,
                debate_improved=debate_improved,
                debate_confidence=debate_result.judge_output.confidence,
                rounds_used=debate_result.rounds_used,
                converged=debate_result.converged,
                total_tokens=total_tokens,
            ))

        return results

    def save_results(self, results: list[BenchmarkResult]) -> str:
        """Save benchmark results to CSV and JSON, return base path."""
        import pandas as pd  # deferred to avoid heavy import at module load time
        self.results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self.results_dir / f"benchmark_{timestamp}"

        # JSON
        json_path = base.with_suffix(".json")
        json_path.write_text(
            json.dumps([r.model_dump() for r in results], indent=2),
            encoding="utf-8",
        )

        # CSV
        csv_path = base.with_suffix(".csv")
        rows = [r.model_dump() for r in results]
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        return str(base)

    def calibration_report(self, results: list[BenchmarkResult]):
        from src.debate.calibration import compute_calibration
        return compute_calibration(results)

    def summary(self, results: list[BenchmarkResult]) -> dict:
        """Compute aggregate benchmark statistics."""
        total = len(results)
        if total == 0:
            return {"total_examples": 0}
        baseline_acc = sum(1 for r in results if r.baseline_correct) / total
        debate_acc = sum(1 for r in results if r.debate_correct) / total
        improved = sum(1 for r in results if r.debate_improved) / total
        avg_confidence = sum(r.debate_confidence for r in results) / total
        avg_rounds = sum(r.rounds_used for r in results) / total
        convergence_rate = sum(1 for r in results if r.converged) / total
        avg_tokens = sum(r.total_tokens for r in results) / total

        return {
            "total_examples": total,
            "baseline_accuracy": round(baseline_acc, 3),
            "debate_accuracy": round(debate_acc, 3),
            "debate_improvement_rate": round(improved, 3),
            "avg_debate_confidence": round(avg_confidence, 3),
            "avg_rounds_used": round(avg_rounds, 2),
            "convergence_rate": round(convergence_rate, 3),
            "avg_total_tokens": round(avg_tokens),
        }
