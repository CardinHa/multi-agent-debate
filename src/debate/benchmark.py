"""BenchmarkRunner — compares single-agent baseline vs debate system."""
from __future__ import annotations
import json
import re
from pathlib import Path
from datetime import datetime
from src.debate.schemas import BenchmarkExample, BenchmarkResult, AgentRole
from src.debate.orchestrator import DebateOrchestrator
from src.debate.utils import BaseLLMClient, AnthropicClient, DEFAULT_MODEL
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


_YES_WORDS = {"yes", "yeah", "yep", "correct", "true", "indeed", "affirmative"}
_NO_WORDS = {"no", "nope", "incorrect", "false", "negative"}


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _leading_polarity(text: str) -> str | None:
    """Return 'yes', 'no', or None based on the leading token of normalized text."""
    words = _normalize(text).split()
    if not words:
        return None
    first = words[0]
    if first in _YES_WORDS:
        return "yes"
    if first in _NO_WORDS:
        return "no"
    return None


def _token_overlap_ratio(answer: str, ground_truth: str) -> float:
    a_tokens = set(_normalize(answer).split())
    g_tokens = set(_normalize(ground_truth).split())
    return len(a_tokens & g_tokens) / max(len(g_tokens), 1)


def _answers_match(answer: str, ground_truth: str) -> bool:
    """
    Polarity-aware correctness check.

    Ground truths in this dataset conventionally open with an explicit
    "Yes."/"No." verdict. Raw keyword overlap alone has a length bias: a
    verbose *wrong* answer that reuses the question's topic vocabulary can
    out-overlap a terse *correct* one-word answer. To fix this, extract an
    explicit yes/no polarity from the leading token of both the answer and
    the ground truth when present. If both state a polarity, agreement
    between them is the primary signal — a stated "Yes" cannot match a
    "No." truth no matter how much vocabulary it shares. Token overlap is
    used as a secondary signal only when at least one side has no explicit
    leading polarity to compare.
    """
    answer_polarity = _leading_polarity(answer)
    truth_polarity = _leading_polarity(ground_truth)

    if answer_polarity is not None and truth_polarity is not None:
        return answer_polarity == truth_polarity

    return _token_overlap_ratio(answer, ground_truth) > 0.35


class BenchmarkRunner:
    """Runs baseline and debate for each benchmark example and compares results."""

    def __init__(
        self,
        client: BaseLLMClient | None = None,
        model: str = DEFAULT_MODEL,
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
        """Run the benchmark, saving each example's result incrementally as it completes.

        A single example's failure (LLM error, malformed response, etc.) is recorded
        as a BenchmarkResult with `error` set and the run continues with the next
        example rather than aborting the whole benchmark.
        """
        examples = _load_examples(dataset_path)
        results: list[BenchmarkResult] = []

        self.results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.last_run_timestamp = timestamp
        incremental_path = self.results_dir / f"benchmark_{timestamp}_partial.jsonl"

        with incremental_path.open("a", encoding="utf-8") as incremental_file:
            for ex in examples:
                print(f"  [{ex.id}] {ex.question[:60]}...")
                try:
                    result = self._run_example(ex)
                except Exception as exc:  # noqa: BLE001 - benchmark must not abort on one bad example
                    print(f"  [{ex.id}] FAILED: {exc!r}")
                    result = BenchmarkResult(
                        example_id=ex.id,
                        question=ex.question,
                        category=ex.category,
                        ground_truth=ex.ground_truth,
                        error=f"{type(exc).__name__}: {exc}",
                    )

                results.append(result)
                incremental_file.write(result.model_dump_json() + "\n")
                incremental_file.flush()

        return results

    def _run_example(self, ex: BenchmarkExample) -> BenchmarkResult:
        """Run baseline + debate for a single example. May raise on LLM/parse errors."""
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
        debate_regressed = baseline_correct and not debate_correct

        return BenchmarkResult(
            example_id=ex.id,
            question=ex.question,
            category=ex.category,
            ground_truth=ex.ground_truth,
            baseline_answer=baseline_answer[:400],
            debate_final_answer=debate_answer[:400],
            baseline_correct=baseline_correct,
            debate_correct=debate_correct,
            debate_improved=debate_improved,
            debate_regressed=debate_regressed,
            debate_confidence=debate_result.judge_output.confidence,
            rounds_used=debate_result.rounds_used,
            converged=debate_result.converged,
            total_tokens=total_tokens,
            judge_parse_failed=debate_result.judge_output.parse_failed,
        )

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
        """Compute aggregate benchmark statistics.

        Examples that failed to run (``result.error`` set) are counted explicitly
        via `failed_examples` and excluded from accuracy/confidence/token
        aggregates so a handful of errored examples don't silently skew them.
        """
        total = len(results)
        if total == 0:
            return {"total_examples": 0, "failed_examples": 0}

        failed = [r for r in results if r.error]
        ok = [r for r in results if not r.error]
        n_ok = len(ok)

        if n_ok == 0:
            return {"total_examples": total, "failed_examples": len(failed)}

        baseline_acc = sum(1 for r in ok if r.baseline_correct) / n_ok
        debate_acc = sum(1 for r in ok if r.debate_correct) / n_ok
        improved = sum(1 for r in ok if r.debate_improved) / n_ok
        regressed = sum(1 for r in ok if r.debate_regressed) / n_ok
        avg_confidence = sum(r.debate_confidence for r in ok) / n_ok
        avg_rounds = sum(r.rounds_used for r in ok) / n_ok
        convergence_rate = sum(1 for r in ok if r.converged) / n_ok
        avg_tokens = sum(r.total_tokens for r in ok) / n_ok

        return {
            "total_examples": total,
            "failed_examples": len(failed),
            "judge_parse_failures": sum(1 for r in ok if r.judge_parse_failed),
            "baseline_accuracy": round(baseline_acc, 3),
            "debate_accuracy": round(debate_acc, 3),
            "debate_improvement_rate": round(improved, 3),
            "debate_regression_rate": round(regressed, 3),
            "avg_debate_confidence": round(avg_confidence, 3),
            "avg_rounds_used": round(avg_rounds, 2),
            "convergence_rate": round(convergence_rate, 3),
            "avg_total_tokens": round(avg_tokens),
        }
