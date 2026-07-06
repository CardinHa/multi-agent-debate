"""BenchmarkRunner — compares single-agent baseline vs debate system."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from multi_agent_debate.debate.schemas import BenchmarkExample, BenchmarkResult, AgentRole, GraderType
from multi_agent_debate.debate.orchestrator import DebateOrchestrator
from multi_agent_debate.debate.utils import BaseLLMClient, AnthropicClient, DEFAULT_MODEL
from multi_agent_debate.debate.prompts import PROPOSER_SYSTEM_PROMPT
from multi_agent_debate.debate.grading import (
    answers_match,
    grade_with_llm,
    GRADER_CALLS_PER_EXAMPLE,
    GRADER_MAX_TOKENS,
    ROUGH_TOKENS_PER_GRADER_CALL,
)

# Re-exported for backward compatibility — this constant is documented as
# living in grading.py, but callers that only import from benchmark.py (e.g.
# the CLI's cost guardrail) can reach it here too.
__all__ = [
    "BenchmarkRunner",
    "count_examples",
    "GRADER_CALLS_PER_EXAMPLE",
    "ROUGH_TOKENS_PER_GRADER_CALL",
]


def _load_examples(path: str) -> list[BenchmarkExample]:
    examples = []
    for line in Path(path).read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if line:
            data = json.loads(line)
            examples.append(BenchmarkExample(**data))
    return examples


def count_examples(path: str) -> int:
    """Count examples in a JSONL dataset without full Pydantic validation.

    Used by the CLI's upfront cost estimate for --grader llm, where we only
    need a cheap count, not validated BenchmarkExample objects.
    """
    return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


def _baseline_answer(client: BaseLLMClient, question: str) -> tuple[str, int]:
    """Get a single-agent answer using the Proposer system prompt."""
    text, inp, out = client.call(
        system=PROPOSER_SYSTEM_PROMPT,
        user=f"Question: {question}\n\nProvide your best answer.",
    )
    return text, inp + out


# The heuristic grader itself now lives in grading.py (it's reused by
# grade_with_llm's fallback path). Kept importable here under its original
# name for backward compatibility with existing callers/tests.
_answers_match = answers_match


class BenchmarkRunner:
    """Runs baseline and debate for each benchmark example and compares results."""

    def __init__(
        self,
        client: BaseLLMClient | None = None,
        model: str = DEFAULT_MODEL,
        max_rounds: int = 3,
        results_dir: str = "results",
        grader: str = "heuristic",
        grader_client: BaseLLMClient | None = None,
    ) -> None:
        if grader not in ("heuristic", "llm"):
            raise ValueError(f"grader must be 'heuristic' or 'llm', got {grader!r}")
        self.grader = grader

        client_was_provided = client is not None
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

        # The LLM grader wants deterministic, cheap calls (temperature 0,
        # small max_tokens) — distinct from the debate client's settings. If
        # the caller built their own client (a MockLLMClient, a test double,
        # or a custom BaseLLMClient), reuse it for grading too rather than
        # guessing how to reconfigure it. Only when we constructed the
        # default AnthropicClient ourselves do we also construct a dedicated
        # low-temperature grading client.
        if grader_client is not None:
            self._grader_client = grader_client
        elif not client_was_provided:
            self._grader_client = AnthropicClient(
                model=model, temperature=0.0, max_tokens=GRADER_MAX_TOKENS
            )
        else:
            self._grader_client = client

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

        grader_tokens = 0
        baseline_heuristic_match: bool | None = None
        debate_heuristic_match: bool | None = None

        if self.grader == "llm":
            # Compute the heuristic verdict alongside for free — it costs no
            # extra API call and lets the summary report an agreement rate.
            baseline_heuristic_match = answers_match(baseline_answer, ex.ground_truth)
            debate_heuristic_match = answers_match(debate_answer, ex.ground_truth)

            baseline_grade = grade_with_llm(
                self._grader_client, ex.question, baseline_answer, ex.ground_truth
            )
            debate_grade = grade_with_llm(
                self._grader_client, ex.question, debate_answer, ex.ground_truth
            )
            grader_tokens = (
                baseline_grade.input_tokens + baseline_grade.output_tokens
                + debate_grade.input_tokens + debate_grade.output_tokens
            )

            baseline_correct = baseline_grade.match
            debate_correct = debate_grade.match
            baseline_grader = baseline_grade.grader
            debate_grader = debate_grade.grader
        else:
            baseline_correct = answers_match(baseline_answer, ex.ground_truth)
            debate_correct = answers_match(debate_answer, ex.ground_truth)
            baseline_grader = GraderType.HEURISTIC
            debate_grader = GraderType.HEURISTIC

        total_tokens = (
            debate_result.total_input_tokens + debate_result.total_output_tokens
            + baseline_tokens + grader_tokens
        )

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
            baseline_grader=baseline_grader,
            debate_grader=debate_grader,
            baseline_heuristic_match=baseline_heuristic_match,
            debate_heuristic_match=debate_heuristic_match,
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
        from multi_agent_debate.debate.calibration import compute_calibration
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

        result = {
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

        # LLM-grading metadata is only added when at least one example was
        # actually graded (or attempted) by the LLM grader — a pure
        # grader="heuristic" run's summary is byte-for-byte identical to the
        # pre-LLM-grader schema.
        llm_graded = [
            r for r in ok
            if r.baseline_grader != GraderType.HEURISTIC
            or r.debate_grader != GraderType.HEURISTIC
        ]
        if llm_graded:
            result["llm_graded_examples"] = len(llm_graded)
            result["grader_fallback_count"] = sum(
                1 for r in ok
                if r.baseline_grader == GraderType.HEURISTIC_FALLBACK
                or r.debate_grader == GraderType.HEURISTIC_FALLBACK
            )
            # Agreement between the heuristic grader and the LLM grader,
            # computed per verdict (baseline and debate each contribute one
            # comparison) over every verdict where both were available.
            agreements = []
            for r in ok:
                if r.baseline_heuristic_match is not None:
                    agreements.append(r.baseline_correct == r.baseline_heuristic_match)
                if r.debate_heuristic_match is not None:
                    agreements.append(r.debate_correct == r.debate_heuristic_match)
            if agreements:
                result["heuristic_llm_agreement_rate"] = round(
                    sum(agreements) / len(agreements), 3
                )

        return result
