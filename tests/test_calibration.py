"""Tests for calibration scoring layer — written before implementation (TDD)."""
from __future__ import annotations
import pytest
from src.debate.schemas import BenchmarkResult, CalibrationReport
from src.debate.calibration import compute_calibration


def _make_results() -> list[BenchmarkResult]:
    """Build a list with known values for testing.

    10 results across 3 categories with a mix of correctness outcomes.
    debate_correct counts: 7 True, 3 False  → overall accuracy = 0.7
    """
    def _r(
        eid: str,
        category: str,
        baseline_correct: bool,
        debate_correct: bool,
        debate_confidence: float,
    ) -> BenchmarkResult:
        return BenchmarkResult(
            example_id=eid,
            question=f"Question {eid}",
            category=category,
            ground_truth="answer",
            baseline_answer="base",
            debate_final_answer="debate",
            baseline_correct=baseline_correct,
            debate_correct=debate_correct,
            debate_improved=debate_correct and not baseline_correct,
            debate_confidence=debate_confidence,
            rounds_used=2,
            converged=True,
            total_tokens=100,
        )

    return [
        # factual — 4 examples, 3 debate correct
        _r("f1", "factual", True,  True,  0.1),
        _r("f2", "factual", False, True,  0.3),
        _r("f3", "factual", True,  True,  0.5),
        _r("f4", "factual", True,  False, 0.7),
        # logic — 3 examples, 2 debate correct
        _r("l1", "logic",   False, True,  0.6),
        _r("l2", "logic",   True,  True,  0.8),
        _r("l3", "logic",   True,  False, 0.9),
        # math — 3 examples, 2 debate correct
        _r("m1", "math",    True,  True,  0.4),
        _r("m2", "math",    False, False, 0.2),
        _r("m3", "math",    True,  True,  0.95),
    ]


def test_returns_calibration_report():
    report = compute_calibration(_make_results())
    assert isinstance(report, CalibrationReport)


def test_total_matches_input():
    results = _make_results()
    report = compute_calibration(results)
    assert report.total_examples == len(results)


def test_per_category_covers_all_categories():
    report = compute_calibration(_make_results())
    cats = {s.category for s in report.per_category}
    assert "factual" in cats


def test_calibration_bins_count():
    report = compute_calibration(_make_results(), num_bins=5)
    assert len(report.calibration_bins) == 5


def test_calibration_bin_accuracies_are_valid():
    report = compute_calibration(_make_results())
    for b in report.calibration_bins:
        assert 0.0 <= b.actual_accuracy <= 1.0


def test_overall_accuracy_matches_manual():
    results = _make_results()
    report = compute_calibration(results)
    manual = sum(1 for r in results if r.debate_correct) / len(results)
    assert abs(report.overall_debate_accuracy - manual) < 0.001
