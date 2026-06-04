"""Calibration scoring layer — per-category accuracy and confidence calibration bins."""
from __future__ import annotations
from collections import defaultdict
from src.debate.schemas import BenchmarkResult, CalibrationReport, CategoryStats, CalibrationBin


def compute_calibration(
    results: list[BenchmarkResult],
    num_bins: int = 5,
) -> CalibrationReport:
    """Compute per-category accuracy and confidence calibration from benchmark results.

    Args:
        results: List of BenchmarkResult objects from a benchmark run.
        num_bins: Number of equal-width confidence bins to divide [0.0, 1.0] into.

    Returns:
        CalibrationReport with overall stats, per-category breakdown, and calibration bins.
    """
    total = len(results)

    # --- Overall stats ---
    overall_baseline_acc = (
        sum(1 for r in results if r.baseline_correct) / total if total else 0.0
    )
    overall_debate_acc = (
        sum(1 for r in results if r.debate_correct) / total if total else 0.0
    )
    overall_improvement_rate = (
        sum(1 for r in results if r.debate_improved) / total if total else 0.0
    )

    # --- Per-category ---
    category_buckets: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        category_buckets[r.category].append(r)

    per_category: list[CategoryStats] = []
    for cat, cat_results in sorted(category_buckets.items()):
        n = len(cat_results)
        per_category.append(
            CategoryStats(
                category=cat,
                total=n,
                baseline_accuracy=sum(1 for r in cat_results if r.baseline_correct) / n,
                debate_accuracy=sum(1 for r in cat_results if r.debate_correct) / n,
                improvement_rate=sum(1 for r in cat_results if r.debate_improved) / n,
                avg_confidence=sum(r.debate_confidence for r in cat_results) / n,
            )
        )

    # --- Calibration bins ---
    bin_width = 1.0 / num_bins
    calibration_bins: list[CalibrationBin] = []
    for i in range(num_bins):
        low = round(i * bin_width, 10)
        # Last bin includes upper boundary (1.0) to avoid off-by-one on confidence == 1.0
        high = round((i + 1) * bin_width, 10)
        if i == num_bins - 1:
            in_bin = [r for r in results if low <= r.debate_confidence <= high]
        else:
            in_bin = [r for r in results if low <= r.debate_confidence < high]
        count = len(in_bin)
        actual_accuracy = (
            sum(1 for r in in_bin if r.debate_correct) / count if count else 0.0
        )
        calibration_bins.append(
            CalibrationBin(
                confidence_low=low,
                confidence_high=high,
                count=count,
                actual_accuracy=actual_accuracy,
            )
        )

    return CalibrationReport(
        total_examples=total,
        overall_baseline_accuracy=overall_baseline_acc,
        overall_debate_accuracy=overall_debate_acc,
        overall_improvement_rate=overall_improvement_rate,
        per_category=per_category,
        calibration_bins=calibration_bins,
    )
