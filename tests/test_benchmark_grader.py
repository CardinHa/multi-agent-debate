"""Tests for BenchmarkRunner's opt-in grader="llm" mode (offline, MockLLMClient only)."""
from __future__ import annotations

import json

import pytest

from multi_agent_debate.debate.benchmark import BenchmarkRunner
from multi_agent_debate.debate.schemas import GraderType
from multi_agent_debate.debate.utils import MockLLMClient


def _write_dataset(tmp_path, examples: list[dict]) -> str:
    path = tmp_path / "claims.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in examples), encoding="utf-8")
    return str(path)


def test_default_grader_is_heuristic():
    runner = BenchmarkRunner(client=MockLLMClient(), results_dir="results")
    assert runner.grader == "heuristic"


def test_invalid_grader_value_rejected():
    with pytest.raises(ValueError):
        BenchmarkRunner(client=MockLLMClient(), grader="bogus")


def test_default_path_unchanged_no_grader_metadata_in_summary(tmp_path):
    """grader='heuristic' (the default) must not add any new summary keys or
    populate the LLM-grading-only result fields — zero behavior change."""
    dataset = _write_dataset(tmp_path, [
        {"id": "a", "question": "Is water wet?", "ground_truth": "Yes.", "category": "factual"},
    ])
    runner = BenchmarkRunner(client=MockLLMClient(), results_dir=str(tmp_path / "results"))
    results = runner.run(dataset)
    result = results[0]

    assert result.baseline_grader == GraderType.HEURISTIC
    assert result.debate_grader == GraderType.HEURISTIC
    assert result.baseline_heuristic_match is None
    assert result.debate_heuristic_match is None

    summary = runner.summary(results)
    assert "llm_graded_examples" not in summary
    assert "grader_fallback_count" not in summary
    assert "heuristic_llm_agreement_rate" not in summary


def test_llm_grader_mode_records_metadata_and_agreement_rate(tmp_path):
    dataset = _write_dataset(tmp_path, [
        {"id": "a", "question": "Is water wet?", "ground_truth": "Yes.", "category": "factual"},
        {"id": "b", "question": "Is the sky green?", "ground_truth": "No.", "category": "factual"},
    ])
    runner = BenchmarkRunner(
        client=MockLLMClient(), grader="llm", results_dir=str(tmp_path / "results"),
    )
    results = runner.run(dataset)

    assert len(results) == 2
    for result in results:
        assert result.error is None
        assert result.baseline_grader in (GraderType.LLM, GraderType.HEURISTIC_FALLBACK)
        assert result.debate_grader in (GraderType.LLM, GraderType.HEURISTIC_FALLBACK)
        assert result.baseline_heuristic_match is not None
        assert result.debate_heuristic_match is not None
        # Grader token accounting must be folded into total_tokens.
        assert result.total_tokens > 0

    summary = runner.summary(results)
    assert summary["llm_graded_examples"] == 2
    assert "grader_fallback_count" in summary
    assert "heuristic_llm_agreement_rate" in summary
    assert 0.0 <= summary["heuristic_llm_agreement_rate"] <= 1.0


def test_llm_grader_uses_provided_client_for_grading_too(tmp_path):
    """When a client is explicitly supplied (e.g. MockLLMClient for --mock),
    the grader must reuse it rather than requiring an ANTHROPIC_API_KEY."""
    dataset = _write_dataset(tmp_path, [
        {"id": "a", "question": "Is water wet?", "ground_truth": "Yes.", "category": "factual"},
    ])
    mock_client = MockLLMClient()
    runner = BenchmarkRunner(client=mock_client, grader="llm", results_dir=str(tmp_path / "results"))
    assert runner._grader_client is mock_client
    # Should not raise (no real API key needed).
    results = runner.run(dataset)
    assert results[0].error is None
