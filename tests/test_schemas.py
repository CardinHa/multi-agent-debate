"""Tests for Pydantic schema validation."""
import pytest
from src.debate.schemas import (
    AgentRole, ConvergenceReason, DebateTurn, DebateTranscript,
    AgentResponse, JudgeOutput, GraphAnalysis, DebateResult,
    BenchmarkExample, BenchmarkResult,
)


def test_debate_turn_serializes():
    turn = DebateTurn(round_num=1, role=AgentRole.PROPOSER, content="Test argument")
    d = turn.model_dump()
    assert d["role"] == "proposer"
    assert d["round_num"] == 1


def test_judge_output_confidence_bounds():
    with pytest.raises(Exception):
        JudgeOutput(
            final_answer="x", verdict="supported", confidence=1.5,
            key_reasons=[], unresolved_uncertainties=[],
            proposer_changed_position=False, skeptic_identified_valid_flaw=False,
            debate_improved_answer=False, recency_bias_check="n/a",
        )


def test_debate_result_json_roundtrip():
    transcript = DebateTranscript(
        question="Is X true?",
        turns=[DebateTurn(round_num=1, role=AgentRole.PROPOSER, content="Yes")]
    )
    judge = JudgeOutput(
        final_answer="Yes", verdict="supported", confidence=0.8,
        key_reasons=["reason1"], unresolved_uncertainties=[],
        proposer_changed_position=False, skeptic_identified_valid_flaw=False,
        debate_improved_answer=True, recency_bias_check="Evaluated all turns equally.",
    )
    result = DebateResult(
        question="Is X true?", transcript=transcript, judge_output=judge,
        converged=False, convergence_reason=None, rounds_used=1,
        graph_analysis=None,
    )
    json_str = result.model_dump_json()
    assert "Is X true?" in json_str


def test_benchmark_example_fields():
    ex = BenchmarkExample(
        id="claim_001", question="Q?", ground_truth="A", category="factual"
    )
    assert ex.category == "factual"
