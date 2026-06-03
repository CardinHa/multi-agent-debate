import pytest
from src.debate.schemas import (
    AgentRole, VerdictType, ConvergenceReason,
    DebateTurn, DebateTranscript, JudgeOutput, GraphAnalysis, DebateResult
)
from src.debate.export import debate_to_markdown


def _make_result(with_graph: bool = True) -> DebateResult:
    transcript = DebateTranscript(
        question="Is the Earth flat?",
        turns=[
            DebateTurn(round_num=1, role=AgentRole.PROPOSER, content="The Earth is round.", token_count=10),
            DebateTurn(round_num=2, role=AgentRole.SKEPTIC, content="Prove it.", token_count=8),
            DebateTurn(round_num=3, role=AgentRole.JUDGE, content="Evaluating...", token_count=5),
        ],
    )
    judge = JudgeOutput(
        final_answer="The Earth is not flat.",
        verdict=VerdictType.SUPPORTED,
        confidence=0.9,
        key_reasons=["Satellite imagery", "Physics"],
        unresolved_uncertainties=["Flat-earther claims"],
        proposer_changed_position=False,
        skeptic_identified_valid_flaw=False,
        debate_improved_answer=True,
        recency_bias_check="No recency bias detected",
    )
    graph = GraphAnalysis(
        num_turns=3,
        num_claims=2,
        num_rebuttals=1,
        num_concessions=0,
        num_revisions=0,
        has_cycles=False,
        proposer_revisions_caused_by_skeptic=0,
        argument_depth=3,
        centrality_scores={"proposer": 0.5},
        edge_type_counts={"claim": 2},
    ) if with_graph else None
    return DebateResult(
        question="Is the Earth flat?",
        transcript=transcript,
        judge_output=judge,
        converged=True,
        convergence_reason=ConvergenceReason.CONCESSION,
        rounds_used=3,
        graph_analysis=graph,
        total_input_tokens=60,
        total_output_tokens=40,
    )


def test_returns_non_empty_string():
    result = _make_result()
    md = debate_to_markdown(result)
    assert isinstance(md, str) and len(md) > 0


def test_question_in_output():
    md = debate_to_markdown(_make_result())
    assert "Is the Earth flat?" in md


def test_all_turn_roles_present():
    md = debate_to_markdown(_make_result())
    assert "proposer" in md.lower() or "Proposer" in md
    assert "skeptic" in md.lower() or "Skeptic" in md


def test_judge_verdict_present():
    md = debate_to_markdown(_make_result())
    assert "supported" in md.lower() or "SUPPORTED" in md


def test_key_reasons_present():
    md = debate_to_markdown(_make_result())
    assert "Satellite imagery" in md
    assert "Physics" in md


def test_graph_metrics_present_when_set():
    md = debate_to_markdown(_make_result(with_graph=True))
    assert "| Nodes |" in md or "| Turns |" in md


def test_no_graph_section_when_absent():
    md = debate_to_markdown(_make_result(with_graph=False))
    assert "## Graph Metrics" not in md


def test_token_usage_in_output():
    md = debate_to_markdown(_make_result())
    assert "*Tokens — input: 60 | output: 40*" in md
