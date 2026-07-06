import pytest
from multi_agent_debate.debate.schemas import (
    AgentRole, VerdictType, ConvergenceReason,
    DebateTurn, DebateTranscript, JudgeOutput, GraphAnalysis, DebateResult
)


def _make_result(with_graph: bool = True) -> DebateResult:
    transcript = DebateTranscript(
        question="Is AI safe?",
        turns=[
            DebateTurn(round_num=1, role=AgentRole.PROPOSER, content="AI is safe.", token_count=10),
            DebateTurn(round_num=1, role=AgentRole.SKEPTIC, content="Prove it.", token_count=8),
        ],
    )
    judge = JudgeOutput(
        final_answer="AI can be safe with proper oversight.",
        verdict=VerdictType.UNCERTAIN,
        confidence=0.6,
        key_reasons=["Alignment research ongoing"],
        unresolved_uncertainties=["AGI risk"],
        proposer_changed_position=False,
        skeptic_identified_valid_flaw=True,
        debate_improved_answer=True,
        recency_bias_check="No recency bias.",
    )
    graph = GraphAnalysis(num_turns=2, num_claims=1, num_rebuttals=1, num_concessions=0,
                          num_revisions=0, has_cycles=False,
                          proposer_revisions_caused_by_skeptic=0,
                          argument_depth=2, centrality_scores={}, edge_type_counts={}) if with_graph else None
    return DebateResult(
        question="Is AI safe?",
        transcript=transcript,
        judge_output=judge,
        converged=False,
        convergence_reason=None,
        rounds_used=1,
        graph_analysis=graph,
        total_input_tokens=60,
        total_output_tokens=40,
        panel_mode=False,
        human_role=None,
    )


def test_returns_string():
    from multi_agent_debate.debate.html_export import debate_to_html
    html = debate_to_html(_make_result())
    assert isinstance(html, str) and len(html) > 0


def test_is_valid_html_structure():
    from multi_agent_debate.debate.html_export import debate_to_html
    html = debate_to_html(_make_result())
    assert "<!DOCTYPE html>" in html
    assert "<html" in html
    assert "</html>" in html


def test_question_in_title_and_body():
    from multi_agent_debate.debate.html_export import debate_to_html
    html = debate_to_html(_make_result())
    assert "Is AI safe?" in html


def test_role_classes_present():
    from multi_agent_debate.debate.html_export import debate_to_html
    html = debate_to_html(_make_result())
    assert "turn-proposer" in html
    assert "turn-skeptic" in html


def test_verdict_badge_present():
    from multi_agent_debate.debate.html_export import debate_to_html
    html = debate_to_html(_make_result())
    assert "verdict-uncertain" in html or "uncertain" in html.lower()


def test_graph_table_when_present():
    from multi_agent_debate.debate.html_export import debate_to_html
    html = debate_to_html(_make_result(with_graph=True))
    assert "num_turns" in html or "Num Turns" in html or "<table" in html


def test_no_graph_table_when_absent():
    from multi_agent_debate.debate.html_export import debate_to_html
    html = debate_to_html(_make_result(with_graph=False))
    assert "Graph Metrics" not in html


def test_html_escaped_content():
    from multi_agent_debate.debate.html_export import debate_to_html
    from multi_agent_debate.debate.schemas import DebateTurn, DebateTranscript, JudgeOutput, VerdictType, DebateResult
    transcript = DebateTranscript(
        question="Is <b>AI</b> safe?",
        turns=[DebateTurn(round_num=1, role=AgentRole.PROPOSER, content="<script>alert(1)</script>", token_count=5)],
    )
    judge = JudgeOutput(
        final_answer="safe", verdict=VerdictType.SUPPORTED, confidence=0.9,
        key_reasons=[], unresolved_uncertainties=[],
        proposer_changed_position=False, skeptic_identified_valid_flaw=False,
        debate_improved_answer=False, recency_bias_check="none",
    )
    result = DebateResult(
        question="Is <b>AI</b> safe?", transcript=transcript, judge_output=judge,
        converged=False, convergence_reason=None, rounds_used=1,
        graph_analysis=None, total_input_tokens=0, total_output_tokens=0,
        panel_mode=False, human_role=None,
    )
    html = debate_to_html(result)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
