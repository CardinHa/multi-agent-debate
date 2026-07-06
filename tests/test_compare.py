"""Tests for A/B debate comparison mode."""
import pytest
from multi_agent_debate.debate.utils import MockLLMClient
from multi_agent_debate.debate.compare import run_comparison, comparison_to_markdown
from multi_agent_debate.debate.schemas import DebateComparison


def _client():
    return MockLLMClient()


def test_run_comparison_returns_debate_comparison():
    result = run_comparison(
        question="Is water wet?",
        client=_client(),
        config_a={"skeptic_mode": "general"},
        config_b={"skeptic_mode": "logic"},
        max_rounds=1,
    )
    assert isinstance(result, DebateComparison)


def test_comparison_has_both_results():
    comp = run_comparison("Test?", _client(), {"skeptic_mode": "general"}, {"skeptic_mode": "factual"}, max_rounds=1)
    assert comp.result_a is not None
    assert comp.result_b is not None


def test_comparison_verdict_match_is_bool():
    comp = run_comparison("Test?", _client(), {"skeptic_mode": "general"}, {"skeptic_mode": "general"}, max_rounds=1)
    assert isinstance(comp.verdict_match, bool)


def test_comparison_confidence_delta_is_float():
    comp = run_comparison("Test?", _client(), {"skeptic_mode": "general"}, {"skeptic_mode": "logic"}, max_rounds=1)
    assert isinstance(comp.confidence_delta, float)


def test_comparison_to_markdown_contains_question():
    comp = run_comparison("Is AI conscious?", _client(), {"skeptic_mode": "general"}, {"skeptic_mode": "safety"}, max_rounds=1)
    md = comparison_to_markdown(comp)
    assert "Is AI conscious?" in md


def test_comparison_to_markdown_contains_verdicts():
    comp = run_comparison("Test?", _client(), {"skeptic_mode": "general"}, {"skeptic_mode": "general"}, max_rounds=1)
    md = comparison_to_markdown(comp)
    assert "Config A" in md and "Config B" in md
