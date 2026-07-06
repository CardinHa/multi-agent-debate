import pytest
from multi_agent_debate.debate.utils import MockLLMClient
from multi_agent_debate.debate.orchestrator import DebateOrchestrator
from multi_agent_debate.debate.schemas import AgentRole

def _run_with_constitutional(question="Is the sky blue?"):
    client = MockLLMClient()
    orch = DebateOrchestrator(
        client=client, save_results=False, enable_graph_analysis=False,
        max_rounds=1, enable_constitutional=True,
    )
    return orch.run(question)

def test_constitutional_review_present_when_enabled():
    result = _run_with_constitutional()
    assert result.constitutional_review is not None

def test_constitutional_review_absent_when_disabled():
    client = MockLLMClient()
    orch = DebateOrchestrator(
        client=client, save_results=False, enable_graph_analysis=False,
        enable_constitutional=False,
    )
    result = orch.run("Test?")
    assert result.constitutional_review is None

def test_constitutional_review_has_overall_safe_field():
    result = _run_with_constitutional()
    assert isinstance(result.constitutional_review.overall_safe, bool)

def test_constitutional_review_has_principles_checked():
    result = _run_with_constitutional()
    assert len(result.constitutional_review.principles_checked) > 0

def test_constitutional_review_violations_is_list():
    result = _run_with_constitutional()
    assert isinstance(result.constitutional_review.violations, list)

def test_constitutional_review_warnings_is_list():
    result = _run_with_constitutional()
    assert isinstance(result.constitutional_review.warnings, list)
