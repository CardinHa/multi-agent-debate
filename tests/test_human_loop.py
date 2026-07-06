"""Tests for Feature C: Human-in-the-Loop Mode."""
import pytest
from multi_agent_debate.debate.utils import MockLLMClient
from multi_agent_debate.debate.orchestrator import DebateOrchestrator
from multi_agent_debate.debate.schemas import AgentRole


def _human_fn(prompt: str) -> str:
    return "This is my human response to the debate."


def test_human_proposer_initial_argument():
    client = MockLLMClient()
    orch = DebateOrchestrator(
        client=client, save_results=False, enable_graph_analysis=False,
        max_rounds=1, human_role="proposer", human_input_fn=_human_fn,
    )
    result = orch.run("Test question?")
    proposer_turns = [t for t in result.transcript.turns if t.role == AgentRole.PROPOSER]
    assert any("human response" in t.content for t in proposer_turns)


def test_human_skeptic_turn():
    client = MockLLMClient()
    orch = DebateOrchestrator(
        client=client, save_results=False, enable_graph_analysis=False,
        max_rounds=1, human_role="skeptic", human_input_fn=_human_fn,
    )
    result = orch.run("Test question?")
    skeptic_turns = [t for t in result.transcript.turns if t.role == AgentRole.SKEPTIC]
    assert any("human response" in t.content for t in skeptic_turns)


def test_human_role_stored_on_result():
    client = MockLLMClient()
    orch = DebateOrchestrator(
        client=client, save_results=False, enable_graph_analysis=False,
        max_rounds=1, human_role="proposer", human_input_fn=_human_fn,
    )
    result = orch.run("Test question?")
    assert result.human_role == "proposer"


def test_no_human_role_ai_runs_normally():
    client = MockLLMClient()
    orch = DebateOrchestrator(
        client=client, save_results=False, enable_graph_analysis=False,
    )
    result = orch.run("Test question?")
    assert result.human_role is None


def test_human_role_none_means_ai_proposer():
    client = MockLLMClient()
    orch = DebateOrchestrator(
        client=client, save_results=False, enable_graph_analysis=False,
        max_rounds=1,
    )
    result = orch.run("Test question?")
    proposer_turns = [t for t in result.transcript.turns if t.role == AgentRole.PROPOSER]
    assert len(proposer_turns) > 0
    assert all("I propose" in t.content for t in proposer_turns)
