"""Tests for multi-skeptic panel debate mode (Feature B)."""
import pytest
from multi_agent_debate.debate.utils import MockLLMClient
from multi_agent_debate.debate.orchestrator import DebateOrchestrator


def test_panel_mode_flag_is_set():
    client = MockLLMClient()
    orch = DebateOrchestrator(client=client, save_results=False, enable_graph_analysis=False,
                               skeptic_modes=["logic", "evidence"])
    result = orch.run("Test question?")
    assert result.panel_mode is True


def test_single_skeptic_panel_mode_false():
    client = MockLLMClient()
    orch = DebateOrchestrator(client=client, save_results=False, enable_graph_analysis=False,
                               skeptic_modes=["general"])
    result = orch.run("Test question?")
    assert result.panel_mode is False


def test_panel_produces_multiple_skeptic_turns_per_round():
    client = MockLLMClient()
    orch = DebateOrchestrator(client=client, save_results=False, enable_graph_analysis=False,
                               max_rounds=1, skeptic_modes=["logic", "evidence"])
    result = orch.run("Test question?")
    from multi_agent_debate.debate.schemas import AgentRole
    skeptic_turns = [t for t in result.transcript.turns if t.role == AgentRole.SKEPTIC]
    assert len(skeptic_turns) >= 2  # at least one turn per skeptic


def test_panel_result_has_judge_output():
    client = MockLLMClient()
    orch = DebateOrchestrator(client=client, save_results=False, enable_graph_analysis=False,
                               skeptic_modes=["factual", "safety"])
    result = orch.run("Test question?")
    assert result.judge_output is not None


def test_panel_backward_compat_single_skeptic():
    """Existing single-skeptic code path still works unchanged."""
    client = MockLLMClient()
    orch = DebateOrchestrator(client=client, save_results=False, enable_graph_analysis=False)
    result = orch.run("Test question?")
    assert result.panel_mode is False
    assert result.judge_output is not None
