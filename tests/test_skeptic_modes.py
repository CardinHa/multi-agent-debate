"""Tests for specialized SkepticAgent modes."""
import pytest
from src.debate.agents import SkepticAgent
from src.debate.prompts import SKEPTIC_MODE_PROMPTS
from src.debate.schemas import AgentRole, DebateTranscript, DebateTurn
from src.debate.utils import MockLLMClient


def test_all_modes_have_distinct_prompts():
    """Every mode must map to a non-empty, unique prompt string."""
    prompts = list(SKEPTIC_MODE_PROMPTS.values())
    assert all(len(p) > 50 for p in prompts), "Every prompt must be non-trivial"
    assert len(set(prompts)) == len(prompts), "Every prompt must be distinct"


def test_skeptic_agent_default_mode_is_general():
    client = MockLLMClient()
    agent = SkepticAgent(client)
    assert agent.mode == "general"


def test_skeptic_agent_accepts_valid_modes():
    client = MockLLMClient()
    for mode in ("general", "factual", "logic", "evidence", "safety"):
        agent = SkepticAgent(client, mode=mode)
        assert agent.mode == mode


def test_skeptic_agent_rejects_unknown_mode():
    client = MockLLMClient()
    with pytest.raises(ValueError, match="Unknown skeptic mode"):
        SkepticAgent(client, mode="nonsense")


def test_skeptic_agent_returns_skeptic_role_for_all_modes():
    client = MockLLMClient()
    transcript = DebateTranscript(
        question="Is X true?",
        turns=[DebateTurn(round_num=1, role=AgentRole.PROPOSER,
                          content="X is true because of Y.")],
    )
    for mode in ("general", "factual", "logic", "evidence", "safety"):
        agent = SkepticAgent(client, mode=mode)
        resp = agent.challenge("Is X true?", transcript)
        assert resp.role == AgentRole.SKEPTIC, f"Mode {mode} returned wrong role"
