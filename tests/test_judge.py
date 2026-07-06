"""Tests for JudgeAgent JSON extraction and parse-failure tracking."""
from __future__ import annotations

import json

import pytest

from src.debate.judge import JudgeAgent, _extract_json
from src.debate.schemas import DebateTranscript, DebateTurn, AgentRole
from src.debate.utils import BaseLLMClient


_VALID_JUDGE_JSON = {
    "final_answer": "Yes.",
    "verdict": "supported",
    "confidence": 0.8,
    "key_reasons": ["Reason one."],
    "unresolved_uncertainties": [],
    "proposer_changed_position": False,
    "skeptic_identified_valid_flaw": False,
    "debate_improved_answer": False,
    "recency_bias_check": "Checked.",
}


def _transcript() -> DebateTranscript:
    return DebateTranscript(
        question="Is water wet?",
        turns=[DebateTurn(round_num=0, role=AgentRole.PROPOSER, content="Yes, water is wet.")],
    )


class _ScriptedClient(BaseLLMClient):
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    def call(self, system: str, user: str) -> tuple[str, int, int]:
        return self._response_text, 10, 10


def test_extract_json_direct_parse():
    assert _extract_json(json.dumps(_VALID_JUDGE_JSON)) == _VALID_JUDGE_JSON


def test_extract_json_strips_markdown_fences():
    wrapped = f"```json\n{json.dumps(_VALID_JUDGE_JSON)}\n```"
    assert _extract_json(wrapped) == _VALID_JUDGE_JSON


def test_extract_json_extracts_balanced_brace_region_from_prose():
    wrapped = f"Sure, here is the verdict:\n{json.dumps(_VALID_JUDGE_JSON)}\nHope that helps!"
    assert _extract_json(wrapped) == _VALID_JUDGE_JSON


def test_extract_json_raises_when_no_braces_present():
    with pytest.raises(json.JSONDecodeError):
        _extract_json("I refuse to answer in JSON.")


def test_judge_agent_sets_parse_failed_on_unparseable_output():
    agent = JudgeAgent(_ScriptedClient("This is not JSON at all, sorry."))
    judge_output, _, _ = agent.evaluate(_transcript())
    assert judge_output.parse_failed is True
    assert judge_output.verdict.value == "uncertain"


def test_judge_agent_parse_failed_false_on_valid_output():
    agent = JudgeAgent(_ScriptedClient(json.dumps(_VALID_JUDGE_JSON)))
    judge_output, _, _ = agent.evaluate(_transcript())
    assert judge_output.parse_failed is False


def test_judge_agent_recovers_via_brace_extraction_fallback():
    prose_wrapped = f"Here's my verdict: {json.dumps(_VALID_JUDGE_JSON)} Thanks!"
    agent = JudgeAgent(_ScriptedClient(prose_wrapped))
    judge_output, _, _ = agent.evaluate(_transcript())
    assert judge_output.parse_failed is False
    assert judge_output.verdict.value == "supported"
