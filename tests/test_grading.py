"""Tests for grade_with_llm — the opt-in LLM benchmark grader."""
from __future__ import annotations

import json

from multi_agent_debate.debate.grading import answers_match, extract_json, grade_with_llm
from multi_agent_debate.debate.schemas import GraderType
from multi_agent_debate.debate.utils import BaseLLMClient


class _ScriptedClient(BaseLLMClient):
    """Returns a fixed response regardless of input, with fixed token counts."""

    def __init__(self, response_text: str, input_tokens: int = 42, output_tokens: int = 17) -> None:
        self._response_text = response_text
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    def call(self, system: str, user: str) -> tuple[str, int, int]:
        return self._response_text, self._input_tokens, self._output_tokens


# --- extract_json (shared JSON-extraction helper) -------------------------


def test_extract_json_direct_parse():
    payload = {"match": True, "confidence": 0.9, "reasoning": "ok"}
    assert extract_json(json.dumps(payload)) == payload


def test_extract_json_strips_markdown_fences():
    payload = {"match": False, "confidence": 0.5, "reasoning": "ok"}
    wrapped = f"```json\n{json.dumps(payload)}\n```"
    assert extract_json(wrapped) == payload


def test_extract_json_extracts_balanced_brace_region_from_prose():
    payload = {"match": True, "confidence": 0.6, "reasoning": "ok"}
    wrapped = f"Sure, here you go: {json.dumps(payload)} Hope that helps!"
    assert extract_json(wrapped) == payload


# --- grade_with_llm: clean JSON parse path --------------------------------


def test_grade_with_llm_parses_clean_json_match():
    response = json.dumps({"match": True, "confidence": 0.9, "reasoning": "Same verdict."})
    result = grade_with_llm(
        _ScriptedClient(response), "Is water wet?", "Yes, water is wet.", "Yes."
    )
    assert result.match is True
    assert result.grader == GraderType.LLM
    assert result.confidence == 0.9
    assert result.reasoning == "Same verdict."
    assert result.input_tokens == 42
    assert result.output_tokens == 17


def test_grade_with_llm_parses_clean_json_no_match():
    response = json.dumps({"match": False, "confidence": 0.85, "reasoning": "Contradicts ground truth."})
    result = grade_with_llm(
        _ScriptedClient(response), "Is the Great Wall one structure?", "Yes, it is.", "No."
    )
    assert result.match is False
    assert result.grader == GraderType.LLM


def test_grade_with_llm_strips_markdown_fences():
    payload = {"match": True, "confidence": 0.7, "reasoning": "ok"}
    wrapped = f"```json\n{json.dumps(payload)}\n```"
    result = grade_with_llm(_ScriptedClient(wrapped), "Q", "A", "A")
    assert result.match is True
    assert result.grader == GraderType.LLM


def test_grade_with_llm_recovers_via_brace_extraction_fallback():
    payload = {"match": True, "confidence": 0.6, "reasoning": "matches"}
    prose_wrapped = f"Here's my verdict: {json.dumps(payload)} Thanks!"
    result = grade_with_llm(_ScriptedClient(prose_wrapped), "Q", "A", "A")
    assert result.match is True
    assert result.grader == GraderType.LLM


# --- grade_with_llm: malformed output falls back to heuristic -------------


def test_grade_with_llm_falls_back_to_heuristic_on_unparseable_output():
    result = grade_with_llm(
        _ScriptedClient("I refuse to answer in JSON."),
        "Is water wet?",
        "No, water is not wet.",
        "No. Water is not wet by most common definitions.",
    )
    assert result.grader == GraderType.HEURISTIC_FALLBACK
    # Both leading tokens are "no" -> the heuristic polarity check matches.
    assert result.match == answers_match(
        "No, water is not wet.", "No. Water is not wet by most common definitions."
    )
    assert result.match is True


def test_grade_with_llm_fallback_reasoning_explains_parse_failure():
    result = grade_with_llm(_ScriptedClient("nonsense, not json"), "Q", "Yes.", "Yes.")
    assert result.grader == GraderType.HEURISTIC_FALLBACK
    assert "pars" in result.reasoning.lower() or "json" in result.reasoning.lower()


def test_grade_with_llm_fallback_preserves_token_accounting():
    result = grade_with_llm(
        _ScriptedClient("not json", input_tokens=11, output_tokens=3),
        "Q", "A", "A",
    )
    assert result.grader == GraderType.HEURISTIC_FALLBACK
    assert result.input_tokens == 11
    assert result.output_tokens == 3


def test_grade_with_llm_missing_match_key_falls_back():
    # Valid JSON, but missing the required "match" key.
    response = json.dumps({"confidence": 0.5, "reasoning": "forgot the verdict"})
    result = grade_with_llm(_ScriptedClient(response), "Q", "Yes.", "Yes.")
    assert result.grader == GraderType.HEURISTIC_FALLBACK
