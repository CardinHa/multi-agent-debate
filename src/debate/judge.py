"""Judge agent — evaluates the full debate transcript and returns a JudgeOutput."""
from __future__ import annotations

import json
import re
import warnings

from src.debate.schemas import JudgeOutput, DebateTranscript, AgentRole, VerdictType
from src.debate.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE
from src.debate.utils import BaseLLMClient


def _summarize_side(transcript: DebateTranscript, role: AgentRole) -> str:
    """Extract all turns for one role and return a bullet summary."""
    turns = [t for t in transcript.turns if t.role == role]
    if not turns:
        return "(no turns)"
    points = []
    for t in turns:
        # Take first sentence as a representative claim
        first_sentence = t.content.split(".")[0].strip()
        points.append(f"- {first_sentence}.")
    return "\n".join(points)


def _format_full_transcript(transcript: DebateTranscript) -> str:
    lines = []
    for turn in transcript.turns:
        label = turn.role.value.upper()
        lines.append(f"[{label} — Round {turn.round_num}]\n{turn.content}")
    return "\n\n".join(lines)


def _extract_json(text: str) -> dict:
    """Extract JSON from model output, stripping markdown fences if present."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    return json.loads(cleaned)


class JudgeAgent:
    """Evaluates the full debate transcript and returns a JudgeOutput."""

    def __init__(self, client: BaseLLMClient) -> None:
        self._client = client

    def evaluate(self, transcript: DebateTranscript) -> tuple[JudgeOutput, int, int]:
        """
        Run judgment over the full transcript.
        Returns (JudgeOutput, input_tokens, output_tokens).
        """
        full_text = _format_full_transcript(transcript)
        proposer_summary = _summarize_side(transcript, AgentRole.PROPOSER)
        skeptic_summary = _summarize_side(transcript, AgentRole.SKEPTIC)

        user_prompt = JUDGE_USER_TEMPLATE.format(
            question=transcript.question,
            transcript=full_text,
            proposer_summary=proposer_summary,
            skeptic_summary=skeptic_summary,
        )

        raw, inp, out = self._client.call(JUDGE_SYSTEM_PROMPT, user_prompt)

        try:
            data = _extract_json(raw)
            judge_output = JudgeOutput(**data)
        except Exception as exc:
            warnings.warn(
                f"JudgeAgent: failed to parse model output as JudgeOutput: {exc!r}. "
                "Falling back to uncertain verdict.",
                stacklevel=2,
            )
            # Fallback: return an uncertain verdict with the raw text as answer
            judge_output = JudgeOutput(
                final_answer=raw[:500],
                verdict=VerdictType.UNCERTAIN,
                confidence=0.0,
                key_reasons=["Judge output could not be parsed as JSON."],
                unresolved_uncertainties=["Full judge response unparseable."],
                proposer_changed_position=False,
                skeptic_identified_valid_flaw=False,
                debate_improved_answer=False,
                recency_bias_check="N/A — parse error.",
            )

        return judge_output, inp, out
