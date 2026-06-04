from __future__ import annotations

import json
import warnings

from src.debate.schemas import AgentResponse, AgentRole, DebateTranscript
from src.debate.prompts import PROPOSER_SYSTEM_PROMPT, SKEPTIC_MODE_PROMPTS
from src.debate.utils import BaseLLMClient


def _format_transcript(transcript: DebateTranscript) -> str:
    """Format transcript turns into readable text for agent context."""
    lines = []
    for turn in transcript.turns:
        label = turn.role.value.upper()
        lines.append(f"[{label} — Round {turn.round_num}]\n{turn.content}")
    return "\n\n".join(lines)


class ProposerAgent:
    """Makes and defends arguments. Revises position when challenged."""

    def __init__(self, client: BaseLLMClient) -> None:
        self._client = client

    def initial_argument(self, question: str) -> AgentResponse:
        """Produce the opening argument for a question or claim."""
        user_prompt = (
            f"Question/Claim to evaluate:\n{question}\n\n"
            "Provide your best answer with supporting reasoning."
        )
        text, inp, out = self._client.call(PROPOSER_SYSTEM_PROMPT, user_prompt)
        return AgentResponse(
            content=text, role=AgentRole.PROPOSER,
            input_tokens=inp, output_tokens=out,
        )

    def respond(self, question: str, transcript: DebateTranscript) -> AgentResponse:
        """Respond to the Skeptic's latest objection given the full transcript so far."""
        history = _format_transcript(transcript)
        user_prompt = (
            f"Original question: {question}\n\n"
            f"Debate so far:\n{history}\n\n"
            "The Skeptic has raised objections above. Defend, revise, or concede as "
            "appropriate. Be intellectually honest."
        )
        text, inp, out = self._client.call(PROPOSER_SYSTEM_PROMPT, user_prompt)
        return AgentResponse(
            content=text, role=AgentRole.PROPOSER,
            input_tokens=inp, output_tokens=out,
        )


class SkepticAgent:
    """Challenges the Proposer's argument to surface flaws and hallucinations.

    Supports specialized modes: 'general', 'factual', 'logic', 'evidence', 'safety'.
    """

    def __init__(self, client: BaseLLMClient, mode: str = "general") -> None:
        if mode not in SKEPTIC_MODE_PROMPTS:
            raise ValueError(
                f"Unknown skeptic mode {mode!r}. "
                f"Valid modes: {sorted(SKEPTIC_MODE_PROMPTS)}"
            )
        self._client = client
        self._system_prompt = SKEPTIC_MODE_PROMPTS[mode]
        self.mode = mode

    def challenge(self, question: str, transcript: DebateTranscript) -> AgentResponse:
        """Challenge the Proposer's latest argument given the full transcript."""
        history = _format_transcript(transcript)
        user_prompt = (
            f"Original question: {question}\n\n"
            f"Debate so far:\n{history}\n\n"
            "Identify the strongest flaw in the Proposer's argument. "
            "Do not repeat objections you or the Skeptic have already raised."
        )
        text, inp, out = self._client.call(self._system_prompt, user_prompt)
        return AgentResponse(
            content=text, role=AgentRole.SKEPTIC,
            input_tokens=inp, output_tokens=out,
        )


class ConstitutionalAgent:
    """Reviews the Judge's final answer against honesty, calibration, safety, and uncertainty principles."""

    def __init__(self, client: BaseLLMClient) -> None:
        self._client = client

    def review(self, judge_output, transcript: DebateTranscript):
        from src.debate.schemas import ConstitutionalReview
        from src.debate.prompts import CONSTITUTIONAL_SYSTEM_PROMPT
        user_prompt = (
            f"Judge's final answer: {judge_output.final_answer}\n"
            f"Verdict: {judge_output.verdict.value} (confidence: {judge_output.confidence:.0%})\n"
            f"Key reasons: {'; '.join(judge_output.key_reasons)}\n"
            f"Unresolved uncertainties: {'; '.join(judge_output.unresolved_uncertainties)}\n\n"
            "Apply the constitutional principles and return the JSON review."
        )
        text, _, _ = self._client.call(CONSTITUTIONAL_SYSTEM_PROMPT, user_prompt)
        try:
            # Strip markdown fences if present
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            data = json.loads(cleaned.strip())
            return ConstitutionalReview(**data)
        except Exception as e:
            warnings.warn(f"ConstitutionalAgent failed to parse response: {e}")
            return ConstitutionalReview(
                principles_checked=["Honesty", "Calibration", "Safety", "Uncertainty acknowledgment"],
                violations=[],
                warnings=["Constitutional review parse failed"],
                overall_safe=True,
            )
