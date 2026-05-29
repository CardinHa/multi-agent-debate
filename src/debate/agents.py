from __future__ import annotations

from src.debate.schemas import AgentResponse, AgentRole, DebateTranscript
from src.debate.prompts import PROPOSER_SYSTEM_PROMPT, SKEPTIC_SYSTEM_PROMPT
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
    """Challenges the Proposer's argument to surface flaws and hallucinations."""

    def __init__(self, client: BaseLLMClient) -> None:
        self._client = client

    def challenge(self, question: str, transcript: DebateTranscript) -> AgentResponse:
        """Challenge the Proposer's latest argument given the full transcript."""
        history = _format_transcript(transcript)
        user_prompt = (
            f"Original question: {question}\n\n"
            f"Debate so far:\n{history}\n\n"
            "Identify the strongest flaw in the Proposer's argument. "
            "Do not repeat objections you or the Skeptic have already raised."
        )
        text, inp, out = self._client.call(SKEPTIC_SYSTEM_PROMPT, user_prompt)
        return AgentResponse(
            content=text, role=AgentRole.SKEPTIC,
            input_tokens=inp, output_tokens=out,
        )
