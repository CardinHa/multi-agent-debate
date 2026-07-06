"""Convergence detection for debate halting."""
from __future__ import annotations
import re
from src.debate.schemas import DebateTurn, DebateTranscript, AgentRole, ConvergenceReason

# NOTE: "that's a fair point" is deliberately NOT a concession pattern.
# prompts.py explicitly instructs the Proposer to use exactly that phrase in
# *false* concessions ("that's a fair point, but…"), so matching it would halt
# debates on rhetorical acknowledgments that don't actually update the position.
_CONCESSION_PATTERNS = [
    r"\bi concede\b",
    r"\byou(?:'re| are) correct\b",
    r"\bi was wrong\b",
    r"\bi acknowledge\b.*\bflaw\b",
    r"\bi(?:'ll| will) revise\b",
    r"\bi(?:'m| am) wrong\b",
    r"\byou(?:'ve| have) convinced me\b",
]

_CONCESSION_RE = re.compile("|".join(_CONCESSION_PATTERNS), re.IGNORECASE)


def _tokenize(text: str) -> set[str]:
    """Lowercase and strip punctuation, returning a set of word tokens."""
    return set(re.sub(r"[^\w\s]", "", text.lower()).split())


def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity on word tokens."""
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


class ConvergenceDetector:
    """
    Heuristic convergence detector for debate transcripts.
    Checks for concessions, repeated objections, and stabilization.
    """

    def __init__(self, repetition_threshold: float = 0.40) -> None:
        self.repetition_threshold = repetition_threshold

    def detect_concession(self, text: str) -> bool:
        """Return True if the text contains an explicit concession."""
        return bool(_CONCESSION_RE.search(text))

    def detect_repetition(
        self, previous_turns: list[DebateTurn], new_turn: DebateTurn
    ) -> bool:
        """
        Return True if new_turn is substantially similar to any prior turn
        from the same role (indicating a repeated objection or argument).
        """
        same_role_turns = [t for t in previous_turns if t.role == new_turn.role]
        for prior in same_role_turns:
            if _token_overlap(prior.content, new_turn.content) >= self.repetition_threshold:
                return True
        return False

    def detect_stabilization(self, transcript: DebateTranscript) -> bool:
        """
        Return True if the last Proposer turn is highly similar to the
        second-to-last Proposer turn (argument has stopped evolving).
        """
        proposer_turns = [t for t in transcript.turns if t.role == AgentRole.PROPOSER]
        if len(proposer_turns) < 2:
            return False
        last, prev = proposer_turns[-1], proposer_turns[-2]
        return _token_overlap(last.content, prev.content) >= self.repetition_threshold

    def should_stop(
        self, transcript: DebateTranscript
    ) -> tuple[bool, ConvergenceReason | None]:
        """
        Evaluate the transcript and return (should_stop, reason).
        Checks concession → repetition → stabilization in priority order.
        """
        if not transcript.turns:
            return False, None

        # Check most recent Proposer turn for concession
        proposer_turns = [t for t in transcript.turns if t.role == AgentRole.PROPOSER]
        if proposer_turns and self.detect_concession(proposer_turns[-1].content):
            return True, ConvergenceReason.CONCESSION

        # Check most recent Skeptic turn for repetition
        skeptic_turns = [t for t in transcript.turns if t.role == AgentRole.SKEPTIC]
        if len(skeptic_turns) >= 2:
            prior_skeptic = skeptic_turns[:-1]
            if self.detect_repetition(prior_skeptic, skeptic_turns[-1]):
                return True, ConvergenceReason.REPETITION

        # Check Proposer stabilization
        if self.detect_stabilization(transcript):
            return True, ConvergenceReason.STABILIZATION

        return False, None
