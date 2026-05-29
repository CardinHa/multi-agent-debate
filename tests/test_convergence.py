"""Unit tests for ConvergenceDetector."""
import pytest
from src.debate.convergence import ConvergenceDetector
from src.debate.schemas import DebateTurn, DebateTranscript, AgentRole


def _make_transcript(*contents: tuple) -> DebateTranscript:
    turns = [
        DebateTurn(round_num=r, role=role, content=text)
        for role, text, r in contents
    ]
    return DebateTranscript(question="Test?", turns=turns)


def test_detect_concession_positive():
    detector = ConvergenceDetector()
    assert detector.detect_concession("I concede that your point is valid.") is True
    assert detector.detect_concession("You are correct, I was wrong.") is True


def test_detect_concession_negative():
    detector = ConvergenceDetector()
    assert detector.detect_concession("I maintain my position.") is False


def test_detect_repetition_same_text():
    detector = ConvergenceDetector()
    turns = [
        DebateTurn(round_num=1, role=AgentRole.SKEPTIC,
                   content="The assumption is unverified."),
        DebateTurn(round_num=2, role=AgentRole.SKEPTIC,
                   content="The assumption remains unverified and unsupported."),
    ]
    assert detector.detect_repetition(turns[:-1], turns[-1]) is True


def test_detect_repetition_new_content():
    detector = ConvergenceDetector()
    prior = [DebateTurn(round_num=1, role=AgentRole.SKEPTIC,
                        content="The assumption is unverified.")]
    new_turn = DebateTurn(round_num=2, role=AgentRole.SKEPTIC,
                          content="The Proposer ignores the counterexample of Y.")
    assert detector.detect_repetition(prior, new_turn) is False


def test_should_stop_after_concession():
    detector = ConvergenceDetector()
    transcript = _make_transcript(
        (AgentRole.PROPOSER, "My claim is X.", 1),
        (AgentRole.SKEPTIC, "Your claim ignores Y.", 1),
        (AgentRole.PROPOSER, "I concede that Y is a valid counterexample.", 2),
    )
    stop, reason = detector.should_stop(transcript)
    assert stop is True
    assert reason.value == "concession"


def test_should_not_stop_early():
    detector = ConvergenceDetector()
    transcript = _make_transcript(
        (AgentRole.PROPOSER, "My claim is X.", 1),
        (AgentRole.SKEPTIC, "Your claim ignores Y.", 1),
        (AgentRole.PROPOSER, "Y does not apply because Z.", 2),
    )
    stop, _ = detector.should_stop(transcript)
    assert stop is False
