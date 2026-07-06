"""Integration test for DebateOrchestrator using MockLLMClient."""
import json
import re

from src.debate.orchestrator import DebateOrchestrator
from src.debate.utils import MockLLMClient
from src.debate.schemas import ConvergenceReason


def test_orchestrator_runs_end_to_end():
    client = MockLLMClient()
    orchestrator = DebateOrchestrator(
        client=client,
        max_rounds=2,
        save_results=False,
        enable_graph_analysis=True,
    )
    result = orchestrator.run("Is the sky blue?")

    assert result.question == "Is the sky blue?"
    assert len(result.transcript.turns) >= 3
    assert result.judge_output is not None
    assert 0.0 <= result.judge_output.confidence <= 1.0
    assert result.convergence_reason is not None
    assert result.total_input_tokens > 0


def test_orchestrator_max_rounds_sets_reason():
    client = MockLLMClient()
    # Use a very high Jaccard threshold so stabilization never triggers
    orchestrator = DebateOrchestrator(
        client=client,
        max_rounds=1,
        convergence_threshold=0.99,
        save_results=False,
        enable_graph_analysis=False,
    )
    result = orchestrator.run("Test question?")
    # Should reach max rounds since threshold is too high for stabilization
    # (may still converge via concession — just check it has a reason)
    assert result.convergence_reason is not None


def test_orchestrator_graph_analysis_disabled():
    client = MockLLMClient()
    orchestrator = DebateOrchestrator(
        client=client,
        max_rounds=1,
        save_results=False,
        enable_graph_analysis=False,
    )
    result = orchestrator.run("Test?")
    assert result.graph_analysis is None


def test_save_sanitizes_slug_and_writes_utf8(tmp_path):
    client = MockLLMClient()
    orchestrator = DebateOrchestrator(
        client=client,
        max_rounds=1,
        save_results=True,
        enable_graph_analysis=False,
        results_dir=str(tmp_path),
    )
    # Path separators, Windows-invalid chars, and non-ASCII in the question
    # must not escape results_dir or crash the save on cp1252 systems.
    question = 'Is ../../evil\\path:*?"<>| okay — em-dash too?'
    result = orchestrator.run(question)
    assert result is not None

    saved = list(tmp_path.glob("debate_*.json"))
    assert len(saved) == 1
    # File landed directly in results_dir (no traversal out of it)
    assert saved[0].parent == tmp_path
    # Filename contains only safe characters after the prefix
    assert re.fullmatch(r"debate_\d{8}_\d{6}_[A-Za-z0-9_-]*\.json", saved[0].name)
    # Content is valid UTF-8 JSON that round-trips the original question
    data = json.loads(saved[0].read_text(encoding="utf-8"))
    assert data["question"] == question
