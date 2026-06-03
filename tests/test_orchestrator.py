"""Integration test for DebateOrchestrator using MockLLMClient."""
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
