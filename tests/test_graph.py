"""Unit tests for debate graph construction and analysis."""
import pytest
import networkx as nx
from src.debate.graph import DebateGraphBuilder, GraphAnalyzer
from src.debate.schemas import DebateTurn, DebateTranscript, AgentRole


def _sample_transcript() -> DebateTranscript:
    return DebateTranscript(
        question="Is X true?",
        turns=[
            DebateTurn(round_num=1, role=AgentRole.PROPOSER, content="X is true because A."),
            DebateTurn(round_num=1, role=AgentRole.SKEPTIC,
                       content="A is unverified. What about B?"),
            DebateTurn(round_num=2, role=AgentRole.PROPOSER,
                       content="I revise: X is true because A and B."),
            DebateTurn(round_num=2, role=AgentRole.SKEPTIC,
                       content="B is weak. Consider counterexample C."),
            DebateTurn(round_num=3, role=AgentRole.PROPOSER,
                       content="I concede that C undermines B."),
        ]
    )


def test_graph_has_correct_node_count():
    transcript = _sample_transcript()
    builder = DebateGraphBuilder()
    graph = builder.build(transcript)
    assert graph.number_of_nodes() == 5


def test_graph_edges_exist():
    transcript = _sample_transcript()
    builder = DebateGraphBuilder()
    graph = builder.build(transcript)
    assert graph.number_of_edges() > 0


def test_graph_analysis_counts():
    transcript = _sample_transcript()
    builder = DebateGraphBuilder()
    graph = builder.build(transcript)
    analyzer = GraphAnalyzer(graph, transcript)
    analysis = analyzer.analyze()
    assert analysis.num_turns == 5
    assert analysis.num_concessions >= 1
    assert analysis.num_rebuttals >= 1


def test_graph_analysis_is_dag():
    transcript = _sample_transcript()
    builder = DebateGraphBuilder()
    graph = builder.build(transcript)
    analyzer = GraphAnalyzer(graph, transcript)
    analysis = analyzer.analyze()
    # A simple linear debate should not have cycles
    assert analysis.has_cycles is False
