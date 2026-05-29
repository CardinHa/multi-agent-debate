"""Debate graph construction and analysis using NetworkX."""
from __future__ import annotations
import networkx as nx
from src.debate.schemas import (
    DebateTranscript, GraphAnalysis, AgentRole, DebateTurn
)
from src.debate.convergence import ConvergenceDetector


def _edge_type(source: DebateTurn, target: DebateTurn) -> str:
    """Infer the relationship type from source turn to target turn."""
    detector = ConvergenceDetector()
    src_role = source.role
    tgt_role = target.role

    if src_role == AgentRole.PROPOSER and tgt_role == AgentRole.SKEPTIC:
        return "proposes"
    if src_role == AgentRole.SKEPTIC and tgt_role == AgentRole.PROPOSER:
        return "rebuts"

    if tgt_role == AgentRole.PROPOSER:
        if detector.detect_concession(target.content):
            return "concedes"
        return "revises"

    return "responds"


class DebateGraphBuilder:
    """Builds a directed NetworkX graph from a DebateTranscript."""

    def build(self, transcript: DebateTranscript) -> nx.DiGraph:
        """
        Each turn is a node; edges connect sequential turns.
        Edge attributes carry the relationship type.
        """
        graph = nx.DiGraph()
        turns = transcript.turns

        for i, turn in enumerate(turns):
            node_id = f"turn_{i}"
            graph.add_node(
                node_id,
                role=turn.role.value,
                round_num=turn.round_num,
                content=turn.content[:120],
            )

        for i in range(len(turns) - 1):
            src_id = f"turn_{i}"
            tgt_id = f"turn_{i + 1}"
            edge_type = _edge_type(turns[i], turns[i + 1])
            graph.add_edge(src_id, tgt_id, edge_type=edge_type)

        return graph


class GraphAnalyzer:
    """Computes metrics from a debate graph + transcript."""

    def __init__(self, graph: nx.DiGraph, transcript: DebateTranscript) -> None:
        self._graph = graph
        self._transcript = transcript
        self._detector = ConvergenceDetector()

    def _count_edge_type(self, edge_type: str) -> int:
        return sum(
            1 for _, _, d in self._graph.edges(data=True)
            if d.get("edge_type") == edge_type
        )

    def analyze(self) -> GraphAnalysis:
        graph = self._graph
        turns = self._transcript.turns
        detector = self._detector

        num_concessions = sum(
            1 for t in turns
            if t.role == AgentRole.PROPOSER and detector.detect_concession(t.content)
        )

        proposer_turns = [t for t in turns if t.role == AgentRole.PROPOSER]
        num_revisions = 0
        if len(proposer_turns) >= 2:
            for i in range(1, len(proposer_turns)):
                overlap = len(
                    set(proposer_turns[i - 1].content.split()) &
                    set(proposer_turns[i].content.split())
                ) / max(len(set(proposer_turns[i].content.split())), 1)
                if 0.1 < overlap < 0.85:
                    num_revisions += 1

        has_cycles = not nx.is_directed_acyclic_graph(graph)
        argument_depth = len(turns)

        centrality = nx.degree_centrality(graph)

        edge_type_counts: dict[str, int] = {}
        for _, _, data in graph.edges(data=True):
            et = data.get("edge_type", "unknown")
            edge_type_counts[et] = edge_type_counts.get(et, 0) + 1

        return GraphAnalysis(
            num_turns=len(turns),
            num_claims=len([t for t in turns if t.role == AgentRole.PROPOSER]),
            num_rebuttals=self._count_edge_type("rebuts"),
            num_concessions=num_concessions,
            num_revisions=num_revisions,
            has_cycles=has_cycles,
            proposer_revisions_caused_by_skeptic=num_revisions,
            argument_depth=argument_depth,
            centrality_scores=centrality,
            edge_type_counts=edge_type_counts,
        )

    def visualize(self, output_path: str) -> None:
        """Save a PNG visualization of the debate graph."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        graph = self._graph
        pos = nx.spring_layout(graph, seed=42)
        colors = {
            "proposer": "#4CAF50",
            "skeptic": "#F44336",
            "judge": "#2196F3",
        }
        node_colors = [
            colors.get(graph.nodes[n].get("role", ""), "#9E9E9E")
            for n in graph.nodes
        ]
        edge_labels = {
            (u, v): d.get("edge_type", "")
            for u, v, d in graph.edges(data=True)
        }

        plt.figure(figsize=(12, 7))
        nx.draw(
            graph, pos, node_color=node_colors, with_labels=True,
            font_size=8, node_size=800, arrows=True,
        )
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)
        plt.title("Debate Argument Graph")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
