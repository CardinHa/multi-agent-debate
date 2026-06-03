"""DebateOrchestrator — coordinates the full debate loop."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from src.debate.schemas import (
    AgentRole, ConvergenceReason, DebateTurn, DebateTranscript,
    DebateResult, GraphAnalysis,
)
from src.debate.agents import ProposerAgent, SkepticAgent
from src.debate.judge import JudgeAgent
from src.debate.convergence import ConvergenceDetector
from src.debate.graph import DebateGraphBuilder, GraphAnalyzer
from src.debate.utils import BaseLLMClient, AnthropicClient


class DebateOrchestrator:
    """
    Orchestrates the full proposer–skeptic debate loop.

    Lifecycle:
    1. Proposer makes initial argument.
    2. Skeptic challenges.
    3. Proposer responds.
    4. Repeat until convergence or max_rounds.
    5. Judge evaluates full transcript.
    6. Optionally run graph analysis.
    """

    def __init__(
        self,
        client: BaseLLMClient | None = None,
        model: str = "claude-3-5-sonnet-latest",
        temperature: float = 0.7,
        max_rounds: int = 3,
        convergence_threshold: float = 0.65,
        enable_graph_analysis: bool = True,
        save_results: bool = True,
        results_dir: str = "results",
        skeptic_mode: str = "general",
    ) -> None:
        if client is None:
            client = AnthropicClient(model=model, temperature=temperature)
        self._client = client
        self._proposer = ProposerAgent(client)
        self._skeptic = SkepticAgent(client, mode=skeptic_mode)
        self._judge = JudgeAgent(client)
        self._detector = ConvergenceDetector(repetition_threshold=convergence_threshold)
        self.max_rounds = max_rounds
        self.enable_graph_analysis = enable_graph_analysis
        self.save_results = save_results
        self.results_dir = Path(results_dir)

    def run(self, question: str) -> DebateResult:
        """Run a complete debate for the given question and return a DebateResult."""
        transcript = DebateTranscript(question=question)
        total_input = 0
        total_output = 0
        converged = False
        convergence_reason: ConvergenceReason | None = None

        # --- Round 1: Proposer initial argument ---
        initial = self._proposer.initial_argument(question)
        total_input += initial.input_tokens
        total_output += initial.output_tokens
        transcript.turns.append(
            DebateTurn(round_num=1, role=AgentRole.PROPOSER,
                       content=initial.content, token_count=initial.output_tokens)
        )

        rounds_used = 1

        for round_num in range(1, self.max_rounds + 1):
            # --- Skeptic challenge ---
            skeptic_resp = self._skeptic.challenge(question, transcript)
            total_input += skeptic_resp.input_tokens
            total_output += skeptic_resp.output_tokens
            transcript.turns.append(
                DebateTurn(round_num=round_num, role=AgentRole.SKEPTIC,
                           content=skeptic_resp.content,
                           token_count=skeptic_resp.output_tokens)
            )

            # Check convergence after skeptic turn
            stop, reason = self._detector.should_stop(transcript)
            if stop:
                converged, convergence_reason, rounds_used = True, reason, round_num
                break

            # --- Proposer response ---
            proposer_resp = self._proposer.respond(question, transcript)
            total_input += proposer_resp.input_tokens
            total_output += proposer_resp.output_tokens
            transcript.turns.append(
                DebateTurn(round_num=round_num, role=AgentRole.PROPOSER,
                           content=proposer_resp.content,
                           token_count=proposer_resp.output_tokens)
            )

            rounds_used = round_num

            # Check convergence after proposer response
            stop, reason = self._detector.should_stop(transcript)
            if stop:
                converged, convergence_reason, rounds_used = True, reason, round_num
                break

        if not converged:
            convergence_reason = ConvergenceReason.MAX_ROUNDS

        # --- Judge evaluation ---
        judge_output, j_inp, j_out = self._judge.evaluate(transcript)
        total_input += j_inp
        total_output += j_out

        # --- Graph analysis ---
        graph_analysis: GraphAnalysis | None = None
        if self.enable_graph_analysis:
            builder = DebateGraphBuilder()
            graph = builder.build(transcript)
            analyzer = GraphAnalyzer(graph, transcript)
            graph_analysis = analyzer.analyze()

        result = DebateResult(
            question=question,
            transcript=transcript,
            judge_output=judge_output,
            converged=converged,
            convergence_reason=convergence_reason,
            rounds_used=rounds_used,
            graph_analysis=graph_analysis,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
        )

        if self.save_results:
            self._save(result)

        return result

    def _save(self, result: DebateResult) -> str:
        """Save DebateResult as JSON and return the file path."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = result.question[:40].replace(" ", "_").replace("?", "").replace("/", "_")
        path = self.results_dir / f"debate_{timestamp}_{slug}.json"
        path.write_text(result.model_dump_json(indent=2))
        return str(path)
