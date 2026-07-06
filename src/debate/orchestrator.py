"""DebateOrchestrator — coordinates the full debate loop."""
from __future__ import annotations
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Callable
from src.debate.schemas import (
    AgentRole, AgentResponse, ConvergenceReason, DebateTurn, DebateTranscript,
    DebateResult, GraphAnalysis, ConstitutionalReview,
)
from src.debate.agents import ProposerAgent, SkepticAgent, ConstitutionalAgent, _format_transcript
from src.debate.judge import JudgeAgent
from src.debate.convergence import ConvergenceDetector
from src.debate.graph import DebateGraphBuilder, GraphAnalyzer
from src.debate.utils import BaseLLMClient, AnthropicClient, DEFAULT_MODEL


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
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_rounds: int = 3,
        convergence_threshold: float = 0.65,
        enable_graph_analysis: bool = True,
        save_results: bool = True,
        results_dir: str = "results",
        skeptic_mode: str = "general",
        skeptic_modes: list[str] | None = None,
        human_role: str | None = None,
        human_input_fn: Callable[[str], str] | None = None,
        enable_constitutional: bool = False,
    ) -> None:
        if max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {max_rounds}")
        if client is None:
            client = AnthropicClient(model=model, temperature=temperature)
        self._client = client
        self._proposer = ProposerAgent(client)
        if skeptic_modes and len(skeptic_modes) > 1:
            self._skeptics = [SkepticAgent(client, mode=m) for m in skeptic_modes]
            self._panel_mode = True
        else:
            self._skeptics = [SkepticAgent(client, mode=skeptic_mode)]
            self._panel_mode = False
        # Keep self._skeptic as alias for backward compat
        self._skeptic = self._skeptics[0]
        self._judge = JudgeAgent(client)
        self._detector = ConvergenceDetector(repetition_threshold=convergence_threshold)
        self.max_rounds = max_rounds
        self.enable_graph_analysis = enable_graph_analysis
        self.save_results = save_results
        self.results_dir = Path(results_dir)
        self._human_role = human_role
        self._human_input_fn = human_input_fn
        self.enable_constitutional = enable_constitutional

    def run(self, question: str) -> DebateResult:
        """Run a complete debate for the given question and return a DebateResult."""
        transcript = DebateTranscript(question=question)
        total_input = 0
        total_output = 0
        converged = False
        convergence_reason: ConvergenceReason | None = None

        # --- Round 0: Proposer opening argument ---
        # The opening statement is round 0; the loop's first skeptic/proposer
        # exchange is round 1. (Previously both were labeled round 1.)
        if self._human_role == "proposer" and self._human_input_fn:
            prompt = f"Question: {question}\n\nYour turn as PROPOSER — make your opening argument:"
            human_text = self._human_input_fn(prompt)
            initial = AgentResponse(content=human_text, role=AgentRole.PROPOSER)
        else:
            initial = self._proposer.initial_argument(question)
        total_input += initial.input_tokens
        total_output += initial.output_tokens
        transcript.turns.append(
            DebateTurn(round_num=0, role=AgentRole.PROPOSER,
                       content=initial.content, token_count=initial.output_tokens)
        )

        # max_rounds >= 1 is enforced in __init__, so the loop below always
        # runs and sets rounds_used to the actual exchange count.
        rounds_used = 0

        for round_num in range(1, self.max_rounds + 1):
            # --- Each skeptic challenges in turn ---
            for skeptic in self._skeptics:
                if self._human_role == "skeptic" and self._human_input_fn:
                    history = _format_transcript(transcript)
                    prompt = (
                        f"Question: {question}\n\nDebate so far:\n{history}\n\n"
                        "Your turn as SKEPTIC — challenge the proposer's argument:"
                    )
                    human_text = self._human_input_fn(prompt)
                    skeptic_resp = AgentResponse(content=human_text, role=AgentRole.SKEPTIC)
                else:
                    skeptic_resp = skeptic.challenge(question, transcript)
                total_input += skeptic_resp.input_tokens
                total_output += skeptic_resp.output_tokens
                transcript.turns.append(
                    DebateTurn(round_num=round_num, role=AgentRole.SKEPTIC,
                               content=skeptic_resp.content,
                               token_count=skeptic_resp.output_tokens)
                )

            # Check convergence after all skeptics have spoken
            # Skip convergence detection for human turns
            if not (self._human_role == "skeptic" and self._human_input_fn):
                stop, reason = self._detector.should_stop(transcript)
                if stop:
                    converged, convergence_reason, rounds_used = True, reason, round_num
                    break

            # --- Proposer response ---
            if self._human_role == "proposer" and self._human_input_fn:
                history = _format_transcript(transcript)
                prompt = (
                    f"Question: {question}\n\nDebate so far:\n{history}\n\n"
                    "Your turn as PROPOSER — respond to the skeptic's challenge:"
                )
                human_text = self._human_input_fn(prompt)
                proposer_resp = AgentResponse(content=human_text, role=AgentRole.PROPOSER)
            else:
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
            # Skip convergence detection for human turns
            if not (self._human_role == "proposer" and self._human_input_fn):
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

        # --- Constitutional review ---
        constitutional_review: ConstitutionalReview | None = None
        if self.enable_constitutional:
            constitutional_agent = ConstitutionalAgent(self._client)
            constitutional_review = constitutional_agent.review(judge_output, transcript)

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
            panel_mode=self._panel_mode,
            human_role=self._human_role,
            constitutional_review=constitutional_review,
        )

        if self.save_results:
            self._save(result)

        return result

    def _save(self, result: DebateResult) -> str:
        """Save DebateResult as JSON and return the file path."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Replace anything that isn't alphanumeric/underscore/hyphen (rather than
        # denylisting a handful of characters) so path separators, colons,
        # question marks, and other characters invalid on Windows can't leak
        # into the filename or traverse out of results_dir.
        slug = re.sub(r"[^A-Za-z0-9_-]", "_", result.question[:40])
        path = self.results_dir / f"debate_{timestamp}_{slug}.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return str(path)
