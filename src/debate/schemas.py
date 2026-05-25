"""Pydantic schemas — single source of truth for all debate data structures."""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    PROPOSER = "proposer"
    SKEPTIC = "skeptic"
    JUDGE = "judge"


class ConvergenceReason(str, Enum):
    CONCESSION = "concession"
    REPETITION = "repetition"
    MAX_ROUNDS = "max_rounds"
    STABILIZATION = "stabilization"


class DebateTurn(BaseModel):
    round_num: int
    role: AgentRole
    content: str
    token_count: Optional[int] = None


class DebateTranscript(BaseModel):
    question: str
    turns: list[DebateTurn] = Field(default_factory=list)


class AgentResponse(BaseModel):
    content: str
    role: AgentRole
    input_tokens: int = 0
    output_tokens: int = 0


class JudgeOutput(BaseModel):
    final_answer: str
    verdict: str  # "supported" | "refuted" | "uncertain"
    confidence: float = Field(ge=0.0, le=1.0)
    key_reasons: list[str]
    unresolved_uncertainties: list[str]
    proposer_changed_position: bool
    skeptic_identified_valid_flaw: bool
    debate_improved_answer: bool
    recency_bias_check: str


class GraphAnalysis(BaseModel):
    num_turns: int
    num_claims: int
    num_rebuttals: int
    num_concessions: int
    num_revisions: int
    has_cycles: bool
    proposer_revisions_caused_by_skeptic: int
    argument_depth: int
    centrality_scores: dict[str, float]
    edge_type_counts: dict[str, int]


class DebateResult(BaseModel):
    question: str
    transcript: DebateTranscript
    judge_output: JudgeOutput
    converged: bool
    convergence_reason: Optional[ConvergenceReason]
    rounds_used: int
    graph_analysis: Optional[GraphAnalysis] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class BenchmarkExample(BaseModel):
    id: str
    question: str
    ground_truth: str
    category: str


class BenchmarkResult(BaseModel):
    example_id: str
    question: str
    category: str
    ground_truth: str
    baseline_answer: str
    debate_final_answer: str
    baseline_correct: Optional[bool] = None
    debate_correct: Optional[bool] = None
    debate_improved: bool
    debate_confidence: float
    rounds_used: int
    converged: bool
    total_tokens: int
