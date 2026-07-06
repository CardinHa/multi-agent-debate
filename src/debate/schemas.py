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


class VerdictType(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNCERTAIN = "uncertain"


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
    verdict: VerdictType
    confidence: float = Field(ge=0.0, le=1.0)
    key_reasons: list[str]
    unresolved_uncertainties: list[str]
    proposer_changed_position: bool
    skeptic_identified_valid_flaw: bool
    debate_improved_answer: bool
    recency_bias_check: str
    # True when the Judge's raw output could not be parsed as JSON and this
    # JudgeOutput is the uncertain-verdict fallback rather than a real judgment.
    parse_failed: bool = False


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


class ConstitutionalReview(BaseModel):
    principles_checked: list[str]
    violations: list[str]
    warnings: list[str]
    overall_safe: bool
    revised_answer: Optional[str] = None


class DebateResult(BaseModel):
    question: str
    transcript: DebateTranscript
    judge_output: JudgeOutput
    converged: bool
    convergence_reason: Optional[ConvergenceReason] = None
    rounds_used: int
    graph_analysis: Optional[GraphAnalysis] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    panel_mode: bool = False
    human_role: Optional[str] = None
    constitutional_review: Optional[ConstitutionalReview] = None


class DebateComparison(BaseModel):
    question: str
    config_a_label: str
    config_b_label: str
    result_a: DebateResult
    result_b: DebateResult
    verdict_match: bool
    confidence_delta: float   # result_b.confidence - result_a.confidence
    convergence_match: bool


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
    baseline_answer: str = ""
    debate_final_answer: str = ""
    baseline_correct: Optional[bool] = None
    debate_correct: Optional[bool] = None
    debate_improved: bool = False
    # True when the baseline answer was correct but the debate's final answer was not.
    debate_regressed: bool = False
    debate_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rounds_used: int = 0
    converged: bool = False
    total_tokens: int = 0
    # Set when this example failed to run; other fields are placeholders.
    error: Optional[str] = None
    # True when the debate's JudgeOutput was an unparseable-JSON fallback.
    judge_parse_failed: bool = False


class CategoryStats(BaseModel):
    category: str
    total: int
    baseline_accuracy: float
    debate_accuracy: float
    improvement_rate: float
    avg_confidence: float


class CalibrationBin(BaseModel):
    confidence_low: float   # lower bound of bin, e.g. 0.0
    confidence_high: float  # upper bound, e.g. 0.2
    count: int
    actual_accuracy: float  # fraction of debate_correct in this bin


class CalibrationReport(BaseModel):
    total_examples: int
    overall_baseline_accuracy: float
    overall_debate_accuracy: float
    overall_improvement_rate: float
    per_category: list[CategoryStats]
    calibration_bins: list[CalibrationBin]
    # Count of results excluded from calibration_bins because the Judge's
    # output failed to parse as JSON (confidence is a meaningless 0.0 fallback).
    excluded_parse_failures: int = 0
