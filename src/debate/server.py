"""FastAPI server exposing the debate system as a REST API."""
from __future__ import annotations
import os
from typing import Optional
from pydantic import BaseModel, Field, field_validator

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
except ImportError:
    raise ImportError("Install server dependencies: pip install 'multi-agent-debate[server]'")

from src.debate.schemas import DebateResult, DebateComparison
from src.debate.orchestrator import DebateOrchestrator
from src.debate.compare import run_comparison
from src.debate.utils import MockLLMClient
from src.debate.prompts import SKEPTIC_MODE_PROMPTS


app = FastAPI(
    title="Multi-Agent Debate API",
    description="Scalable oversight via adversarial LLM debate.",
    version="0.1.0",
)

# Known-valid skeptic personality modes (see src.debate.prompts.SKEPTIC_MODE_PROMPTS).
_VALID_SKEPTIC_MODES = sorted(SKEPTIC_MODE_PROMPTS)

# Question length cap — generous for a debate claim/question, but bounds
# request size and downstream prompt/token cost.
_MAX_QUESTION_LENGTH = 2000


def _validate_skeptic_mode(value: str) -> str:
    if value not in _VALID_SKEPTIC_MODES:
        raise ValueError(
            f"Invalid skeptic mode {value!r}. Valid modes: {_VALID_SKEPTIC_MODES}"
        )
    return value


def _validate_skeptic_modes_list(value: Optional[list[str]]) -> Optional[list[str]]:
    if value is None:
        return value
    invalid = sorted({m for m in value if m not in _VALID_SKEPTIC_MODES})
    if invalid:
        raise ValueError(
            f"Invalid skeptic modes {invalid}. Valid modes: {_VALID_SKEPTIC_MODES}"
        )
    return value


class DebateRequest(BaseModel):
    question: str = Field(..., max_length=_MAX_QUESTION_LENGTH)
    max_rounds: int = Field(3, ge=1, le=10)
    skeptic_mode: str = "general"
    skeptic_modes: Optional[list[str]] = None
    enable_constitutional: bool = False
    mock: bool = True  # default True — safe for API without env vars

    @field_validator("skeptic_mode")
    @classmethod
    def _check_skeptic_mode(cls, v: str) -> str:
        return _validate_skeptic_mode(v)

    @field_validator("skeptic_modes")
    @classmethod
    def _check_skeptic_modes(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        return _validate_skeptic_modes_list(v)


class CompareRequest(BaseModel):
    question: str = Field(..., max_length=_MAX_QUESTION_LENGTH)
    mode_a: str = "general"
    mode_b: str = "logic"
    max_rounds: int = Field(3, ge=1, le=10)
    mock: bool = True

    @field_validator("mode_a", "mode_b")
    @classmethod
    def _check_mode(cls, v: str) -> str:
        return _validate_skeptic_mode(v)


def _require_real_llm_allowed(mock: bool) -> None:
    """Gate real (non-mock) LLM calls behind an explicit opt-in env var.

    Without this, a publicly reachable server with ANTHROPIC_API_KEY set would
    let any caller spend real API budget just by passing mock=false.
    """
    if not mock and os.environ.get("ALLOW_REAL_LLM") != "1":
        raise HTTPException(
            status_code=403,
            detail=(
                "Real LLM calls are disabled on this server. Pass mock=true, "
                "or set the ALLOW_REAL_LLM=1 environment variable on the "
                "server to allow real Anthropic API calls."
            ),
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "multi-agent-debate"}


@app.post("/debate", response_model=None)
def run_debate(req: DebateRequest) -> JSONResponse:
    _require_real_llm_allowed(req.mock)
    client = MockLLMClient() if req.mock else None
    orch = DebateOrchestrator(
        client=client,
        max_rounds=req.max_rounds,
        save_results=False,
        enable_graph_analysis=False,
        skeptic_mode=req.skeptic_mode,
        skeptic_modes=req.skeptic_modes,
        enable_constitutional=req.enable_constitutional,
    )
    result = orch.run(req.question)
    return JSONResponse(content=result.model_dump(mode="json"))


@app.post("/compare", response_model=None)
def run_compare(req: CompareRequest) -> JSONResponse:
    _require_real_llm_allowed(req.mock)
    client = MockLLMClient() if req.mock else None
    comp = run_comparison(
        question=req.question,
        client=client,
        config_a={"skeptic_mode": req.mode_a},
        config_b={"skeptic_mode": req.mode_b},
        max_rounds=req.max_rounds,
    )
    return JSONResponse(content=comp.model_dump(mode="json"))
