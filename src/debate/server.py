"""FastAPI server exposing the debate system as a REST API."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
except ImportError:
    raise ImportError("Install server dependencies: pip install 'multi-agent-debate[server]'")

from src.debate.schemas import DebateResult, DebateComparison
from src.debate.orchestrator import DebateOrchestrator
from src.debate.compare import run_comparison
from src.debate.utils import MockLLMClient


app = FastAPI(
    title="Multi-Agent Debate API",
    description="Scalable oversight via adversarial LLM debate.",
    version="0.1.0",
)


class DebateRequest(BaseModel):
    question: str
    max_rounds: int = 3
    skeptic_mode: str = "general"
    skeptic_modes: Optional[list[str]] = None
    enable_constitutional: bool = False
    mock: bool = True  # default True — safe for API without env vars


class CompareRequest(BaseModel):
    question: str
    mode_a: str = "general"
    mode_b: str = "logic"
    max_rounds: int = 3
    mock: bool = True


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "multi-agent-debate"}


@app.post("/debate", response_model=None)
def run_debate(req: DebateRequest) -> JSONResponse:
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
    client = MockLLMClient() if req.mock else None
    comp = run_comparison(
        question=req.question,
        client=client,
        config_a={"skeptic_mode": req.mode_a},
        config_b={"skeptic_mode": req.mode_b},
        max_rounds=req.max_rounds,
    )
    return JSONResponse(content=comp.model_dump(mode="json"))
