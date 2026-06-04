"""Tests for the FastAPI REST server — uses TestClient, no live server needed."""
import pytest

try:
    from fastapi.testclient import TestClient
    from src.debate.server import app
    _fastapi_available = True
except ImportError:
    _fastapi_available = False

pytestmark = pytest.mark.skipif(not _fastapi_available, reason="fastapi not installed")

client = TestClient(app) if _fastapi_available else None


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_debate_endpoint_returns_200():
    resp = client.post("/debate", json={"question": "Is the sky blue?", "mock": True, "max_rounds": 1})
    assert resp.status_code == 200


def test_debate_endpoint_has_judge_output():
    resp = client.post("/debate", json={"question": "Is water wet?", "mock": True, "max_rounds": 1})
    data = resp.json()
    assert "judge_output" in data
    assert "verdict" in data["judge_output"]


def test_debate_endpoint_skeptic_mode():
    resp = client.post("/debate", json={
        "question": "Is AI safe?", "mock": True, "max_rounds": 1, "skeptic_mode": "logic"
    })
    assert resp.status_code == 200


def test_compare_endpoint_returns_200():
    resp = client.post("/compare", json={
        "question": "Is the earth round?", "mock": True, "max_rounds": 1,
        "mode_a": "general", "mode_b": "factual"
    })
    assert resp.status_code == 200


def test_compare_endpoint_has_verdict_match():
    resp = client.post("/compare", json={
        "question": "Test question?", "mock": True, "max_rounds": 1,
        "mode_a": "general", "mode_b": "general"
    })
    data = resp.json()
    assert "verdict_match" in data
    assert "confidence_delta" in data
