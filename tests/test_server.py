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


# --- Server hardening: validation and real-LLM gating ----------------------

def test_debate_max_rounds_too_high_is_422():
    resp = client.post("/debate", json={"question": "Q?", "mock": True, "max_rounds": 11})
    assert resp.status_code == 422


def test_debate_max_rounds_zero_is_422():
    resp = client.post("/debate", json={"question": "Q?", "mock": True, "max_rounds": 0})
    assert resp.status_code == 422


def test_debate_max_rounds_negative_is_422():
    resp = client.post("/debate", json={"question": "Q?", "mock": True, "max_rounds": -1})
    assert resp.status_code == 422


def test_debate_question_too_long_is_422():
    resp = client.post("/debate", json={"question": "x" * 2001, "mock": True, "max_rounds": 1})
    assert resp.status_code == 422


def test_debate_question_at_max_length_is_ok():
    resp = client.post("/debate", json={"question": "x" * 2000, "mock": True, "max_rounds": 1})
    assert resp.status_code == 200


def test_debate_invalid_skeptic_mode_is_422_not_500():
    resp = client.post("/debate", json={
        "question": "Q?", "mock": True, "max_rounds": 1, "skeptic_mode": "not-a-real-mode"
    })
    assert resp.status_code == 422


def test_debate_invalid_skeptic_modes_list_is_422():
    resp = client.post("/debate", json={
        "question": "Q?", "mock": True, "max_rounds": 1,
        "skeptic_modes": ["logic", "bogus-mode"],
    })
    assert resp.status_code == 422


def test_compare_invalid_mode_a_is_422():
    resp = client.post("/compare", json={
        "question": "Q?", "mock": True, "max_rounds": 1,
        "mode_a": "bogus-mode", "mode_b": "logic",
    })
    assert resp.status_code == 422


def test_compare_max_rounds_too_high_is_422():
    resp = client.post("/compare", json={
        "question": "Q?", "mock": True, "max_rounds": 20,
        "mode_a": "general", "mode_b": "logic",
    })
    assert resp.status_code == 422


def test_debate_real_mode_forbidden_without_env_var(monkeypatch):
    monkeypatch.delenv("ALLOW_REAL_LLM", raising=False)
    resp = client.post("/debate", json={"question": "Q?", "mock": False, "max_rounds": 1})
    assert resp.status_code == 403
    assert "ALLOW_REAL_LLM" in resp.json()["detail"]


def test_compare_real_mode_forbidden_without_env_var(monkeypatch):
    monkeypatch.delenv("ALLOW_REAL_LLM", raising=False)
    resp = client.post("/compare", json={
        "question": "Q?", "mock": False, "max_rounds": 1,
        "mode_a": "general", "mode_b": "logic",
    })
    assert resp.status_code == 403


def test_debate_mock_mode_unaffected_by_missing_env_var(monkeypatch):
    monkeypatch.delenv("ALLOW_REAL_LLM", raising=False)
    resp = client.post("/debate", json={"question": "Q?", "mock": True, "max_rounds": 1})
    assert resp.status_code == 200
