"""Tests for BenchmarkRunner resilience: per-example failures must not abort the run."""
from __future__ import annotations

import json

import pytest

from src.debate.benchmark import BenchmarkRunner
from src.debate.utils import BaseLLMClient, MockLLMClient
from src.debate.schemas import BenchmarkExample


class _FailOnKeywordClient(BaseLLMClient):
    """Raises for any prompt mentioning 'boom', otherwise delegates to MockLLMClient."""

    def __init__(self) -> None:
        self._mock = MockLLMClient()

    def call(self, system: str, user: str) -> tuple[str, int, int]:
        if "boom" in user:
            raise RuntimeError("simulated LLM failure")
        return self._mock.call(system, user)


def _write_dataset(tmp_path, examples: list[dict]) -> str:
    path = tmp_path / "claims.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in examples), encoding="utf-8")
    return str(path)


def test_failed_example_recorded_and_run_continues(tmp_path):
    dataset = _write_dataset(tmp_path, [
        {"id": "ok_1", "question": "Is water wet?", "ground_truth": "Yes.", "category": "factual"},
        {"id": "boom_1", "question": "boom trigger", "ground_truth": "No.", "category": "factual"},
        {"id": "ok_2", "question": "Is the sky blue?", "ground_truth": "Yes.", "category": "factual"},
    ])
    runner = BenchmarkRunner(client=_FailOnKeywordClient(), results_dir=str(tmp_path / "results"))
    results = runner.run(dataset)

    assert len(results) == 3
    by_id = {r.example_id: r for r in results}
    assert by_id["ok_1"].error is None
    assert by_id["ok_2"].error is None
    assert by_id["boom_1"].error is not None
    assert "simulated LLM failure" in by_id["boom_1"].error


def test_incremental_jsonl_written_per_example(tmp_path):
    dataset = _write_dataset(tmp_path, [
        {"id": "a", "question": "Is water wet?", "ground_truth": "Yes.", "category": "factual"},
        {"id": "b", "question": "boom trigger", "ground_truth": "No.", "category": "factual"},
    ])
    results_dir = tmp_path / "results"
    runner = BenchmarkRunner(client=_FailOnKeywordClient(), results_dir=str(results_dir))
    runner.run(dataset)

    partial_files = list(results_dir.glob("benchmark_*_partial.jsonl"))
    assert len(partial_files) == 1
    lines = partial_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    ids = {json.loads(line)["example_id"] for line in lines}
    assert ids == {"a", "b"}


def test_summary_counts_failures_explicitly(tmp_path):
    dataset = _write_dataset(tmp_path, [
        {"id": "ok_1", "question": "Is water wet?", "ground_truth": "Yes.", "category": "factual"},
        {"id": "boom_1", "question": "boom trigger", "ground_truth": "No.", "category": "factual"},
    ])
    runner = BenchmarkRunner(client=_FailOnKeywordClient(), results_dir=str(tmp_path / "results"))
    results = runner.run(dataset)
    summary = runner.summary(results)

    assert summary["total_examples"] == 2
    assert summary["failed_examples"] == 1


def test_anthropic_client_raises_clear_error_on_empty_content():
    from src.debate.utils import AnthropicClient

    class _FakeUsage:
        input_tokens = 5
        output_tokens = 0

    class _FakeResponse:
        content: list = []
        stop_reason = "tool_use"
        usage = _FakeUsage()

    class _FakeMessages:
        def create(self, **kwargs):
            return _FakeResponse()

    class _FakeAnthropicSDKClient:
        messages = _FakeMessages()

    client = AnthropicClient.__new__(AnthropicClient)
    client._client = _FakeAnthropicSDKClient()
    client.model = "claude-sonnet-4-6"
    client.temperature = 0.7
    client.max_tokens = 100

    with pytest.raises(RuntimeError, match="no text content block"):
        client.call("system", "user")
