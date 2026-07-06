"""LLM client abstraction layer for the multi-agent debate system."""

import json
import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv

load_dotenv()

# Single source of truth for the default Claude model used across the CLI,
# orchestrator, and benchmark runner. Update this constant when migrating to a
# newer model rather than editing each call site individually.
DEFAULT_MODEL = "claude-sonnet-4-6"


class BaseLLMClient(ABC):
    """Abstract base for LLM clients."""

    @abstractmethod
    def call(self, system: str, user: str) -> tuple[str, int, int]:
        """
        Call the LLM.
        Returns (response_text, input_tokens, output_tokens).
        """


class AnthropicClient(BaseLLMClient):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file or export it as an environment variable."
            )
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Install the 'anthropic' package: pip install anthropic")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def call(self, system: str, user: str) -> tuple[str, int, int]:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text_block = next(
            (block for block in response.content if getattr(block, "type", None) == "text"),
            None,
        )
        if text_block is None:
            raise RuntimeError(
                "AnthropicClient.call: response contained no text content block "
                f"(stop_reason={response.stop_reason!r}). This can happen on a "
                "refusal or an all-tool-use response."
            )
        return text_block.text, response.usage.input_tokens, response.usage.output_tokens


class MockLLMClient(BaseLLMClient):
    """
    Deterministic mock client for unit tests.
    Returns predictable responses keyed on system prompt keywords.
    """

    _RESPONSES: dict[str, str] = {
        "You are the Proposer": (
            "I propose that the claim is well-supported by available evidence. "
            "My key assumption is that standard conditions apply. "
            "I acknowledge uncertainty around edge cases."
        ),
        "You are the Skeptic": (
            "The Proposer's argument relies on an unverified assumption. "
            "Specifically, the claim does not account for counterexample X. "
            "I challenge the Proposer to address this gap."
        ),
        "You are the Judge": json.dumps({
            "final_answer": "The claim is likely true under standard conditions.",
            "verdict": "supported",
            "confidence": 0.72,
            "key_reasons": ["Evidence is consistent", "Skeptic objection was minor"],
            "unresolved_uncertainties": ["Edge case X not fully resolved"],
            "proposer_changed_position": False,
            "skeptic_identified_valid_flaw": True,
            "debate_improved_answer": True,
            "recency_bias_check": (
                "Verdict reflects the full transcript. "
                "The Proposer's initial answer was strong; the Skeptic's final "
                "objection did not change the core conclusion."
            ),
        }),
        "moderator": json.dumps({"converged": False, "reason": "Debate still active"}),
        "You are the Constitutional": json.dumps({
            "principles_checked": ["Honesty", "Calibration", "Safety", "Uncertainty acknowledgment"],
            "violations": [],
            "warnings": ["Confidence level is acceptable but could be more precisely stated."],
            "overall_safe": True,
            "revised_answer": None,
        }),
    }

    def call(self, system: str, user: str) -> tuple[str, int, int]:
        for keyword, response in self._RESPONSES.items():
            if keyword in system:
                return response, len(user) // 4, len(response) // 4
        return "Mock response.", 10, 5
