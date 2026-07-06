"""Guard test: DEFAULT_MODEL must not be a known-retired Anthropic model ID.

Runs in CI so a retired default model fails the build instead of failing at
runtime with a 404 from the API.
"""
from __future__ import annotations

import re

from multi_agent_debate.debate.utils import DEFAULT_MODEL

# Known-retired model ID prefixes. Extend this list as models are retired.
_RETIRED_MODEL_PATTERNS = [
    r"^claude-1\b",
    r"^claude-2\b",
    r"^claude-instant",
    r"^claude-3-haiku",
    r"^claude-3-sonnet",
    r"^claude-3-opus",
    r"^claude-3-5-haiku",
    r"^claude-3-5-sonnet",
    r"^claude-3-7-sonnet",
]


def test_default_model_is_not_retired():
    for pattern in _RETIRED_MODEL_PATTERNS:
        assert not re.match(pattern, DEFAULT_MODEL), (
            f"DEFAULT_MODEL {DEFAULT_MODEL!r} matches retired-model pattern "
            f"{pattern!r} — update multi_agent_debate/debate/utils.py to a "
            "current model alias (see https://docs.claude.com/en/docs/about-claude/models)."
        )


def test_default_model_looks_like_a_claude_model_id():
    assert DEFAULT_MODEL.startswith("claude-"), (
        f"DEFAULT_MODEL {DEFAULT_MODEL!r} does not look like a Claude model id"
    )
