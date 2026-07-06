"""Benchmark correctness grading — heuristic (default) and opt-in LLM.

The heuristic grader (`answers_match`) is a zero-cost, offline, polarity-aware
keyword check. It is fast and free but is a crude proxy for factual
correctness: it cannot tell that two differently-worded answers reach the
same conclusion, and it can be fooled by verbose answers that reuse the
question's vocabulary while asserting the wrong thing.

The LLM grader (`grade_with_llm`) is an entailment-style judge: it asks a
model whether a candidate answer is factually consistent with a ground-truth
reference. It is far more semantically aware than the heuristic, but it
spends an API call per grading decision — so it is strictly opt-in, wired up
via `BenchmarkRunner(grader="llm")` / `debate benchmark --grader llm`, never
the default.
"""
from __future__ import annotations

import json
import re

from multi_agent_debate.debate.schemas import GradeResult, GraderType
from multi_agent_debate.debate.utils import BaseLLMClient

# ---------------------------------------------------------------------------
# JSON extraction (shared with judge.py — see extract_json below)
# ---------------------------------------------------------------------------


def extract_json(text: str) -> dict:
    """Extract JSON from model output, stripping markdown fences if present.

    Three-pass strategy, in order of preference:
    1. Direct `json.loads` — the common case for a well-behaved model.
    2. Strip markdown code fences (```json ... ```) and retry.
    3. Last resort: the model wrapped valid JSON in prose ("Here is the
       verdict: {...}"). Extract the first balanced-looking brace region.

    Raises `json.JSONDecodeError` if no JSON object can be recovered.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise json.JSONDecodeError("No JSON object found in grader output", text, 0)


# ---------------------------------------------------------------------------
# Heuristic grader (moved from benchmark.py — see answers_match below)
# ---------------------------------------------------------------------------

_YES_WORDS = {"yes", "yeah", "yep", "correct", "true", "indeed", "affirmative"}
_NO_WORDS = {"no", "nope", "incorrect", "false", "negative"}


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _leading_polarity(text: str) -> str | None:
    """Return 'yes', 'no', or None based on the leading token of normalized text."""
    words = _normalize(text).split()
    if not words:
        return None
    first = words[0]
    if first in _YES_WORDS:
        return "yes"
    if first in _NO_WORDS:
        return "no"
    return None


def _token_overlap_ratio(answer: str, ground_truth: str) -> float:
    a_tokens = set(_normalize(answer).split())
    g_tokens = set(_normalize(ground_truth).split())
    return len(a_tokens & g_tokens) / max(len(g_tokens), 1)


def answers_match(answer: str, ground_truth: str) -> bool:
    """
    Polarity-aware correctness check (the default, zero-cost grader).

    Ground truths in this dataset conventionally open with an explicit
    "Yes."/"No." verdict. Raw keyword overlap alone has a length bias: a
    verbose *wrong* answer that reuses the question's topic vocabulary can
    out-overlap a terse *correct* one-word answer. To fix this, extract an
    explicit yes/no polarity from the leading token of both the answer and
    the ground truth when present. If both state a polarity, agreement
    between them is the primary signal — a stated "Yes" cannot match a
    "No." truth no matter how much vocabulary it shares. Token overlap is
    used as a secondary signal only when at least one side has no explicit
    leading polarity to compare.
    """
    answer_polarity = _leading_polarity(answer)
    truth_polarity = _leading_polarity(ground_truth)

    if answer_polarity is not None and truth_polarity is not None:
        return answer_polarity == truth_polarity

    return _token_overlap_ratio(answer, ground_truth) > 0.35


# ---------------------------------------------------------------------------
# LLM grader (opt-in)
# ---------------------------------------------------------------------------

# Rough cost-guardrail constants (see BenchmarkRunner / CLI --grader llm).
# Grading is run once for the baseline answer and once for the debate's final
# answer, so each benchmark example costs two grader calls.
GRADER_CALLS_PER_EXAMPLE = 2
# A grader call's system+user prompt plus a ~300-token response is roughly
# this many tokens total. This is a deliberately rough estimate for the
# upfront cost guardrail, not a measured average.
ROUGH_TOKENS_PER_GRADER_CALL = 700

GRADER_MAX_TOKENS = 300

GRADER_SYSTEM_PROMPT = """\
You are an entailment-style grader for a factual-claims benchmark. You will be
shown a question, a ground-truth reference answer, and a candidate answer
produced by a separate system. Your job is to judge whether the candidate
answer is factually consistent with — and does not contradict — the ground
truth. This is not a keyword-overlap or wording-similarity check: a candidate
that reaches the same substantive conclusion in different words is a match; a
candidate that reaches the opposite or a materially different conclusion is
not, no matter how much vocabulary it shares with the ground truth.

## How to judge

- Focus on the core claim/verdict, not phrasing, verbosity, or style.
- If the ground truth states an explicit polarity (e.g. "Yes, ..." / "No,
  ..."), the candidate must agree with that polarity to match.
- A candidate that hedges into genuine ambiguity where the ground truth is
  unambiguous is not a match.
- A candidate that is correct on the core claim but omits secondary detail
  present in the ground truth is still a match.

## Output format — strict

Respond with a single JSON object and nothing else. No preamble, no
explanation outside the object, no markdown fences. The object must match
this schema exactly:

{{
  "match": <true | false>,
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one or two sentences explaining the verdict>"
}}

Do not output any text outside the JSON object.
"""

GRADER_USER_TEMPLATE = """\
## Question

{question}

## Ground truth (reference answer)

{ground_truth}

## Candidate answer to grade

{candidate_answer}

---

Judge whether the candidate answer is factually consistent with the ground
truth. Respond with the JSON object per your instructions.
"""


def grade_with_llm(
    client: BaseLLMClient,
    question: str,
    candidate_answer: str,
    ground_truth: str,
) -> GradeResult:
    """
    Ask an LLM to judge whether `candidate_answer` is factually consistent
    with / entails `ground_truth`, given `question` for context.

    Deterministic settings (temperature 0, modest max_tokens) are the
    caller's responsibility — pass a `client` already configured that way
    (see `BenchmarkRunner`'s dedicated grader client).

    On JSON parse failure, falls back to the heuristic grader
    (`answers_match`) and marks `grader=GraderType.HEURISTIC_FALLBACK` so
    callers can distinguish "the LLM graded this" from "the LLM grader
    misbehaved and we fell back."
    """
    user_prompt = GRADER_USER_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        candidate_answer=candidate_answer,
    )
    raw, inp, out = client.call(GRADER_SYSTEM_PROMPT, user_prompt)

    try:
        data = extract_json(raw)
        return GradeResult(
            match=bool(data["match"]),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=str(data.get("reasoning", "")),
            grader=GraderType.LLM,
            input_tokens=inp,
            output_tokens=out,
        )
    except Exception as exc:  # noqa: BLE001 - any parse/shape failure triggers fallback
        fallback_match = answers_match(candidate_answer, ground_truth)
        return GradeResult(
            match=fallback_match,
            confidence=0.0,
            reasoning=(
                "LLM grader output could not be parsed as a valid grade "
                f"({type(exc).__name__}: {exc}); fell back to the heuristic "
                "polarity/keyword-overlap grader."
            ),
            grader=GraderType.HEURISTIC_FALLBACK,
            input_tokens=inp,
            output_tokens=out,
        )
