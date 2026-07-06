# Multi-Agent Debate System

**An adversarial Proposer–Skeptic–Judge framework using the Anthropic Claude API to test whether structured inter-agent debate improves factual reliability and calibration over single-agent baselines.**

---

## Why This Matters

Scalable oversight is one of the central unsolved problems in AI alignment: how do you evaluate a highly capable AI system when the evaluator (human or model) may lack the expertise to detect subtle errors? Debate is a candidate mechanism — if two agents argue opposing positions, the flaws in the weaker argument become legible to an overseer who could not have found those flaws independently. Irving et al. (2018) formalized this intuition and showed that, under certain assumptions, a truth-telling strategy is optimal in zero-sum debate. This project is a working prototype of that idea.

The system addresses a specific failure mode of single-agent LLMs: unchallenged generation. When a model produces an answer to a hard question, it has no external signal that pushes back on its assumptions, demands evidence for its claims, or forces it to distinguish between what it knows and what it is confabulating. Introducing a dedicated Skeptic agent — one whose only job is to find the strongest flaw in the Proposer's reasoning — creates exactly that pressure. The Proposer must either defend its position with specifics or revise it. The resulting transcript gives a Judge agent (and any human reader) far more signal than a single uncontested response.

This prototype is relevant to alignment research because it is empirically evaluable. The benchmark component runs 12 claims across four categories — common misconceptions, factual questions, mathematical reasoning, and logical fallacies — and directly measures whether the debate pipeline produces more accurate and better-calibrated answers than a single-agent call. The convergence detection machinery, graph-based structural analysis, and structured Judge output are all engineering choices motivated by this measurement goal: the system is designed to produce evidence about debate quality, not just debate outputs.

---

## Architecture

```
                          ┌─────────────────────────────────┐
                          │        User Question / Claim     │
                          └────────────────┬────────────────┘
                                           │
                                           ▼
                          ┌─────────────────────────────────┐
                          │         ProposerAgent            │
                          │  (initial argument + defense)    │
                          └────────────────┬────────────────┘
                                           │  proposes
                                           ▼
                          ┌─────────────────────────────────┐
                          │         SkepticAgent             │
                          │  (challenges flaws, assumptions, │
                          │   hallucinations, weak evidence) │
                          └────────────────┬────────────────┘
                                           │  rebuts
                                           ▼
                          ┌─────────────────────────────────┐
                          │         ProposerAgent            │
                          │    (defend / revise / concede)   │
                          └────────────────┬────────────────┘
                                           │
                              ┌────────────┴────────────┐
                              │   ConvergenceDetector    │
                              │  concession regex        │
                              │  Jaccard repetition      │
                              │  stabilization check     │
                              └────────────┬────────────┘
                                           │
                          ┌────────────────▼────────────────┐
                          │  Continue N rounds OR halt early │
                          └────────────────┬────────────────┘
                                           │
                                           ▼
                          ┌─────────────────────────────────┐
                          │          JudgeAgent              │
                          │  (structured rubric, per-side    │
                          │   summaries, recency-bias check) │
                          └────────────────┬────────────────┘
                                           │
                                           ▼
                          ┌─────────────────────────────────┐
                          │          DebateResult            │
                          │  verdict · confidence · reasons  │
                          │  convergence_reason · token use  │
                          └─────────────────────────────────┘

  ┌──────────────────────────────┐    ┌──────────────────────────────┐
  │     DebateGraphBuilder       │    │       BenchmarkRunner        │
  │  NetworkX DiGraph of turns   │    │  baseline vs debate on 12    │
  │  edge types: proposes,       │    │  claims across 4 categories; │
  │  rebuts, concedes, revises,  │    │  8 aggregate metrics tracked │
  │  responds · GraphAnalyzer    │    │  · CSV + JSON output         │
  └──────────────────────────────┘    └──────────────────────────────┘
```

---

## Features

| Feature | Description |
|---|---|
| **Three-role debate agents** | `ProposerAgent`, `SkepticAgent`, `JudgeAgent` — each with dedicated system prompts that enforce their role mechanics |
| **Convergence detection** | `ConvergenceDetector` halts debates early via concession regex (8 patterns), Jaccard-similarity repetition detection (default threshold 0.40), and Proposer stabilization checks — reducing wasted API calls |
| **Structured Judge output** | `JudgeOutput` Pydantic schema with 9 fields including `verdict` enum, calibrated `confidence` float, `recency_bias_check` string, and three boolean debate-quality flags |
| **Anti-recency-bias judging** | Per-side argument summaries injected into the Judge prompt alongside the full transcript; explicit rubric and mandatory self-audit field push the Judge to evaluate the full debate arc rather than the last exchange |
| **Debate graph modeling** | `DebateGraphBuilder` constructs a NetworkX `DiGraph`; `GraphAnalyzer` produces a `GraphAnalysis` with 10 fields including rebuttal density, revision count, cycle detection, argument depth, and degree centrality |
| **Benchmark runner** | `BenchmarkRunner` evaluates baseline (single-agent Proposer call) vs debate output across a 12-item JSONL dataset; computes 7 aggregate metrics and exports CSV + JSON |
| **Mock LLM client** | `MockLLMClient` returns deterministic, role-keyed responses — full test suite runs with no API key or network access |
| **Rich CLI** | `Typer`-based CLI with two commands (`debate`, `benchmark`), `Rich` panel/table output, and flags for round count, model, graph analysis, graph visualization, and mock mode |

---

## Installation

```bash
git clone https://github.com/your-username/multi-agent-debate.git
cd multi-agent-debate
pip install -e ".[dev]"
cp .env.example .env
# Add your Anthropic API key to .env:
# ANTHROPIC_API_KEY=sk-ant-...
```

**Requirements:** Python 3.11+, `anthropic>=0.40.0`, `pydantic>=2.0.0`, `networkx>=3.3`, `pandas>=2.2.0`, `typer>=0.12.0`, `rich>=13.7.0`, `matplotlib>=3.9.0`

---

## Feature Matrix

| Feature | Flag | Description |
|---|---|---|
| Core debate | — | Proposer → Skeptic → Judge loop with convergence detection |
| Skeptic modes | `--skeptic-mode` | `general` \| `factual` \| `logic` \| `evidence` \| `safety` |
| Panel debates | `--panel` | Multiple skeptics challenge per round, e.g. `logic,evidence` |
| Human-in-the-loop | `--human-role` | Play as `proposer` or `skeptic` yourself |
| Constitutional review | `--constitutional` | Post-verdict audit against honesty, calibration, safety principles |
| Graph analysis | `--graph` | NetworkX argument graph with centrality and edge-type metrics |
| Markdown export | `--export` | Clean structured Markdown of the full debate result |
| HTML export | `--export-html` | Self-contained color-coded HTML with confidence bar and stats |
| A/B comparison | `compare` command | Same question, two configs — verdict/confidence delta report |
| Benchmark | `benchmark` command | Debate vs single-agent baseline across 12 categorized claims |
| Calibration report | `--report` | Per-category accuracy + confidence calibration bins → Markdown |
| REST API | `serve` command | FastAPI server: `POST /debate`, `POST /compare`, `GET /health` |

---

## Usage

**Basic debate:**
```bash
python -m multi_agent_debate.main debate "Your question here"
```

**With graph analysis and more rounds:**
```bash
python -m multi_agent_debate.main debate "Your question here" --rounds 4 --graph --graph-viz
```

**Specialized skeptic + constitutional review + exports:**
```bash
python -m multi_agent_debate.main debate "Does social media harm democracy?" \
  --skeptic-mode evidence \
  --constitutional \
  --export results/debate.md \
  --export-html results/debate.html
```

**Panel debate (two skeptics with different modes):**
```bash
python -m multi_agent_debate.main debate "Is nuclear energy safe?" --panel logic,safety --mock --no-save
```

**Human-in-the-loop (you play the skeptic):**
```bash
python -m multi_agent_debate.main debate "The Earth is 6,000 years old" --human-role skeptic
```

**A/B comparison (same question, two skeptic configs):**
```bash
python -m multi_agent_debate.main compare "Is consciousness an emergent property?" \
  --mode-a general --mode-b logic --mock --export results/comparison.md
```

**Benchmark with calibration report:**
```bash
python -m multi_agent_debate.main benchmark --dataset data/sample_claims.jsonl --mock --report
```

**Start the REST API server:**
```bash
pip install ".[server]"
python -m multi_agent_debate.main serve --port 8000
# POST /debate  POST /compare  GET /health
```

**Dry run (no API key needed):**
```bash
python -m multi_agent_debate.main debate "Test question" --mock --no-save
```

---

## Example Debate Transcript

```
─────────────────────── Multi-Agent Debate System ────────────────────────

╭─────────────────────── Question / Claim ──────────────────────────────╮
│ Does the Great Wall of China form a continuous unbroken wall?          │
╰────────────────────────────────────────────────────────────────────────╯

Debate Transcript

╭──────────────── PROPOSER — Round 1 ────────────────────────────────────╮
│                                                                         │
│  The Great Wall is not a single continuous structure. It is a network   │
│  of walls, fortifications, and barriers built by successive Chinese     │
│  dynasties — most notably the Qin, Han, and Ming — over roughly 2,000  │
│  years. Sections were built in different locations, with different      │
│  materials, and many were never connected. The Ming-dynasty walls       │
│  (largely what tourists visit today) are the best preserved, but even  │
│  these have significant gaps. The "continuous wall" image is a modern   │
│  myth, partly reinforced by maps that aggregate all sections.           │
│                                                                         │
╰─────────────────────────────────────────────────────────────────────────╯

╭──────────────── SKEPTIC — Round 1 ─────────────────────────────────────╮
│                                                                         │
│  The Proposer is broadly correct but overstates certainty on a key     │
│  claim: that gaps are widespread in the Ming sections. The Ming walls   │
│  in Hebei province do form a largely continuous barrier across several  │
│  hundred kilometers. If the question is specifically about those        │
│  sections, "not continuous" needs qualification. Additionally, the      │
│  claim that the myth is "modern" is unsourced — the Wall has been       │
│  misrepresented in Western sources since at least the 17th century.    │
│                                                                         │
╰─────────────────────────────────────────────────────────────────────────╯

╭──────────────── PROPOSER — Round 2 ────────────────────────────────────╮
│                                                                         │
│  I accept the Skeptic's correction on Ming continuity in Hebei — that  │
│  section is largely unbroken across ~2,000 km. I revise: the Wall is   │
│  not a single unified structure spanning China end to end; individual  │
│  dynasty-era segments vary widely in connectivity. On the "modern myth" │
│  point, the Skeptic is right that I stated that without evidence;       │
│  I will not assert a timeline I cannot support.                         │
│                                                                         │
╰─────────────────────────────────────────────────────────────────────────╯

Debate converged after round 2 (stabilization)

╭───────────────────────── Judge Verdict ────────────────────────────────╮
│                                                                         │
│  Final Answer: The Great Wall is not a single continuous barrier        │
│  spanning China. It comprises walls and fortifications from multiple    │
│  dynasties, most of which were never connected. Specific sections —     │
│  notably the Ming-era walls in Hebei — are largely continuous, but     │
│  this does not hold for the full structure.                             │
│                                                                         │
│  Verdict: SUPPORTED    Confidence: 88%                                  │
│                                                                         │
│  Key Reasons:                                                           │
│    • Historical consensus confirms multiple disconnected wall segments  │
│    • Proposer appropriately revised the Hebei claim when challenged     │
│    • Skeptic's objection improved precision without altering verdict    │
│                                                                         │
│  Recency bias check: Re-evaluated round 1 before scoring. The          │
│  Proposer's initial answer was substantively correct; the revision was  │
│  a precision improvement, not a retraction. Confidence reflects the    │
│  full debate, not only the final exchange.                              │
│                                                                         │
╰─────────────────────────────────────────────────────────────────────────╯

Tokens — input: 4,218 | output: 891
```

---

## Benchmark Methodology

The benchmark runs each claim through two pipelines in sequence and compares their outputs:

**Baseline:** A single call to `ProposerAgent` using the identical system prompt as the debate Proposer — no Skeptic, no iteration, no Judge. This isolates the effect of the debate process from the effect of the prompt itself.

**Debate:** The full `DebateOrchestrator` pipeline: Proposer initial argument → Skeptic challenge → Proposer response, repeated up to `max_rounds`, with convergence detection halting early when appropriate, followed by `JudgeAgent` evaluation.

**Dataset:** 12 claims in `data/sample_claims.jsonl` spanning four categories:
- `common_misconception` (4 claims) — widely believed falsehoods (Great Wall continuity, 10% brain myth, Einstein failing math, Coriolis draining effect)
- `factual` (4 claims) — precise factual questions (boiling point pressure dependence, speed of light in materials, tomato botanical classification, lightning as static discharge)
- `math` (2 claims) — mathematical reasoning (gambler's fallacy, zero as non-positive)
- `logical` (2 claims) — logical principles (correlation vs causation, halting problem undecidability)

**Correctness check:** `_answers_match` computes keyword overlap between an answer and the ground truth string. An answer is scored correct if the token overlap ratio exceeds 0.35 (i.e., more than 35% of ground-truth key terms appear in the answer). This is an intentionally coarse heuristic — appropriate for a research prototype benchmarking answer quality without requiring embedding similarity or LLM-as-judge scoring.

**Metrics tracked (7 aggregate + 1 per-result):**

| Metric | Description |
|---|---|
| `baseline_accuracy` | Fraction of examples where the single-agent answer meets the correctness threshold |
| `debate_accuracy` | Fraction of examples where the Judge's final answer meets the threshold |
| `debate_improvement_rate` | Fraction where debate was correct and baseline was not |
| `avg_debate_confidence` | Mean Judge `confidence` score across all examples |
| `avg_rounds_used` | Mean rounds before convergence or max-rounds halt |
| `convergence_rate` | Fraction of debates that converged early (not max-rounds) |
| `avg_total_tokens` | Mean total tokens (baseline + debate) per example |
| `debate_improved` | Per-result boolean: debate correct and baseline wrong |

Results are saved as both CSV and JSON with a timestamp-stamped filename under `results/`.

---

## Graph Analysis

Every debate transcript is optionally modeled as a directed graph using NetworkX. Each turn is a node; edges connect sequential turns. Edge type is inferred from the roles of the source and target turns:

| Edge Type | Source → Target | Semantics |
|---|---|---|
| `proposes` | Proposer → Skeptic | Proposer makes a claim the Skeptic will evaluate |
| `rebuts` | Skeptic → Proposer | Skeptic raises an objection to the preceding Proposer turn |
| `concedes` | — → Proposer | Proposer's next turn contains an explicit concession (regex-detected) |
| `revises` | — → Proposer | Proposer responds without a detectable concession |
| `responds` | Any → Any (fallback) | Turn-to-turn connection not matching the above patterns |

`GraphAnalyzer.analyze()` returns a `GraphAnalysis` Pydantic model with 10 fields:

| Field | Description |
|---|---|
| `num_turns` | Total nodes in the debate graph |
| `num_claims` | Proposer turn count |
| `num_rebuttals` | Count of `rebuts` edges |
| `num_concessions` | Proposer turns with detected concession language |
| `num_revisions` | Proposer turns where token overlap with the prior Proposer turn is between 0.10 and 0.85 (substantive change, not a complete rewrite) |
| `has_cycles` | `True` if the graph is not a DAG — indicates argument loops |
| `proposer_revisions_caused_by_skeptic` | Alias for `num_revisions` (all Proposer revisions follow a Skeptic challenge by construction) |
| `argument_depth` | Total turn count — a proxy for debate length |
| `centrality_scores` | NetworkX degree centrality for each node — identifies structurally pivotal turns |
| `edge_type_counts` | Dictionary of edge type → count across the full graph |

The `DebateGraphBuilder.visualize()` method saves a labeled PNG (requires `matplotlib`) with nodes colored by role: green for Proposer, red for Skeptic, blue for Judge.

---

## Key Engineering Challenges

### Convergence Detection

Naive debate loops either run all N rounds regardless of content (wasteful) or require a separate LLM call to check convergence (expensive and adds latency). `ConvergenceDetector` uses three purely heuristic, zero-cost checks applied after each turn:

1. **Concession regex:** Eight patterns compiled into a single `re.compile` expression (e.g., `r"\bi concede\b"`, `r"\byou(?:'ve| have) convinced me\b"`) are matched against the latest Proposer turn. A match signals that the Proposer has acknowledged a valid flaw — the debate has served its purpose.

2. **Jaccard repetition:** For each new Skeptic turn, `_token_overlap` computes `|A ∩ B| / |A ∪ B|` against all prior Skeptic turns after lowercasing and stripping punctuation. If overlap exceeds the threshold (default 0.40), the Skeptic is recycling objections — a signal that no new critique is available.

3. **Stabilization:** The last two Proposer turns are compared using the same Jaccard metric. If the Proposer's argument has stopped changing (overlap ≥ 0.40), continued rounds are unlikely to refine it further.

Checks are applied in priority order — concession first, then Skeptic repetition, then stabilization — and the first matching check sets the `ConvergenceReason` enum value stored in `DebateResult`.

### Judge Recency Bias

The last turn in any transcript is the most salient to a language model evaluating the debate. A naive Judge prompt risks simply agreeing with whoever spoke last. Three countermeasures are implemented:

1. **Per-side summaries:** `_summarize_side()` extracts the first sentence of each Proposer turn and each Skeptic turn independently and injects both bullet lists into the Judge prompt alongside the full transcript. This forces the Judge to process the arc of each side's argument rather than treating the full transcript as a single undifferentiated context.

2. **Explicit `recency_bias_check` schema field:** The `JudgeOutput` Pydantic schema includes a required string field for the Judge's self-audit. The Judge system prompt mandates that the Judge explicitly ask whether it would reach the same verdict reading the transcript in reverse order before finalizing its confidence score. The audit result is stored in the output and surfaced in the CLI display.

3. **Structured rubric:** The Judge system prompt defines four explicit evaluation dimensions — initial answer quality, Skeptic validity, Proposer responsiveness, and convergence quality — that require the Judge to score the full debate arc before forming a verdict. This discourages collapsing the evaluation onto the final exchange.

### Debate Graph Modeling

Modeling debate as a graph (rather than a flat list of turns) enables structural queries that are invisible in transcript form: Does rebuttal density correlate with answer improvement? Does a high revision count under a low concession count indicate the Skeptic is forcing genuine updates without the Proposer explicitly acknowledging them? Are there argument cycles that reveal circular reasoning?

`DebateGraphBuilder` constructs a `networkx.DiGraph` where each node stores role, round number, and a 120-character content snippet. Edges connect sequential turns and are typed by `_edge_type()`, which checks the source and target roles and applies concession detection to classify `concedes` vs `revises` edges. `GraphAnalyzer` computes `nx.degree_centrality` to identify which turns are structurally pivotal (high in-degree or out-degree relative to the graph), and `nx.is_directed_acyclic_graph` to flag debates where the argument structure loops back — a potential indicator of unresolved circular reasoning.

### Structured Judge Output

The Judge must return a 9-field JSON object. Raw LLM output frequently wraps JSON in markdown fences (` ```json ... ``` `), includes preamble text, or produces malformed JSON under adversarial prompt conditions. `_extract_json()` in `judge.py` handles this in two passes: direct `json.loads` first, then a regex strip of markdown fences followed by a second parse attempt. If both passes fail, a fallback `JudgeOutput` is constructed with `verdict=VerdictType.UNCERTAIN`, `confidence=0.0`, and a truncated raw response in `final_answer` — ensuring the system never raises an unhandled exception on a parse failure. The `VerdictType` enum (`supported`, `refuted`, `uncertain`) and the Pydantic `Field(ge=0.0, le=1.0)` constraint on `confidence` mean that any successfully parsed output is structurally valid before it reaches downstream code.

---

## Running Tests

```bash
pytest tests/ -v
```

```bash
pytest tests/ -v --cov=src
```

All tests use `MockLLMClient` — no API key required. 72 tests across 11 modules:

| Module | Tests | Covers |
|---|---|---|
| `test_convergence` | 6 | Concession detection, Jaccard repetition, stabilization, `should_stop` |
| `test_graph` | 4 | Graph construction, edge types, `GraphAnalysis` fields, DAG property |
| `test_schemas` | 4 | Pydantic validation, JSON round-trip, field bounds |
| `test_orchestrator` | 3 | End-to-end debate, max-rounds reason, graph-disabled path |
| `test_skeptic_modes` | 5 | All 5 modes resolve distinct prompts, `ValueError` on unknown mode |
| `test_export` | 8 | All 6 Markdown sections, graph conditional, HTML escaping |
| `test_calibration` | 6 | Per-category stats, bin count, accuracy validity, overall accuracy |
| `test_panel` | 5 | Panel flag, multiple skeptic turns per round, backward compat |
| `test_human_loop` | 5 | Human proposer, human skeptic, `human_role` field, AI fallback |
| `test_html_export` | 6 | DOCTYPE, question, role classes, verdict badge, graph table, XSS escape |
| `test_constitutional` | 6 | Review present/absent, `overall_safe`, `principles_checked`, violations/warnings |
| `test_compare` | 6 | `DebateComparison` type, both results populated, `verdict_match`, Markdown output |
| `test_server` | 6 | `/health`, `POST /debate`, verdict field, skeptic mode, `POST /compare`, `verdict_match` |

---

## Constitutional Review

The `--constitutional` flag adds a post-verdict audit layer inspired by Anthropic's Constitutional AI work. After the Judge delivers its verdict, a `ConstitutionalAgent` evaluates the output against four principles:

1. **Honesty** — No false or misleading claims; verified facts distinguished from inference
2. **Calibration** — Expressed confidence matches the evidence strength in the transcript
3. **Safety** — No harmful recommendations, dangerous misconceptions, or risk-amplifying information
4. **Uncertainty acknowledgment** — Genuine unknowns flagged rather than papered over

The agent returns a `ConstitutionalReview` Pydantic model with `violations`, `warnings`, `overall_safe`, and an optional `revised_answer` when a violation makes the original answer actively misleading. PASS/FAIL is displayed in the CLI after the judge panel and is embedded in Markdown exports.

---

## REST API

```bash
pip install ".[server]"
python -m multi_agent_debate.main serve --port 8000
```

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/debate` | Run a debate; body: `{"question": "...", "skeptic_mode": "logic", "mock": true}` |
| `POST` | `/compare` | A/B comparison; body: `{"question": "...", "mode_a": "general", "mode_b": "evidence", "mock": true}` |

Both `POST` endpoints return the full Pydantic model serialized as JSON. The `mock: true` default keeps the API usable without an API key.

---

## Resume Framing

- Built a multi-agent LLM debate framework (Proposer–Skeptic–Judge) implementing Irving et al.'s scalable oversight via debate proposal; measured accuracy and calibration improvement vs. single-agent baseline across a 12-item benchmark spanning four question categories.

- Designed convergence detection using heuristic concession recognition, Jaccard-similarity repetition detection, and Proposer stabilization checks — halting debates early at zero LLM cost when continued rounds are unlikely to improve the answer.

- Implemented anti-recency-bias judging via per-side argument summaries, an explicit `recency_bias_check` schema field, and a structured four-dimension rubric — reducing the risk of the Judge collapsing its evaluation onto the final exchange.

- Extended the system with a Constitutional Review agent (post-verdict honesty/calibration/safety audit), multi-skeptic panel debates, human-in-the-loop role substitution, A/B configuration comparison, per-category calibration scoring, and a FastAPI REST server — 72 tests, all offline via `MockLLMClient`.

- Modeled debate transcripts as directed argument graphs (NetworkX) enabling structural analysis of rebuttal density, revision causation, cycle detection, and argument depth as proxies for debate quality beyond accuracy.

---

## References

- Irving, G., Christiano, P., & Amodei, D. (2018). [AI safety via debate](https://arxiv.org/abs/1805.00899). *arXiv:1805.00899*.
- Bowman, S. R., et al. (2022). [Measuring Progress on Scalable Oversight for Large Language Models](https://arxiv.org/abs/2211.03540). *arXiv:2211.03540*.
- Bai, Y., et al. (2022). [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073). *Anthropic. arXiv:2212.08073*.
