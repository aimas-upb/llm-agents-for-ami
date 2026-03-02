# Reproducing the Evaluation

This document explains how to reproduce the two evaluation experiments from the paper (Section 6).

---

## Experiment 1 — Explicit Request Understanding and BT Planning (Section 6.1)

**Paper reference:** Table 1 (overall performance across 400 HomeBench test cases, four LLM models).

**What it tests:** Correct parsing of explicit user requests into structured intents, and BT plan generation for single and compound, feasible and infeasible requests, using the HomeBench benchmark.

**Setup:**
```bash
export OPENAI_API_KEY="<your-key>"
export OPENAI_MODEL="gpt-4o-mini"    # or gpt-4o, gpt-5-mini, gpt-5-nano
```

**Run:**
```bash
pytest tests/e2e/test_demo_homebench.py -v -m e2e
```

**What it does:**
- Loads HomeBench Home 17 environment fixture (study room: light, climate, entertainment)
- Feeds 400 test utterances (single/compound, feasible/infeasible)
- Each utterance is parsed by UserAssistant via `INTENT_EXTRACTION_SYSTEM_PROMPT`
- Structured intents are routed through the dual-path BT builder (Algorithm 1)
- `set` intents → direct BT leaf nodes; `modify` intents → LLM BT generation
- Evaluates: full success, partial success, impossible detection, precision, recall, F1, latency

**Expected results (paper Table 1):**
| Model | Full succ. | Full+Partial | Imp. det. | F1 |
|---|---|---|---|---|
| GPT-4o | 76.8% | 95.8% | 99.3% | 0.929 |
| GPT-4o-mini | 80.5% | 96.3% | 99.6% | 0.941 |
| GPT-5-mini | 63.8% | 96.3% | 100% | 0.953 |
| GPT-5-nano | 64.0% | 88.3% | 100% | 0.872 |

---

## Experiment 2 — Experience Reuse and Cross-Environment Transfer (Section 6.2)

**Paper reference:** Table 2 (three phases: cold start, warm start, cross-environment transfer).

**What it tests:** End-to-end dual-path planning with signifier reuse (fast-path vs. LLM path), and cross-environment transfer of signifiers from Lab308 to HomeBench Home 17.

### Phase A — Cold Start (no signifiers)

```bash
export OPENAI_API_KEY="<your-key>"
export OPENAI_MODEL="gpt-4o-mini"

# Run Lab308 cold-start scenario (17 requests, no prior signifiers)
pytest tests/e2e/test_demo_lab308.py -v -m e2e -k "cold"
```

All 14 actionable requests go through LLM planning path. Produces 18 LLM calls, ~2449 ms avg latency.

### Phase B — Warm Start (signifiers from Phase A)

```bash
# Signifiers from Phase A are persisted in the SQLite store (data/signifiers.db)
# Run again -- 12/14 requests should hit the fast path
pytest tests/e2e/test_demo_lab308.py -v -m e2e -k "warm"
```

Expected: 12 fast-path hits, 5 LLM calls total, ~337 ms avg latency, 7.3x speedup.

### Phase C — Cross-Environment Transfer

```bash
# Lab308 signifiers serve as community hints for HomeBench Home 17
pytest tests/e2e/test_cross_env_sharing.py -v -m e2e
```

Expected: All 6 actionable HomeBench requests succeed using Lab308 hints. BTs correctly reference HomeBench affordance URLs. 27% latency reduction vs. Phase A.

**Expected results (paper Table 2):**
| Phase | Req. | Fast | LLM | Imp. | Succ. | Avg Lat. | LLM Calls |
|---|---|---|---|---|---|---|---|
| A (Cold) | 17 | 0 | 14 | 3 | 100% | 2449 ms | 18 |
| B (Warm) | 17 | 12 | 2 | 3 | 100% | 337 ms | 5 |
| C (Cross) | 7 | 0 | 6 | 1 | 100% | 1789 ms | 7 |

---

## Full End-to-End Demo (paper demo video)

For the interactive demo corresponding to the companion demo paper, see [docs/demo.md](demo.md).

The demo covers:
- Sequence 3: explicit + implicit request (cold start, signifier recording)
- Sequence 4: implicit request with signifier reuse (fast path)

---

## Unit Tests

Unit tests for individual components can be run without any external services:

```bash
# BT schema validation
pytest tests/unit/test_bt_schema.py -v

# BT IR executor
pytest tests/unit/test_ir_executor.py -v

# Signifier bridge (BT execution -> signifier extraction)
pytest tests/unit/test_signifier_bridge.py -v

# UserAssistant intent extraction and state machine
pytest tests/unit/test_user_assistant_behaviours.py -v
pytest tests/unit/test_user_assistant_models.py -v

# Intent type validation
pytest tests/unit/test_intent_type_validation.py -v

# All unit tests
pytest tests/unit/ -v
```

## Yggdrasil Integration Tests

Requires a running Yggdrasil instance at `http://localhost:8080/`:

```bash
python tests/check_localhost.py            # verify HMAS platform
pytest tests/environment/ -v -s           # integration tests
```

See [docs/testing.md](testing.md) for HMAS ontology format requirements and RDF content samples.
