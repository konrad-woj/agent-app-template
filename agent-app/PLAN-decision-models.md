# Plan: Decision Models (CPU) for Routing, Evals and PHI — with LLM Fallback

Status: **research complete, bake-off not started.** Nothing in `app/` or `evals/` has changed yet.

## Table of Contents

- [Context](#context)
- [Goals and Non-Goals](#goals-and-non-goals)
- [Core Pattern: Decision Model First, LLM on Low Confidence](#core-pattern-decision-model-first-llm-on-low-confidence)
- [Serving Under 10–50 Concurrent Requests](#serving-under-1050-concurrent-requests)
- [Track A — Routing](#track-a--routing)
- [Track B — Evals](#track-b--evals)
- [Track C — PHI (input and output)](#track-c--phi-input-and-output)
- [Candidate Models](#candidate-models)
- [Fine-Tuning Strategy (little or no labeled data)](#fine-tuning-strategy-little-or-no-labeled-data)
- [Bake-Off Design](#bake-off-design)
- [Phases](#phases)
- [EKS Notes (back of the head)](#eks-notes-back-of-the-head)
- [Risks and Open Questions](#risks-and-open-questions)
- [Evidence and Sources](#evidence-and-sources)

---

## Context

`agent-app` is a healthcare-adjacent LangGraph research assistant. Today:

| Concern | Current implementation | Weakness |
|---|---|---|
| Guard | `GLiGuardClient` (`app/guards/gliguard.py`), `fastino/gliguard-LLMGuardrails-300M`, one shared model, `asyncio.Semaphore(4)` + `to_thread`, one text per call | No batching; calls queue under load; the guard runs 4 times per request (`input_guard`, `planner`, `resume_guard`, `output_guard`) |
| Input topic check | `input_guard` layer 3: LLM call (~300 ms) | Slow and nondeterministic for a closed-set decision |
| Routing | None. Fixed graph; only `NodeLLMConfig` picks a model per node | No intent, tool or model-tier routing |
| PII redaction | `output_guard`: six labels (email, phone, card, SSN, API key, IP) | No HIPAA identifier categories (names, dates, MRN, ...); no input-side PHI handling |
| Evals | `evals/evaluators.py`: free-form LLM rubric judge + deterministic trace assertions | Judge is nondeterministic, slow, and sees PHI-like data |

**Demo environment:** all local, Apple Silicon Mac, CPU inference. Plan for **10–50 concurrent requests** to the decision model. Production target is AWS EKS; the design must carry over (see [EKS Notes](#eks-notes-back-of-the-head)).

## Goals and Non-Goals

**Goals**
- Reduce latency and run-to-run nondeterminism of three decision types without losing quality: **routing**, **evals**, **PHI detection/redaction**.
- Decision model answers first; **fall back to the LLM when confidence is low**.
- Choose models on measured results on our own (synthetic + clinician-reviewed) healthcare data, not vendor numbers.
- Keep each track independently evaluable and independently shippable.

**Non-goals (for now)**
- No EKS deployment, autoscaling or GPU serving.
- No changes to `app/` before the bake-off picks winners (feature-flagged wiring comes in Phase 4).
- Not a general-purpose guardrail overhaul; harm-category coverage stays with the existing GLiGuard unless the bake-off says otherwise.

## Core Pattern: Decision Model First, LLM on Low Confidence

One shared pattern for all three tracks, implemented once:

```
request ─► cache lookup (hash(text) + label-set version + model version)
        ─► decision model (CPU, batched)
        ─► confidence gate
              ├─ accept  ─► decision (deterministic path)
              └─ reject  ─► LLM fallback (temperature 0, structured output) ─► decision
        ─► log {track, label, confidence, path=decision|fallback, latencies}
```

**Design rules**
1. **One interface, three tracks.** A `DecisionClient` protocol (`classify`, `extract_spans`) with a result type carrying `label`, `confidence`, `abstained`, `spans`, `model_version`, `label_set_version`. Concrete clients wrap each candidate model; tests inject a mock, mirroring `GLiGuardClient` (lazy `load()`, no import-time side effects).
2. **Confidence gate per track and per label**, not a global threshold. Thresholds are *calibrated* on the gold set to hit a target (see below), stored in config, and versioned.
3. **What "confidence" means must be verified per model in Phase 0.** Vela exposes probabilities for choice/score questions. The Decide model card (as relayed by third parties) documents a label string or labels above `cls_threshold` but *not* a full probability vector — if so, abstention = "no label above threshold" or a margin we compute ourselves. Do not assume; test.
4. **Fallback is asymmetric by track.**
   - Routing: fall back to the existing LLM prompt (today's topic check).
   - Evals: fall back to the existing LLM rubric judge.
   - PHI: **fail closed.** Low confidence on a possible PHI span means redact (or block), and the LLM is only used to *downgrade* a redaction when it is a local/in-VPC model. Recall targets dominate precision.
5. **Determinism boundary.** The decision-model path must give identical output for identical input across repeats, thread counts and batch compositions. The fallback path is only as deterministic as the LLM (temperature 0, structured output, response caching). Report both separately and report the **fallback rate**, because it sets the effective determinism and latency of the cascade.
6. **Fallback outputs are not labels.** LLM answers on fallback cases feed a *review queue*; only clinician-reviewed items enter training data (avoids training the small model on LLM noise).

**Cascade metrics (all tracks):** selective accuracy (accuracy on accepted items), end-to-end accuracy vs LLM-only, fallback rate, expected latency = `decision_ms + fallback_rate × llm_ms`, and cost per 1,000 requests.

## Serving Under 10–50 Concurrent Requests

Findings that drive the design:
- A single GLiNER forward pass keeps only 2–3 cores busy; concurrent batches gave 1.84x at 5 batches, throughput peaked at about half of torch's thread count, and each extra batch cost ~1.7 GB (one project's test, not documented behavior).
- Thread safety of the Python `gliner2` batch APIs is empirical, not documented — verify with identical-output tests.
- Some community ports of Decide (GGUF, LiteRT) do not batch or cap input length (198 / 512 tokens). Use the Python package or ONNX for the bake-off.
- The current guard (`Semaphore(4)`, batch size 1) makes a burst of 50 wait roughly `50 × service_time / 4` (back-of-envelope, e.g. ~1–2 s at 100–170 ms).
- No public benchmark compares torch-CPU, MPS, CoreML and ONNX for small encoders on Apple Silicon. We measure.

**Serving variants to compare** (same models, same load):

| Variant | Description |
|---|---|
| S1 baseline | Current design: semaphore + `to_thread`, batch size 1 |
| S2 micro-batcher | Async queue collects requests for 5–10 ms (or N items), one padded batch call, fan results back to waiters; length-bucketing; max batch cap |
| S3 worker pool | 2–3 worker processes, each with its own model copy and pinned threads, behind S2 |

**Knobs:** torch thread count, batch cap, batch window, backend (torch CPU / MPS / ONNX Runtime CPU / ONNX+CoreML EP / CoreML port), INT8 where supported (re-check accuracy after quantizing), warm-up, input truncation.

**Reduce calls, not just latency:** merge tasks into one encoder pass where the model supports it (GLiNER2 entities + classification in one schema; Vela multi-signal questions in one call). The four guard call sites per request make this the biggest lever.

**Load generator:** open-loop arrivals at concurrency 1 / 10 / 25 / 50, bursty and steady, mixed text lengths. Report p50/p95/p99, queue wait, throughput, RSS memory, CPU utilization. Pin everything else on the host (vendor benchmarks needed a clean-host rerun because shared hosts skewed results).

## Track A — Routing

**Decisions to route**

| Decision | Labels (examples) | Fallback |
|---|---|---|
| Scope / topic (replaces `input_guard` layer 3 LLM check) | `in_scope_research`, `out_of_scope`, `clinical_advice_request`, `unclear` | Existing topic-check LLM prompt |
| Tool / sub-agent selection before ReAct | `web_search`, `fetch_url`, `fact_check`, `none` | LLM tool planner |
| Model tier per node | `fast`, `strong`, `strong+thinking` | Default to `strong` |

**Evidence and caveats**
- Decide reports 75.3% (support intent) and 64.3% (banking intent) on Fastino's own suite; an independent DecisionBench run reports very different numbers on a different benchmark. Treat as unverified.
- vLLM-SR's Kai→Vega cascade went 65.8% → 82.3% (large model alone 87.9%) on 231 public items with ~54% estimated cost savings (assumes equal per-call cost). This supports the cascade pattern, not specific numbers.
- Vela's router decision schema accepts only choice / yes-no / score questions; set and span questions are rejected there ([issue #4638]). We use Vela directly, not through that router.
- One post reports label wording changed Decide's answers → label sets are versioned artifacts and need paraphrase-robustness tests.

**Evaluation:** macro-F1 and per-label recall on a healthcare routing set (synthetic + reviewed); abstain rate; selective accuracy; repeat/thread/batch determinism; paraphrase robustness (3–5 rewrites of each label description); latency by serving variant.

**Integration point (Phase 4, flagged):** `input_guard` layer 3; a new `route` step before `react_researcher`; `NodeLLMConfig` selection.

## Track B — Evals

**Where decision models fit.** Short, typed, bounded checks — not open-ended critique:
- "Is this claim supported by this snippet?" (verification results)
- "Did the answer refuse / hedge appropriately?" / "Does it contain a clinical-advice statement?"
- Rubric sub-questions with a small ordinal scale (Phase 5b's `CriterionVerdict` scores) answered as `score` questions.

The LLM judge stays for open-ended rubric scoring and for every low-confidence case. Deterministic trace assertions (`evals/trace_assertions.py`) are unchanged.

**Constraints:** encoder context is limited (Decide ports 198–512 tokens; Vela 0.3B 8,192). Long answer + evidence pairs will be chunked per claim; truncation must be measured, not assumed harmless. Vela's hallucination result (0.774 example-F1 on RAGTruth) is for the 9B; no 0.3B figure was found.

**Reference:** Jev (hosted, proprietary, no weights) showed far lower judge variance in LangChain's narrow test. It cannot see raw PHI; allowed only as an *optional* reference on synthetic/de-identified data. Not a deployment candidate.

**Evaluation:** agreement (Cohen's κ / accuracy) with a clinician-reviewed gold set; **run-to-run variance over 100 repeats** (decision path vs LLM judge); selective accuracy; fallback rate; wall-clock for a 200-item eval run at concurrency 1/10/25/50; cost.

**Integration point (Phase 4, flagged):** `evals/evaluators.py` — a `decision_judge` evaluator beside `make_quality_score_evaluator`, same `Evaluation` return type, so Langfuse plumbing is unchanged.

## Track C — PHI (input and output)

**Scope (as decided):** detect/redact PHI in **user input** and in **agent output**. PHI-aware routing and trace/eval-data de-identification are out of scope for this plan (noted as follow-ups).

**What the evidence says**
- Generic PII models degrade badly on clinical text: GLiNER-style models ~0.41 F1 on clinical datasets (vendor posts); six open taggers at recall 0.18–0.48 on i2b2 2014 (TIDE 2.0 preprint). A GLiNER fine-tuned on one site's psychiatric EHRs reached 0.98 F1 — fine-tuning is the lever, but that result is single-site.
- A 125M DeBERTa PHI detector (blog post): ~29 ms on CPU, good on names/places/dates, poor on structured IDs (MRN, license numbers). → deterministic regex for structured identifiers in front of the model.
- Vela 0.3B reports 0.995 F1 on **short-text** PII; this is not clinical text.
- The current `output_guard` handles six labels only.

**Pipeline**
1. Deterministic layer: regex/checksum patterns for structured IDs (SSN, phone, email, MRN-like formats, dates, URLs, IPs, device/plate-like IDs).
2. Decision model span extraction over the HIPAA Safe Harbor categories (names, geography smaller than state, dates except year, ages > 89, phone, fax, email, SSN, MRN, health plan numbers, account numbers, license numbers, vehicle/device IDs, URLs, IPs, biometric IDs, photos [N/A to text], other unique IDs).
3. Confidence gate with **asymmetric thresholds (recall-first)**; spans below the accept threshold but above a floor go to a local LLM fallback only to *remove* a redaction; spans below the floor are redacted by default.
4. Redact with typed placeholders (reuse `redact()` / `[REDACTED:<type>]`), keeping offsets for evaluation.

**Controls in the bake-off:** existing `GLiGuard` (PII labels), Presidio, and one dedicated PII model (e.g. NVIDIA `gliner-PII` or Knowledgator `gliner-pii`). Controls tell us whether the candidates beat what is already available.

**Evaluation:** span-level (strict and relaxed) precision/recall/F1 **per HIPAA category**, recall first; leakage rate = fraction of documents with ≥1 missed identifier; over-redaction rate on clinically important non-PHI tokens (drug names, doses, lab values); latency at concurrency levels. Primary acceptance criterion is recall per category, not average F1.

**Data (cold start):** synthetic clinical notes and chat-style queries generated with fake identifiers (including adversarial formats: names that are common words, partial dates, MRNs embedded in sentences, mixed languages), plus a clinician-reviewed gold set. No real PHI enters the repo or any hosted service.

## Candidate Models

| Model | Size | License | Role in bake-off | Known risk |
|---|---|---|---|---|
| GLiNER2.5 small (74M) / base (194M) | boundary models | Apache-2.0 | Fast classification/extraction candidates | Community benchmarks are small; weaker on clinical text out of the box |
| GLiNER2.5-Decide (340M; 1B and multilingual siblings exist) | encoder | Apache-2.0 | Routing/classification candidate; LoRA adapters 2–10 MB | 167 ms p50 at batch 1 on 48 vCPUs (vendor); label-wording sensitivity |
| Vela-2.0-0.3B (vLLM-SR / KR Labs) | 0.3B ModernBERT encoder | Apache-2.0 | Multi-signal (safety, PII, domain, hallucination) with spans | ~79–128 ms CPU p50 vs ~16 ms for Vela 1.0 (~560-token schema per request); Vela 0.8B on CPU ~2.8 s → not used |
| GLiFormer (Knowledgator) | base → large | Apache-2.0 | Control for structuring/classification | 547 ms CPU for structuring (vendor) — expected to miss the latency budget |
| fastino/gliguard-LLMGuardrails-300M | 300M | Apache-2.0 | Current guard; baseline for guard + PII | Batch size 1 in the current wrapper |
| Presidio, one dedicated PII GLiNER model | — | — | PHI controls | Not clinical-tuned |
| Jev (hosted) | — | proprietary | Optional eval-track reference on synthetic data only | No weights; cannot see raw PHI |

All vendor-reported figures above are unverified until reproduced in Phase 2.

## Fine-Tuning Strategy (little or no labeled data)

1. **Define schemas first:** route labels + descriptions, eval questions + scales, HIPAA span labels. Version each as `label_set_version`.
2. **Synthetic generation** with a strong teacher LLM *inside the trust boundary* (local model for demo; in-VPC in production): diverse phrasing, hard negatives, and adversarial cases per label.
3. **Gold set:** a few hundred clinician-reviewed examples per track, used **only** for validation, calibration and final reporting — never for training.
4. **Train LoRA adapters** (GLiNER2 recipe: `use_lora=True`, r=8, alpha=16; early stopping on validation) per track on Decide / GLiNER2.5 base and Vela 0.3B where fine-tuning is supported. Adapters are 2–10 MB and hot-swappable. Confirm Vela's fine-tuning path in Phase 0 — the recipe was not found in the sources reviewed.
5. **Calibrate the gate:** pick thresholds on the gold set (per label) for a target selective precision (routing/evals) or target recall (PHI); report the resulting fallback rate. Re-check calibration after any quantization or adapter change.
6. **Close the loop:** fallback cases → review queue → clinician-reviewed → next training round.

## Bake-Off Design

A standalone harness under `agent-app/evals/decision_bench/` (no changes to `app/`):

```
evals/decision_bench/
├── datasets/           # synthetic + gold sets per track (no real PHI)
├── clients/            # one adapter per model/backend (DecisionClient)
├── serving/            # S1 baseline, S2 micro-batcher, S3 worker pool
├── loadgen.py          # open-loop concurrency 1/10/25/50
├── metrics.py          # accuracy/F1, selective accuracy, fallback rate, determinism, latency
├── cascade.py          # decision → gate → LLM fallback (uses build_llm())
└── run.py              # matrix runner → JSON + markdown report
```

**Matrix:** track × model × backend × serving variant × concurrency × {zero-shot, fine-tuned}.

**Acceptance gates (initial, to be refined in Phase 0):**
- Routing/evals: cascade accuracy within 2 points of LLM-only on the gold set at ≤ 50% fallback rate; p95 end-to-end decision latency ≤ 200 ms at concurrency 25 on the demo machine.
- PHI: per-category recall ≥ the best control and ≥ 0.95 on structured IDs and names (target; revise with clinicians); leakage rate reported with confidence intervals.
- Determinism: 100% identical outputs on the decision path across 100 repeats, thread counts {1, 2, 4, 8} and batch compositions.

## Phases

### Phase 0 — Verify assumptions (small)
- Install `gliner2`; load Decide, GLiNER2.5 small/base, Vela 0.3B, GLiFormer on the Mac; record what each exposes (confidence/probabilities, abstention, batch API, max tokens, fine-tuning support).
- Confirm thread-safety and batch-invariance with identical-output tests.
- **Done when:** a table of per-model signals exists and the confidence-gate definition per model is fixed.

### Phase 1 — Datasets and metrics
- Label schemas, synthetic generators, first gold sets per track; `metrics.py` with tests.
- **Done when:** each track has a validated dataset and reproducible metrics on a trivial baseline (regex / LLM-only).

### Phase 2 — Zero-shot bake-off
- Run the matrix zero-shot; serving variants S1–S3; produce the report.
- **Done when:** the report shows per-track winners (or "no model clears the gate") with latency at 1/10/25/50.

### Phase 3 — Fine-tune and calibrate
- LoRA adapters for the top two models per track; calibrate gates; re-run the matrix.
- **Done when:** cascade metrics (selective accuracy, fallback rate, end-to-end accuracy) are reported against the acceptance gates.

### Phase 4 — Integrate behind flags
- `AGENT_DECISION_*` settings; wire winners into `input_guard` layer 3, a route step, `output_guard`/input PHI, and the `decision_judge` evaluator; keep the LLM path as the default when the flag is off.
- Replace the guard's per-call semaphore with the shared micro-batcher.
- **Done when:** existing tests pass with flags off; new tests cover cascade behavior (accept, fallback, fail-closed PHI, timeout → fallback).

### Phase 5 — Operational hardening (optional)
- Metrics for fallback rate and drift; review-queue tooling; dataset refresh loop.

## EKS Notes (back of the head)

- The S2 micro-batcher becomes the model server; run it as a **shared service per decision type**, not a sidecar per agent pod (warm models, batching works, adapters hot-swap). Call it from an agent middleware or Envoy `ext_proc`.
- ONNX Runtime INT8 on Graviton3 (AWS reports up to ~30% on some NLP models); pin threads and cores; autoscale on queue depth/CPU. Re-validate accuracy after quantization.
- PHI: models self-hosted with open weights; fallback LLM in-VPC; audit-log decisions (label, confidence, path) without logging raw PHI.
- Public guidance on sidecar-vs-shared placement is thin; this is engineering judgment to be validated by load tests.

## Risks and Open Questions

- **Vendor numbers may not reproduce.** Mitigation: Phase 2 reproduces them on our hardware and data.
- **Confidence may not be calibrated or even exposed** (Decide). Mitigation: Phase 0; margin/threshold abstention; calibration set.
- **Apple Silicon ≠ Graviton/Xeon.** Latency and backend ranking may differ on EKS; keep the harness runnable on Linux x86/ARM.
- **Label-wording sensitivity** can make routing brittle; paraphrase tests and versioned label sets.
- **Zero-shot PHI recall may be too low to be useful**; fine-tuning and the regex layer are likely mandatory, and a fail-closed default is required.
- **Synthetic data may not reflect real clinical text.** Gold sets must be clinician-reviewed; real-note validation (de-identified, approved) is a later step outside this plan.
- **Encoder context limits** for eval checks on long answers.
- **Licenses and model naming** across vLLM-SR docs (Vela / Decision 2.0 / Kai–Vega) are inconsistent; pin exact artifact names and revisions in the harness.
- **Open:** acceptable fallback rate and latency budget per track; who provides clinician review and how much; whether `input_guard` should keep GLiGuard for harm categories alongside a new scope router.

## Evidence and Sources

Unverified vendor or third-party claims unless noted. Official model cards for Decide and several vLLM-SR pages could not be read directly in the research session; details were taken from search summaries and third-party pages.

- Vela 2.0 launch: https://vllm-sr.ai/blog/vela-2-0-open-foundation-routing-models/
- Vela-2.0-0.3B model card: https://huggingface.co/vllm-sr/Vela-2.0-0.3B
- Vela CPU latency issue: https://github.com/vllm-project/semantic-router/issues/4668
- Router decision schema limitation [issue #4638]: https://github.com/vllm-project/semantic-router/issues/4638
- Decision model cascade ("Decision Models Need a Router, Too"): https://vllm-sr.ai/blog/system-one-auto/
- GLiNER2.5-Decide launch (Fastino): https://fastino.ai/blog/gliner-2-5-decide-open-weight-decision-model
- Decide label-wording analysis: https://huggingface.co/blog/Hanno-Labs/less-is-more-gliner25-decide-and-the-shape-of-a-de
- Independent DecisionBench result: https://github.com/Hanno-Labs/decision-bench-results/pull/69
- GLiNER2 training (LoRA, data format): https://github.com/fastino-ai/GLiNER2#-training-custom-models
- GLiFormer: https://github.com/Knowledgator/GLiFormer
- GLiNER concurrency findings (cognee PR): https://github.com/topoteretes/cognee/pull/5213
- LangChain × Jev: https://www.langchain.com/blog/jev-is-now-available-in-langsmith-evals
- Clinical de-identification: https://www.johnsnowlabs.com/clinical-de-identification-benchmarks-2026-john-snow-labs-against-openai-databricks-presidio-and-llm-apis/ · https://arxiv.org/html/2610.07224 · https://intuitionlabs.ai/articles/open-source-phi-de-identification-tools
- Small PHI detector write-up: https://dev.to/raihan-js/where-small-models-beat-frontier-llms-and-where-they-dont-a-125m-phi-detector-4edb
- PII models: https://huggingface.co/nvidia/gliner-PII · https://arxiv.org/pdf/2605.09973
- LLM-as-judge in healthcare: https://arxiv.org/html/2605.25273v1
- Graviton ONNX INT8: https://aws.amazon.com/blogs/machine-learning/accelerate-nlp-inference-with-onnx-runtime-on-aws-graviton-processors/
