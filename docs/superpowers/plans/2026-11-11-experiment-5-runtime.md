# Experiment 5 Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the remaining runnable Experiment 5 development/evaluation pipeline, without launching study models during implementation.

**Architecture:** A token-native replay backend retains exact conversation tokens and replays every registered mask/edit. Separate task construction, numerical probe/statistics code, and runner/storage orchestration; use the existing offline preparer and strict JSON parser where applicable. A labelled simulator exercises the complete workflow; it can never establish target-model readiness.

**Tech Stack:** Python 3.12, NumPy, PyTorch and Transformers; standard-library CLI/JSON/gzip/hashlib. The isolated `.venv-exp5` uses existing NumPy/Torch with Transformers 5.5.0 installed for Qwen3 support. CPU integration tests use a randomly initialized tiny model/tokenizer, never download a checkpoint. Real auto-device placement additionally requires Accelerate.

**Spec:** [Experiment 5](../specs/2026-11-08-experiment-5-peer-capability-retention-design.md). The user now explicitly requests the remaining implementation; actual experiment execution remains deferred. Defaults below are assistant proposals, recorded in the question log.

## Global Constraints

- Pin `Qwen/Qwen3-32B` revision `9216db5781bf21249d130ec9da846c4624c16137`; shared frozen weights, separate role prefixes, `enable_thinking=False`.
- Seeded sampling for all roles; matching peer parameters. Proposed sampler: temperature 0.7, top_p 0.8, top_k 20, min_p 0.0; no parameter is user-approved merely by implementation.
- A/B delegation only, free from the first task; selected-peer response/checker feedback only. Source peers use fresh task-only requests as the proposed initial context policy.
- Possession/location; equal information and separate correct/incomplete and correct/incorrect comparisons. Private-input schedules are 100%, 50% supplied, and 50% false; actual answers are generated live.
- Initial linear readout target is observed success rate with exposure/unknown coverage. Decoding alone never establishes causal use.
- Raw development captures bounded at proposed 256 MiB; total new run artifacts bounded at 1073741824 bytes. Compact main-run measurements; no saved weights/KV snapshots.
- Keep all trials/failures and all seven declared primary entries. Freeze before evaluation, group by independent source family, separate controlled calibration from spontaneous online results.
- Existing staged/untracked work and question log are preserved. No commits, model downloads, cluster launch, or actual study inference in this implementation task.

## Interfaces shared by implementation tasks

Create `src/jpeers/exp5/` with a lightweight `__init__.py`. `protocol.py` owns `CELL_ORDER = (("A","possession"),("A","location"),("B","possession"),("B","location"))`, `derived_seed(seed: int, *parts: object) -> int`, config validation/hash, strict action parsing, and artifact budgets.

### Task construction (`tasks.py`)

- `session_tasks(config: dict, split: str, family: int, comparison: str, *, audit: bool=False) -> list[dict]` returns fresh task rows with `id`, `domain`, `question`, `query_subject`, `options`, `private_facts` (A/B lists of subject/value objects), and `analysis` (`true_answer`, `world`, `conditions`, `reference_by_domain`, `phase`). Families share matched worlds across comparisons; never render analysis fields.
- `calibration_tasks(config: dict, family: int, profile: list[float]) -> list[dict]` returns checked scripted observations with an added `peer`, `response` (JSON answer text), and `correct` boolean; exactly ten observations per cell at default. This is a separate controlled assay, not main live responses.
- `calibration_profiles(config: dict) -> list[dict]` returns `{id, rates}` for all 16 independent binary four-cell profiles plus eight anchors (target high/low and competing cell 0.6).
- `peer_messages(task: dict, peer: str) -> list[dict]`; `roster(config: dict, split: str, family: int) -> dict[str,str]`; `public_task(task: dict) -> dict`; `grade_peer(task: dict, text: str) -> dict` with `answer`, `valid`, `correct`, `unknown`; `counterfactual_feedback(task: dict, peer: str, correct: bool) -> str` for controlled donor-history construction only.
- Exogenous balanced domain schedule, independently seeded streams, fixed family peer identities/domain assignment, explicit fresh scene, equal-size two-fact packets. Reversal starts at configured `tasks.reversal_round` (None disables). Audit requests have independent worlds/streams and phase coverage.

### Backend (`backend.py`)

- `HFBackend(model, tokenizer, *, model_id, revision, dtype="float32")`; classmethod `from_pretrained(config: dict, cache_dir: str|None=None)` loads only local cached pinned weights, eager attention, BF16 by default. Imports Torch/Transformers lazily.
- `info() -> dict` includes model_id, revision, n_layers, hidden_size, max_context, dtype, synthetic.
- `encode_text(text: str) -> list[int]`, `decode(ids: list[int]) -> str`.
- `append_message(prefix: list[int], role: str, content: str, *, close: bool=True) -> tuple[list[int], tuple[int,int]]`: append a Qwen-compatible fragment without re-rendering or re-tokenizing earlier tokens; return absolute content interval. Assistant fragments include empty non-thinking tags. `close=False` supports neutral receipt checkpoints and generation prefixes.
- `finish_assistant(prefix: list[int]) -> list[int]`: ensure the native generated/forced assistant turn has a terminator and trailing newline without changing previous tokens.
- `observe(tokens: list[int], capture: dict[int,list[int]], *, masks: list[dict]=(), edits: list[dict]=()) -> dict`: `captures[layer][position]` float32 NumPy vectors, last next-token `logits`, and compact `receipt`.
- `generate(tokens: list[int], *, sampler: dict, seed: int, max_new_tokens: int, masks: list[dict]=(), edits: list[dict]=()) -> dict`: text, token_ids (including a generated terminator), hit_cap, receipt. Never reset/use a global RNG; mask and edit lifetime includes all decoding steps.
- Mask: `{key_spans: [(start,end),...], start_query: int, tag: str}`; block those key positions for every query at/after start_query, alongside causal masking.
- Edit: `{layer:int, position:int, kind:"delete"|"patch"|"random_delete"|"random_patch", basis:np.ndarray[d,r], donor:np.ndarray[d] (patches), strength:float, seed:int, tag:str}`. Fixed component geometry, matched random norm/inner product, actual model-dtype displacement receipts. Rescue uses an independently identified later-layer basis, not same-site cancellation.
- `validate_runtime() -> dict`: actual full-versus-segmented cache/replay and edit/mask persistence comparisons, actual blocked attention check when supported, non-thinking token-format checks. Return passed/checks/fingerprint, never a fake pass. Layer numbers are zero-based block outputs. Tests inject a tiny local Qwen3 model and byte tokenizer.

### Numerical analysis (`probes.py`, `analysis.py`)

- `fit_readout(records: list[dict], *, ridge: float, layer: int) -> dict`: each record has family, captures (int layer to vector), target (4 floats), mask (4 bools), and delay. Fold-local centering/scaling and regularized linear heads; no label leakage.
- `predict_readout(artifact: dict, vector: np.ndarray) -> dict`: raw and clipped four-cell estimates.
- `select_readout(records: list[dict], layers: list[int], ridges: list[float], *, primary_delay:int=512) -> tuple[dict,dict]`: grouped development validation only, compare prior/identity-blind/permuted controls, unseen-family transfer, macro binary accuracy midpoint 0.6 on controlled panels. Expose gates/report; failure stays failure. Need at least two independent groups.
- `conditional_bases(records: list[dict], *, layer:int, rank:int) -> dict[int,np.ndarray]`: paired activation differences where only one target rate changes and other rates match; SVD bases. Never treat decoder weights as encoding directions.
- `save_artifact(path: Path, artifact: dict)`, `load_artifact(path: Path)`: versioned JSON metadata plus NPZ numeric arrays, no pickle; preserve fitted preprocessing, dimensions, site, evidence targets, and validation.
- `family_test(values: list[float], *, seed:int=20261108, n_permutations:int=50000, n_bootstrap:int=100000, margin:float=0) -> dict`: exact sign flips at <=20 families, otherwise seeded random assignments; family bootstrap intervals; record assumptions. Noninferiority as separate supported procedure, not treating p>0.05 as preservation.
- `holm(p_values: dict[str,float], alpha:float=.05) -> dict`; `summarize_records(records: list[dict], config: dict) -> dict` and `write_report(output_dir: Path, summary: dict)`. Aggregate families before inference; P1–P7 conjunctions use max component p, missing/unavailable entries remain family members. Include trajectories, conditional/unconditional allocation, observed unknown coverage, audits, literal/other-domain/validity specificity, and explicit incomplete/synthetic status. Never report simulated results as evidence for Qwen.

## Review Focus

1. No hidden world/quality labels reach orchestrator messages; source invalid/unknown answers and unselected peers stay explicit.
2. Chat rendering never replaces earlier native tokens; every mask/edit is reapplied on replay and decoding; model dtype matching is measured.
3. Probe fitting/splitting/selection cannot use evaluation trajectories or hidden capabilities as known observed evidence.
4. Diagnostics/audits/donors have independent RNG streams and never advance the main trajectory or supply it extra evidence.
5. Missing gates, budget stops, unavailable donors, invalid outputs, and incomplete trials remain in the registry/denominators; simulation cannot satisfy target-model readiness.

## Tasks and verification

### Task 1: Protocol, study configuration, and bounded artifacts (coordinator)

Files: `src/jpeers/exp5/protocol.py`, `storage.py`, `configs/exp5_study.json`, `tests/test_exp5_protocol.py`.

- [x] Write failing tests for pin/mode/action validation, role-separated deterministic seeds, strict JSON, safe no-overwrite and 1GiB/category bounds, and protocol hashes excluding only operational output/limits.
- [x] Implement config and content-addressed token histories/compact JSONL/NPZ artifacts, crash/incomplete manifests, exact byte accounting and reserved final-manifest space. Proposed fixture/study settings remain labelled proposals.
- [x] Run `py -3.12 -m unittest discover -s tests -p test_exp5_protocol.py -v`.

### Task 2: World tasks and controlled calibration (independent agent)

Files: `src/jpeers/exp5/tasks.py`, `tests/test_exp5_tasks.py`.

- [x] Test schedules/coverage, fresh independent scenes, matched conditions, name/order and reference assignments, reversals/audit isolation, all 16 profiles/eight anchors, actual checker agreement, and public/private separation.
- [x] Implement the task interfaces above without touching coordinator files.
- [x] Run the task-specific standalone unittest command under Python 3.12. Obtain fresh read-only review.

### Task 3: Native token/replay backend (independent agent)

Files: `src/jpeers/exp5/backend.py`, `tests/test_exp5_backend.py`.

- [x] Use an injected tiny randomly initialized Qwen3 model/local tokenizer; tests must not call from_pretrained or download anything.
- [x] Test append-only tokens, actual residual captures, deterministic sampling, attention to source positions blocked across every generated step, selective fixed edits/random geometry, downstream rescue, cache/replay equivalence, context-cap rejection, and truthful runtime receipts.
- [x] Implement the backend contract; run its unittest suite under the existing Python 3.12 Torch/Transformers environment. Obtain fresh read-only review.

### Task 4: Linear probes, conditional bases, statistics/reporting (independent agent)

Files: `src/jpeers/exp5/probes.py`, `analysis.py`, `tests/test_exp5_probes.py`, `test_exp5_analysis.py`.

- [x] Test grouped held-out selection on known binding signals, unknown-cell masks, preprocessing leakage guards, no-pickle round trips, conditional single-cell contrasts, exact/null/alternative statistics, fixed registry multiplicity, and synthetic/incomplete reporting.
- [x] Implement the numerical interfaces; no Qwen parameter updates or prompt feedback. Run targeted CPU tests and obtain fresh read-only review.

### Task 5: Live runner, diagnostics and handoff (coordinator)

Files: `src/jpeers/exp5/runner.py`, `simulation.py`, `scripts/run_exp5.py`, `tests/test_exp5_runner.py`, setup/runtime guides and decision log.

- [x] Test complete seeded simulated development/freeze/evaluation/report workflow; no label leaks, no main RNG drift from diagnostics, zero-retry model invalid handling, independent audits, correct unknown exposures, exact masks/donor-target construction, and incomplete-budget reporting.
- [x] Implement token-native main conversations, neutral post-feedback receipts, calibration capture and fit/freeze, immediate/128/512-token retention, I0–I7 and specificity controls under R0/R2, matched donor construction, live audits, updating, compact every-turn records, strict freeze/evaluation validation and independent recomputation.
- [x] Predeclare irrelevant neutral context in main sessions to match source/control masking opportunities; store exact counts/age gaps, and mark route matching inconclusive if the registered tolerance fails. This controlled-context construction is a disclosed proposal.
- [x] CLI commands: `preflight`, `simulate`, `validate-runtime`, `develop`, `freeze`, `evaluate`, `report`. Explicit real-model commands only; `simulate` labels all records synthetic and cannot produce a Qwen-ready freeze. Missing model caches fail with a preparation message, never download implicitly.
- [x] Run all `test_exp5_*.py` suites plus the previous offline setup regression. Exercise full simulation and inspect actual artifact/report receipts. Model-based discovery may fail scientifically; tests must not convert failure to success.
- [x] Update run instructions, unresolved-before-launch settings, and the question log; perform broad fresh-context review. Preserve user files/index and keep questioning.md unstaged.

## Preflight scan and execution ledger

| Tasks | Shared interface / self-check | Ruling |
|---|---|---|
| 1→2/4/5 | CELL_ORDER/config/seeds | Freeze contracts here; coordinator owns protocol/storage only. |
| 2→5 | Tasks expose truth in analysis only | Render explicit allowlists; no serialized full-row agent prompts. |
| 3→5 | Exact token arrays, absolute sites, masks/edits | Append fragments, never re-render earlier chat; verify replay before real evaluation. |
| 4→5 | Arrays, grouped records, artifact metadata | Use the named keys/signatures; no pickle or evaluation fitting. |
| 1 | Budget vs setup authorization | Budget can remain unset for building/simulation; real launches require declared limits. |
| 2 | Scripted calibration vs live study | Calibration is labelled controlled; main/audit peer answers are generated live. |
| 3 | Missing default-interpreter Torch and old Python 3.12 Transformers | Reuse existing Python 3.12 Torch through an isolated environment and install Transformers 5.5.0 there. No model files are downloaded. |
| 4 | Strongest claim vs failed gates | Keep all primaries; report unavailable/inconclusive instead of dropping failed cases. |
| 5 | Complete pipeline vs no study execution | Implement and simulate all paths; leave actual Qwen validation/execution for the operator. |

Ruling: extend the already-reviewed spec with explicit implementation proposals instead of asking settled questions again. The build request authorizes required library preparation and CPU/synthetic checks; it does not authorize study model inference, model downloads, Git mutations, or a claim of validated peer knowledge.

### Implementation and review record, 2026-11-13

- Implemented the protocol/task/storage/backend/probe/analysis interfaces, live
  runner and fork diagnostics, complete CLI, and operator guide. Raw main-turn
  states are not archived; compact measurements and exact native histories are.
- Final verification passed all 128 `test_exp5_*.py` tests in 314.517 seconds and
  all seven previous offline setup tests. This includes the complete simulation
  with diagnostics, exact artifact checks, freeze/evaluation/report recomputation,
  tamper rejection, and the added live-gate/combined-storage regressions.
- Direct CLI `simulate --no-diagnostics` completed at
  `runs/exp5-runtime-handoff-20261113`, followed by CLI report recomputation.
  Development/frozen/evaluation/report artifacts total 8,807,657 logical bytes;
  all manifests are complete and synthetic. The verified two-family report keeps
  P1-P7 and every target-model support flag false, as required. The shorter CLI
  fixture does not estimate the real study's storage or validate its assays.
- Tiny random Qwen3 CPU tests use a local tokenizer and Transformers 5.5.0. No
  target-model checkpoint was downloaded or executed. Workspace-owned temporary
  directories avoid this sandbox's restrictive Windows temporary-directory ACLs.
- Independent backend/numerical/task/runner reviews found and corrected missing
  receipt/gate handling, missing-family selection, cross-session audit pooling,
  primary/secondary delay mixing, route/component pooling, BF16 control mismatch,
  and attribution failures being omitted. Regression tests preserve these cases.
- Final fresh review checked live-only held-out attribution, full-study and
  ancestor forecasts, the CLI's refusal before loading weights, and the guide.
  It reported no important issues; all 14 targeted review checks passed.
- Live transfer requires applicable evidence from every held-out development
  family and cannot borrow controlled calibration attribution. Explicitly tied
  or unobserved attribution is inapplicable; missing/failed applicable results
  invalidate the gate.
- Source/file/backend fingerprints and ancestor receipts make a handoff
  reproducible. Keep package sources stable between development and evaluation.
  Final evaluation N may be set prospectively at freeze without refitting.
- Default preflight uses zero model calls and exposes a combined planning minimum
  of 2,418,802,688 bytes, excluding future frozen copies/reports. This exceeds the
  user-approved 1 GiB allowance; real launches reject the proposed workload.
- The actual CPU NI null check also fails its proposed precision gate at N=96
  (uniform 27/500 rejections; upper bound 0.073154 > 0.070). It remains failed.
  No sample-power or neural-validation success is claimed.
- Remaining before real execution: revise/freeze the workload to fit the cap,
  review numerical proposals, prepare deployment dependencies/local model cache,
  declare compute limits, validate the actual runtime/assay, and finalize the
  evaluation power/precision plan from development. These are future study gates,
  not unfinished software tasks authorized to run now.

Handoff: [runtime guide](../../experiment-5-runtime.md),
[study configuration](../../../configs/exp5_study.json), and temporary
[decision log E5-063 onward](../specs/questioning.md#e5-063).
