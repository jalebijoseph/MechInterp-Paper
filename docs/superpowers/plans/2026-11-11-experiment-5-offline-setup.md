# Experiment 5 Offline Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare inspectable Experiment 5 configuration, world-task fixtures, prompt contracts, checkers, and offline validation without running a model.

**Architecture:** Add one standard-library setup module and a thin preparation script. Generate matched development fixtures and analysis-only truth separately from rendered model messages. Export a deterministic, size-checked setup bundle whose manifest explicitly reports that execution and neural validation remain incomplete.

**Tech Stack:** Python 3.10+, standard-library JSON, random, hashlib, argparse, pathlib, unittest. No new dependencies.

**Spec:** [Experiment 5 design](../specs/2026-11-08-experiment-5-peer-capability-retention-design.md), especially sections 3, 5, 6, 11, 12, and 14. Setup decisions are recorded in [questioning.md](../specs/questioning.md), E5-042/E5-058/E5-059.

## Global Constraints

- User requested experimental setup; model inference, downloads, servers, cluster jobs, pilot/evaluation runs, and compute-budget selection are deferred.
- Shared checkpoint: `Qwen/Qwen3-32B`, revision `9216db5781bf21249d130ec9da846c4624c16137`.
- `enable_thinking=False`; seeded sampling for all three roles, with matching sampler parameters for the two peers. Numerical parameters and role seeds remain unresolved.
- Delegation choices are `A` or `B`; abstention is outside the initial experiment, including delegation diagnostics.
- Domains: possession and location. Include equal information, correct versus incomplete, and correct versus incorrect information.
- Development private-input rates: reference correct target 100%; incomplete correct target 50%, omitted otherwise; incorrect correct target 50%, false otherwise. These are not forced output-accuracy rates.
- Initial result-data ceiling: 1 GiB (`1073741824` bytes) for one logical copy. A setup-bundle size is not a full-study storage forecast.
- Frozen online probes, native state/replay equivalence, route masks, causal edits, and full workload/registry validation remain later implementation milestones.
- Preserve existing work and the temporary question log. Do not stage, commit, create branches, or launch execution as part of this setup task.

## Files and Interfaces

- Create `configs/exp5_setup.json`: settled choices plus explicitly proposed fixture defaults; no endpoints or launch command.
- Create `src/jpeers/experiments/peer_capability_setup.py`: validation, deterministic fixture construction, objective checker, prompt rendering, and bundle export.
- Create `scripts/prepare_exp5.py`: offline CLI, usable without installing the GPU stack.
- Create `tests/test_peer_capability_setup.py`: standalone CPU-only unittest checks; do not invoke the repository's model/tokenizer fixtures.
- Create `docs/experiment-5-setup.md`: preparation commands, artifact boundaries, and remaining implementation work.

Proposed fixture defaults are seed `0`, four examples per domain, and peer names `Cedar`/`Maple`. These are examples, not approved study seeds, names, horizon, sample sizes, or evaluation stimuli. Independently cross the reference peer for the two domains: four assignments per unequal-information comparison, plus one equal-information family. This produces 72 example rows. Limit this preview to an even 2–64 cases per domain; this bound is a setup safeguard, not a study constraint.

Use three independently sampled people per scene, each with one binary state. Private packets contain two facts: a common non-target fact and either the queried fact or another true non-target fact. Incorrect packets replace only the queried fact on scheduled cases. This keeps packet fact counts equal without forcing generated model answers. These world/template choices are provisional fixtures.

## Review Focus

1. Public task messages must not acquire hidden truth, private packets, condition labels, or reference-peer assignments.
2. A missing target fact must not be silently converted to false information; incorrect targets must actually disagree with world truth.
3. Reference identity must vary independently by domain while remaining fixed within a fixture family.
4. Preparation must work without torch, transformers, network access, model calls, or execution authorization.
5. Existing output directories and the storage ceiling must be respected; fixture validation must never claim neural/runtime readiness.

### Task 1: Configuration, matched fixtures, and message/checker contracts

**Files:** Create configuration, setup module, and test file listed above.

**Interfaces:**

- `validate_config(config: dict) -> None`: accept setup-only configuration; reject contradictory settled choices, invalid fixture settings, or budgets above 1 GiB.
- `build_fixtures(config: dict) -> list[dict]`: deterministic example families with `private_facts` and separate `analysis` truth/condition metadata.
- `render_prompts(fixture: dict, config: dict) -> dict`: initial orchestrator messages and separate messages for each peer; explicit allowlists, never render the full fixture into an agent prompt.
- `grade_answer(fixture: dict, answer: str) -> bool`: exact answer-label comparison after stripping/case folding; no model grader or substring scoring.

- [x] Write tests `test_protocol_validation`, `test_private_input_schedules_and_world_truth`, `test_reference_assignments_and_public_information`, and `test_objective_checker` with these assertions: default inventory has 72 unique rows; each unequal family/domain has exactly two of four target-present/correct cases for the disadvantaged peer; every packet has two facts; other supplied facts match world truth; all four domain assignments exist; changing truth/private packets leaves the orchestrator prompt unchanged; an incorrect private answer fails the objective checker.
- [x] Run `python -m unittest discover -s tests -p test_peer_capability_setup.py -v`; observe failure because the new module is absent.
- [x] Implement the interfaces using only the standard library. Keep fixture IDs, assignments, and truth out of rendered messages. Label generated stimuli as development fixtures, not frozen evaluation data.
- [x] Run the same command; inspect the fixture/checker checks before proceeding.

### Task 2: Safe offline preparation and preflight report

**Files:** Extend setup module/tests; create `scripts/prepare_exp5.py`.

**Interfaces:**

- `prepare_setup(config_path: Path, output_dir: Path) -> dict`: validate inputs; write `config.json`, `fixtures.jsonl`, and `manifest.json` into a new directory; return a JSON-serializable storage/readiness report.
- CLI: `python scripts/prepare_exp5.py --config configs/exp5_setup.json --out runs/exp5-setup-20261111`.
- Manifest: schema/stage, file hashes, fixture count, setup status, and unresolved execution gates. Both manifest and report must have `ready_for_execution: false` and `model_calls: 0`.

- [x] Add `test_export_is_deterministic_offline_and_not_execution_ready`, `test_export_preserves_existing_outputs_and_checks_budget`, and `test_cli_prepares_only_setup_artifacts`. Assert identical bytes for repeated preparation to different fresh directories, valid SHA-256 receipts, actual bytes within the ceiling, zero calls while sockets/model imports are unavailable, preservation of a pre-existing sentinel, and rejection before creating output when the bundle exceeds a smaller configured limit. The CLI must reject `--run`.
- [x] Run the test command and observe the new checks failing for missing export/CLI behavior.
- [x] Implement canonical UTF-8 serialization, compute all payload sizes before writing, reject existing output directories, and expose preparation only. Do not register an Experiment 5 live runner or add an endpoint client. The CLI may add the repository's `src` to its own import path so preparation needs no GPU-stack installation.
- [x] Run the standalone suite. Run the CLI once to prepare the local example bundle and inspect its manifest/report.

### Task 3: Documentation and honest readiness handoff

**Files:** Create setup guide; update the design/question log and README guide link.

- [x] Document CPU-only preparation/testing, the analysis-only fields in the fixture bundle, provisional fixture choices, actual setup bytes versus unknown full-study bytes, and the remaining runtime/probe/causal work.
- [x] Record all assistant defaults in E5-058/E5-059 without marking them user-approved. Keep the question log unstaged.
- [x] Check the plan against the accepted design and run `python -m unittest discover -s tests -p test_peer_capability_setup.py -v` plus the preparation CLI in a fresh directory.
- [x] Obtain a fresh read-only code review of these setup files and address material findings. Report verified setup readiness separately from execution readiness.

## Later Implementation Milestones

The offline bundle is not a runnable full Experiment 5. Later work adds the live orchestrator/selected-peer loop and exposure logging; native continuation/forks or verified replay; activation capture and offline linear-probe fitting; frozen every-turn readouts; independent audits; validated attention masks; donor/removal/rescue controls; bounded live artifact retention; and the registry/power/resource freeze. Model-based validation and execution budgets are deferred until the user plans a run.

## Execution Record

- Scope review: fresh builder check returned “nothing important left” for the offline milestone.
- Workspace: retain the existing checkout and untracked review documents; no branch/worktree or Git mutation is needed for the new setup files.
- TDD evidence: first observed the missing-module failure; fixture/checker checks passed before export/CLI existed; then all seven checks passed. Added failing strict-schema/source-scene checks and verified their fixes.
- Verification: `python -m unittest discover -s tests -p test_peer_capability_setup.py -v` — 7 tests passed. The command uses the approved system-temp-directory access; no model/tokenizer fixtures were invoked.
- Prepared current preview: `runs/exp5-setup-20261111-v2`, 72 rows, 146424 bytes, zero model calls, execution readiness false. The earlier preview is preserved; choose a fresh directory for subsequent preparation.
- Fresh read-only code review found no critical or important issues. Its minor execution-record/checklist finding is resolved here.
- Self-review: each Review Focus item maps to the fixture/prompt, checker, export, or CLI tests. Live/model-based milestones remain explicitly deferred; numerical defaults are proposals in the temporary question log.
