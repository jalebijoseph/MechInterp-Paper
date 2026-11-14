# Experiment 5 runtime handoff

The implementation covers development, frozen handoff, evaluation, and report
recomputation. Actual Qwen3-32B validation and experiments have not been run.
The numerical settings in [exp5_study.json](../configs/exp5_study.json) are
proposals; they are not a completed preregistration.

## Check the pipeline locally

From the repository root, use a Python environment with NumPy. The existing
`.venv-exp5/Scripts/python.exe` also has the dependencies for the tiny-model tests.

```powershell
.venv-exp5/Scripts/python.exe scripts/run_exp5.py preflight --out runs/exp5-development
.venv-exp5/Scripts/python.exe scripts/run_exp5.py preflight --phase evaluation --statistical-check
.venv-exp5/Scripts/python.exe scripts/run_exp5.py simulate --out runs/exp5-smoke
.venv-exp5/Scripts/python.exe scripts/run_exp5.py report --run runs/exp5-smoke/evaluation
```

Use fresh output paths. `preflight` creates no artifacts and makes no model
calls. `simulate` uses a small software-test protocol and generates development,
frozen, and evaluation folders. Every result remains synthetic, and the frozen
handoff cannot authorize target-model execution. `--no-diagnostics` can check
the shorter orchestration path, but cannot establish online-transfer readiness.

Run the scoped tests without the repository's model-loading pytest fixtures:

```powershell
.venv-exp5/Scripts/python.exe -m unittest discover -s tests -p 'test_exp5_*.py' -v
.venv-exp5/Scripts/python.exe -m unittest discover -s tests -p test_peer_capability_setup.py -v
```

CPU backend tests include a randomly initialized tiny Qwen3 and local tokenizer;
they download no checkpoint. The tested environment is Python 3.12, NumPy 1.26.4,
Torch 2.4.1+cu124, and Transformers 5.5.0. A real deployment also needs Accelerate
for `device_map="auto"`, a suitable Torch/GPU installation, and a separately
prepared pinned model cache. These CPU checks do not validate that deployment.

## Resolve before a real launch

- Review the proposed workload, capture layers, ridge/rank/dose candidates,
  thresholds, delay/checkpoint schedule, matching tolerances, and analysis windows.
  See [the temporary decision log](superpowers/specs/questioning.md#e5-063).
- Measure development serialization and runtime, then choose the final family
  count, power/precision target, and positive call/time limits before evaluation.
  The candidate 96 evaluation families are not a power guarantee.
- Fit the complete workload within the approved **1 GiB total artifact cap**.
  The current per-phase planning estimates are about 843 MiB development
  and 1,463 MiB evaluation, already about **2.25 GiB combined**, before frozen
  artifact copies. Those defaults require a revised workload/storage plan.
  No increase in the allowance has been approved.
- Check the statistical preflight. The current default null simulation fails
  its precision gate: the uniform boundary check has a one-sided upper bound
  about 0.073, above the proposed 0.070 ceiling. A failed gate stays failed;
  changing the procedure or calibration settings requires a prospective decision.

Forecasts use explicit planning assumptions, including 4 KiB per ordinary call
and 1 KiB per audit call. They are not empirical compression measurements. The
loose uncompressed upper bounds can be much larger. Artifact writes enforce the
total and category caps regardless of the forecast. Development ancestors,
frozen copies, evaluation, and reports share one logical allowance. Existing
model weights, library installations, and additional backups/LFS copies are
outside that result-data accounting.

## Real-model phase order

The following commands are an operator handoff, not a request to launch now.
First prepare the exact local snapshot of `Qwen/Qwen3-32B` at revision
`9216db5781bf21249d130ec9da846c4624c16137`. The loader uses local files only.
Set `$Exp5ModelCache` to that cache and `$Exp5CallLimit` to a deliberately chosen
positive request limit. A positive `--max-seconds` may be supplied instead or too.

```powershell
.venv-exp5/Scripts/python.exe scripts/run_exp5.py validate-runtime --execute --cache-dir $Exp5ModelCache --max-model-calls $Exp5CallLimit
.venv-exp5/Scripts/python.exe scripts/run_exp5.py develop --execute --config configs/exp5_study.json --out runs/exp5-development --cache-dir $Exp5ModelCache --max-model-calls $Exp5CallLimit
.venv-exp5/Scripts/python.exe scripts/run_exp5.py freeze --development runs/exp5-development --out runs/exp5-frozen
.venv-exp5/Scripts/python.exe scripts/run_exp5.py evaluate --execute --config runs/exp5-frozen/config.json --frozen runs/exp5-frozen --out runs/exp5-evaluation --cache-dir $Exp5ModelCache --max-model-calls $Exp5CallLimit
.venv-exp5/Scripts/python.exe scripts/run_exp5.py report --run runs/exp5-evaluation
```

Before freezing, `--evaluation-families N` can finalize the evaluation count
without refitting the probe. It recomputes the statistical preflight; evaluation
must then use the resulting frozen configuration. This option does not calculate
power or choose N automatically. Keep the implementation unchanged between
development, freeze, and evaluation: source hashes and backend fingerprints are
verified. Keep ancestor directories available, preserving their relative layout
if moving the handoff together.

Real development/evaluation print the pre-run storage notice before loading
weights and reject inadequate space, an existing destination, or an estimate
above the allowance. Development includes the planned evaluation in that check;
evaluation includes its verified retained ancestors. For an offline evaluation
forecast with ancestors, supply `preflight --phase evaluation --config
runs/exp5-frozen/config.json --frozen runs/exp5-frozen`. Frozen-copy and future
report overhead remain additional to the displayed minimum; hard byte accounting
covers them when written. Request/time caps stop further runner calls; they are not
hard GPU-job cancellation limits, and runtime validation performs additional
internal forwards. Use a scheduler limit for an external wall-clock deadline.

Full mechanistic readiness requires actual native replay/mask/edit validation,
controlled probe validation, live transfer/attribution, selective causal discovery,
and statistical preflight. `--allow-inconclusive` permits an explicitly descriptive
handoff after neural assay failures, while still requiring real runtime validation.
It does not convert failed gates into mechanistic evidence. Budget stops remain
incomplete and cannot be frozen as completed development.

## What is retained

| Artifact | Use |
|---|---|
| `config.json`, `backend.json`, `runtime.json` | Protocol, environment identity, and actual runtime checks. |
| `records.jsonl` | Choices, checked live replies, exposure/unknown counts, every-turn readouts, audits, diagnostic failures, and causal receipts. |
| `histories/` | Deduplicated compressed exact native token histories for replay. |
| Development captures and `probe.json` / `probe.npz` | Bounded discovery vectors and the frozen readout/bases; no pickled objects. |
| `validation.json`, `noninferiority-preflight.json`, `frozen.json` | Gates and the permitted interpretation of the handoff. |
| `artifact-index.json`, `manifest.json` | File hashes, exact byte accounting, ancestry, and completion status. |
| `report.json`, `report.md` | Recomputed family-level outcomes, all P1-P7 entries, missingness, and interpretation limits. |

Main trajectories generate selected-peer responses live and keep compact
measurements rather than full weights, KV states, or every-turn activation arrays.
Controlled calibration is separate. Audits, attribution swaps, retention branches,
donors, and causal interventions operate on independent forks and never feed their
results back into the main conversation. Decoding alone is not a causal result.

See the [scientific design](superpowers/specs/2026-11-08-experiment-5-peer-capability-retention-design.md)
and [implementation ledger](superpowers/plans/2026-11-11-experiment-5-runtime.md).
