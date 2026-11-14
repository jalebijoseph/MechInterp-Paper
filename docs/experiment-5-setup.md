# Experiment 5: offline experimental setup

The first setup milestone prepares task inputs and validates their contracts on CPU.
Model-based development, GPU allocation, and experiment execution follow later.

## Prepare the example bundle

From the repository root:

```powershell
python scripts/prepare_exp5.py --config configs/exp5_setup.json --out runs/exp5-setup-20261111
```

This uses Python's standard library. It needs no model installation, endpoint,
tokenizer download, or GPU. Choose a fresh output directory: existing directories
are preserved and preparation fails rather than overwriting them.

The command prints the measured size of the **setup bundle**, its fixture count,
`model_calls: 0`, and `ready_for_execution: false`. `full_study_bytes: null` means
the complete experiment's storage forecast still depends on its future workload.
The configured 1 GiB ceiling remains the allowance for one logical copy of all new
Experiment 5 result artifacts; the preview's size does not establish that a full
study will fit.

## Inspect the artifacts

| File | Purpose |
|---|---|
| `config.json` | Settled scientific choices and proposed fixture settings. |
| `fixtures.jsonl` | Matched world-task examples, private peer facts, rendered per-role messages, and analysis-only truth. |
| `manifest.json` | Input hashes, generator identity, Python version, fixture count, and pending execution gates. |

Each fixture has separate `private_facts`, `analysis`, and `prompts` fields. Only
the relevant role's rendered messages belong in a future model request. The
`analysis` fields include world truth and condition assignments for checking;
they are never rendered into the orchestrator's messages. This preparer creates
inputs, while the future live runner will generate actual peer responses.

## What the fixtures cover

- Shared pinned Qwen3-32B identity; non-thinking and seeded sampling policies for all roles, with matching peer sampler parameters.
- A/B-only delegation and possession/location questions.
- Equal-information, correct-versus-incomplete, and correct-versus-incorrect examples.
- All four independent assignments of the reference peer across the two domains in each unequal-information comparison.
- Correct target information on every reference example; 50% target omission for incomplete information; 50% false target facts for incorrect information.
- Two facts in every private packet. Omitted targets are replaced by another true, irrelevant fact; misinformation replaces only the target fact.
- Objective answer checking against world truth, including when a private packet contains false information.

The default seed `0`, names `Cedar`/`Maple`, and four examples per domain are
**assistant-proposed fixture defaults**, not approved study seeds, names, round
counts, or evaluation sample sizes. Nine fixture families produce 72 rows. An
even 2–64 examples per domain is supported for preview; this bound is not a limit
on the later study.

Condition groups reuse matched scenes, identified by `analysis.scene_id`. The
72 rows and nine fixture groups are not independent experimental replications.
The later live study needs its own session/family grouping for statistical inference.

These initial examples use three independent people and binary static world
facts. The queried person is Alice. Other people's states do not determine her
state. Final task/name/template pools, world complexity, counterbalances, and
development/evaluation splits remain to be frozen before model-based validation.

The proposed peer response contract is `{"answer": "chair"}` (or a location
label); `"unknown"` is available to peers with insufficient information. The
checker compares the extracted label after whitespace trimming and case folding;
it does not accept explanatory sentences as labels. This is distinct from the
orchestrator's A/B-only delegation action.

## Check the setup

```powershell
python -m unittest discover -s tests -p test_peer_capability_setup.py -v
```

The standalone unittest suite avoids `tests/conftest.py`, whose model fixtures
can import the GPU stack and download a tokenizer. The setup suite checks private
input rates, independent domain assignments, truth/prompt separation, scoring,
deterministic exports, file hashes, storage bounds, overwrite protection, and
the preparation-only CLI.

## Runtime implementation

The [runtime pipeline and operator guide](experiment-5-runtime.md) now cover live
conversation execution, native replay with persistent edits/masks, bounded
development captures, frozen linear probes, every-turn readouts, independent
audits, retention/causal branches, artifact accounting, and family-level reports.
CPU and synthetic checks exercise the implementation; actual Qwen3-32B runtime,
readout, and causal validation remain pending. This fixture preparer's manifest
continues to describe its own preparation-only milestone.

Numerical study settings, compute ceilings, power/precision planning, and a full
workload that fits the approved storage allowance still need to be frozen before
evaluation. Setup validation supplies no evidence that Qwen internally represents
peer skills or that a causal intervention works.

See the [design](superpowers/specs/2026-11-08-experiment-5-peer-capability-retention-design.md),
[implementation plan](superpowers/plans/2026-11-11-experiment-5-offline-setup.md), and
[temporary decision log](superpowers/specs/questioning.md).
