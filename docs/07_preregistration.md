# 7. Pre-registration

Fixed before the main (8B) runs. Every experiment reports many metrics; this page says
in advance which **one** per experiment decides the result, how it is tested and what
counts as a positive. Everything else is secondary and is reported as exploratory.
Any change made after main-run data exists goes in the deviations log at the bottom,
with the date and the reason.

## Models and order

* Primary model: **Qwen3-8B** (Neuronpedia lens, default band). Llama-3.1-8B-Instruct
  and Gemma-3-12B-it are replications; Exp. 3 `self_preference` needs a second family.
* **Exp. 0 is the gate.** Exp. 1–3 results are interpreted only if Exp. 0 passes on the
  same model. Exp. 0 is also run across the Qwen3 ladder (0.6B–32B) to show where, if
  anywhere, belief tracking appears.
* Exp. 4 runs last, on the model with the clearest Exp. 1–3 signal.

## Primary metrics

Each primary metric is read away from the answer position, or against the output
baseline, so a positive result cannot simply restate what the model was about to say.

| Exp | Primary metric | Unit | Test | Positive if |
|---|---|---|---|---|
| 0 | `beats_story_end`: belief word outranks the decoy at the last story token, before the question is asked | item | exact binomial vs 0.5, separately on false-belief items **and** true-belief controls | both rates > 0.5 and both p < α |
| 1 | `partner_kw_mrr_beyond`: team − named_solo | task (mean over turns) | paired sign-flip | diff > 0, p < α |
| 2 | `susp_mrr_reading_beyond`, contrast `tom_false`: incentive − no_incentive, false claims | fact | paired sign-flip | diff > 0, p < α |
| 3 | `task_prefers_right` (context) and `chose_own` (self_preference, cross-family) | decision | exact binomial vs 0.5 | reported separately; `task_prefers_right` > 0.5 supports peer modelling, `chose_own` ≠ 0.5 shows family bias |
| 4 | `catch_score` on false claims (mean of `reply_has_true` and `reply_flags`): `jspace_generated` − `matched_generated` | (source, fact) | paired sign-flip | diff < 0, p < α, **and** `fluency_ok` (fluency within 0.1 of the control) |

**α.** 0.05 family-wise across the five experiments, Holm-corrected. Exp. 3's two
metrics are reported as one family member each, with Holm applied to the six p-values.

**Exp. 4 reading.** A J-space removal that lowers `catch_score` more than an equal-size
random edit at the same positions, with the text still fluent, means the agent needs
J-space to communicate well. A fall in fluency as well means the edit broke the text,
not the communication, and the result does not count.

## Sample sizes

Set from the 8B pilot with `scripts/power.py` (80% power at α = 0.01, the worst-case
Holm threshold). Fill in before the main runs:

| Exp | Pilot effect | n needed | n planned | Bank size now |
|---|---|---|---|---|
| 0 | | | | generator: unlimited; Hi-ToM: 600 |
| 1 | | | | **8 tasks** (expand if n needed > 8) |
| 2 | | | | 60 facts |
| 3 | | | | 12 tasks × 4 counterbalancings |
| 4 | | | | 60 facts × 3 sources |

If the planned n exceeds a bank, the bank is extended *before* the main run, with the
same construction rules (single-token words, no prompt overlap; enforced by
`tests/test_watch_words.py` and `tests/test_scenarios.py`).

## Exclusions (decided in advance)

* Exp. 2/4: items where a live peer never stated the claim (`claim_in_draft` false).
* Exp. 3: decisions whose reply names neither option (`parsed` false). In the bandit
  they are filled at random and excluded from lock-in metrics.
* Exp. 0 Hi-ToM: items with a container noun that is not a single token (counted).
* No other exclusions; no item is dropped for its result.

## Secondary and exploratory

Everything else is reported with uncorrected p-values and labelled exploratory. That
includes the `_logit` / `_output` baselines and other sites, the other Exp. 2 contrasts
(`tom_true`, `fact`, `agent_vs_document`), the brand-label `family` variant, the bandit,
steering, and the NARCBench replay.

## Deviations log

| date | change | reason |
|---|---|---|
| | | |
