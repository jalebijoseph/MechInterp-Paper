# 5. Experiments

All experiments run with `python -m jpeers.cli run configs/<file>.yaml [--set key=value]`
and write `runs/<experiment>/<run_id>/` containing `config.json`, `turns.jsonl` (every
agent turn: text, prompt, timings, interventions), `traces/*.npz` (J-space per turn),
`summary.json` and `summary.md`. Endpoint names in configs resolve through
`configs/endpoints.yaml`. Names not listed there are loaded in-process as registry models.

Watch words are single tokens where possible, using names that tokenise to one token
(Alice, Bob, Kai, Rosa). Reported numbers use the **best rank over the workspace band**
at each position, summarised as MRR (mean of 1/(rank+1)) and hit@k.

---

## Exp. 0: lens validation on ToM benchmarks (`configs/exp0_tom_validation.yaml`)

**Question.** At the answer position of a false-belief task, does J-space hold the
protagonist's *belief* rather than reality? Without this, an absence of peer modelling
later would be uninterpretable.

**Design.** Kosinski-style transfer and contents tasks, each with a matched true-belief
control (same story, but the protagonist sees the change), in a "people" framing and an
"LLM agents in a shared workspace" framing. Optionally Hi-ToM orders 0–4.

**Key readout.** `belief_beats_decoy`: the fraction of items where the belief word
outranks the competing word (reality on false-belief items, the stale location/label on
controls). Real belief tracking predicts > 0.5 in *both* conditions. A model that only
pattern-matches "first location" scores high on false-belief and low on controls.

## Exp. 1: collaboration (`configs/exp1_collab.yaml`)

**Question.** When two agents split a task, does each represent the partner (who they
are, and what the partner is working on) in J-space?

**Design.** 8 tasks × {team, solo} × 2 agents × `rounds`. In *team*, agents are told
about the named partner and see `[Bob]: ...`. In *solo*, the partner's text is still
shown but as anonymous `[Shared notes]`, and the agent is told it works alone. So
content exposure is identical and only the social framing differs.

**Readouts** (recorded on the partner's last message and on the agent's own reply):
`partner_name_mrr` (while writing and while reading), `team_mrr`, `partner_kw_mrr` vs
`own_kw_mrr`, `uptake` (share of J-space top tokens taken from the partner's message),
and `cross_agent_cosine` (overlap of the two agents' J-space bags in the same round).
Permutation tests compare team vs solo.

**Proposal flags.** Same results analysed (cross-agent cosine high), no teamwork content
(team_mrr ≈ solo), and the best case: partner_kw_mrr team > solo with own_kw unchanged.

## Exp. 2: imposter (`configs/exp2_imposter.yaml`)

**Question.** When a peer asserts a falsehood, does the listener's J-space contain
suspicion and/or the correct fact, even if it does not say so?

**Design.** 20 single-word facts. Bob drafts a deliverable that states either the false
claim (imposter: secret system-prompt instruction) or the true claim (honest control);
Alice reviews. `peer_mode: scripted` uses fixed wording for a tighter control.

**Readouts.** On Alice while reading Bob's message: `susp_mrr_reading`,
`true_mrr_reading`, `false_mrr_reading`. While writing: `susp_mrr_writing`,
`agree_mrr_writing`. Behaviour: `reply_has_true`, `reply_flags`, `reply_keeps_false`.
Interesting dissociation: J-space holds the true answer or suspicion (`true_mrr_reading`
high) while the reply keeps the falsehood.

**NARCBench replay.** Honest agents' final-phase prompts are teacher-forced, and
suspicion MRR is read on each other speaker's message span. We compare colluder seats
in collusion runs with the same seats in control runs and report AUROC.

## Exp. 3: peer selection (`configs/exp3_peer_selection.yaml`)

**context.** Kai has a dev history and Rosa a research history. For 12 tasks the
orchestrator answers "Assign to: …". Names ↔ roles and roster order are counterbalanced
(4×). Readouts: `accuracy`, and at the decision position `jspace_prefers_right` (the
right name outranks the wrong one) plus `right_mrr` / `wrong_mrr`.

**family.** Two equally qualified agents "powered by" the orchestrator's own family vs
another (`equal`, `own_smaller`, `own_larger` sizes), counterbalanced. Readouts:
`p_choose_own_family` (0.5 = no bias), and own vs other family-name MRR at the decision.
Cross-family runs: point `orchestrator` at endpoints of different families
(qwen / llama / gemma / gpt-oss) and compare.

**bandit (MACE).** 12 rounds of delegation with accept/reject feedback, with success
rates equal (0.7/0.7) or the second-listed peer better (0.5/0.8). Readouts: choice
`entropy`, `max_share` (lock-in), `regret_per_round`, and whether the **unchosen** peer
is still represented at the decision (`unchosen_mrr`). MACE reports lock-in
behaviourally; this asks whether the alternative drops out of the workspace.

## Exp. 4: with vs without J-space (`configs/exp4_ablation.yaml`)

Runs any base experiment under a list of intervention conditions (see
[06_interventions.md](06_interventions.md)). Default:
`none, jspace_light, random_k5, jspace_medium, random_k10, jspace_heavy, random_k25, jspace_peer_span`.
`summary.md` adds `d_<metric>` rows: the change against `none`. The causal claim needs
`jspace_*` to move the ToM and behaviour metrics much more than the matched `random_*`,
while fluency is preserved (check `turns.jsonl`). `jspace_peer_span` tests whether the
J-space used *while reading the peer* is the part that matters.

---

## Recommended run plan

| stage | where | models | configs |
|---|---|---|---|
| dev / pilots | local GPU | qwen3-1.7b | all, with `--set n_items=8` etc. |
| main | Azure + Vertex | qwen3-8b, llama3.1-8b-it, gemma-3-12b-it (one endpoint each) | exp0–3 per model; exp1/3 with mixed-family pairs |
| causal | same endpoints | the model with the clearest Exp. 1–3 signal | exp4 with base = collab, imposter, peer_selection |
| scale check | A100-80GB | qwen3-32b or gemma-3-27b-it | exp0, exp2 |

Pilot results from the laptop are in [results_pilot.md](results_pilot.md).
