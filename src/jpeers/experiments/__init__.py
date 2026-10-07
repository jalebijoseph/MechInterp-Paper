"""Experiment runners. Each module exposes ``run(cfg: dict) -> dict`` (summary) and
writes ``runs/<experiment>/<run_id>/{config.json, turns.jsonl, traces/, summary.json,
summary.md}``.

    tom_validation    benchmarks: does the lens see belief states? (false belief, Hi-ToM)
    collab            Exp. 1: two agents splitting a task; peer-awareness in J-space
    imposter          Exp. 2: one agent asserts falsehoods; suspicion in J-space
                      (+ NARCBench transcript replay)
    peer_selection    Exp. 3: orchestrator delegation, family bias, MACE-style lock-in
    ablation          Exp. 4: re-run any of the above under J-space interventions
"""

EXPERIMENTS = ["tom_validation", "collab", "imposter", "peer_selection", "ablation"]
