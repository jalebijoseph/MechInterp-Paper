"""Named J-space intervention presets (Experiment 4 and ad-hoc runs).

Every preset is a list of :class:`~jpeers.schemas.InterventionSpec` applied inside the
forward pass of the *subject* agent while it reads its prompt and writes its reply.
Presets can be parameterised with ``{peer}``, ``{self}`` and ``{words}`` placeholders,
filled in by :func:`resolve` from the experiment's context.

    none                 no edit (baseline)
    jspace_light/medium/heavy
                         remove the top k = 5 / 10 / 25 J-lens vectors at every position
                         across the workspace band (paper: "deleting the J-space")
    matched_k5/k10/k25   size-matched control (Exp. 4 default): at each position, project
                         out a random direction holding exactly as much of the state as the
                         top-k J-space did, so only *which* content is removed differs
    random_k5/k10/k25    unmatched control: project out k fixed random directions (removes
                         far less than J-space ablation; kept for comparison only)
    jspace_peer_span     remove the top-10 J-lens vectors only while reading the peer's
                         last message (does the model need J-space to *model the peer*?)
    jspace_generated     remove the top-10 only on generated tokens (Exp. 4 main condition)
    matched_generated / matched_peer_span
                         size-matched controls on the same positions as jspace_generated /
                         jspace_peer_span
    jspace_early / jspace_late
                         top-10 removal restricted to the early / late workspace band
    halve_jspace / double_jspace
                         scale the top-10 J-space component by 0.5 / 2.0
    suppress_peer        project out the J-lens vectors of the peer's name
    suppress_words       project out the J-lens vectors of ``{words}``
    steer_suspicion      add +0.005 x median residual norm (at every band layer) along "suspicion"/"dishonest"
    steer_trust          add +0.01 along "trustworthy"/"reliable"
    steer_teamwork       add +0.01 along "cooperation"/"joint"
                         (strengths: STEER_STRENGTH, from scripts/steer_sweep.py)
    swap_peer_self       swap the J-lens coordinates of the peer's and the agent's names

Steering words are never measured words (tests/test_watch_words.py checks this against
every experiment's lexicon): steering toward a word and then counting that same word in
J-space would show an increase by construction. Judge steering by behaviour; a rise in
the measured lexicon is only a check that the edit reached the intended concept.
"""

from __future__ import annotations

from jpeers.schemas import InterventionSpec, PositionSelector

_ALL = PositionSelector(scope="all", skip_first=4)

STEER_WORDS = {
    "suspicion": ["suspicion", "dishonest"],
    "trust": ["trustworthy", "reliable"],
    "teamwork": ["cooperation", "joint"],
}

# x median residual norm, added at every band layer. From scripts/steer_sweep.py on
# Qwen3-1.7B; re-sweep per model (the usable value scales with model and band width).
STEER_STRENGTH = {
    "suspicion": 0.005,  # 0.01 makes the reviewer echo the peer's draft
    "trust": 0.01,
    "teamwork": 0.01,
}


def _topk(k: int, **kw) -> InterventionSpec:
    return InterventionSpec(kind="ablate_topk", k=k, positions=kw.pop("positions", _ALL), **kw)


PRESETS: dict[str, list[dict]] = {
    "none": [],
    "jspace_light": [_topk(5).model_dump()],
    "jspace_medium": [_topk(10).model_dump()],
    "jspace_heavy": [_topk(25).model_dump()],
    "random_k5": [InterventionSpec(kind="random_ablate", k=5, positions=_ALL).model_dump()],
    "random_k10": [InterventionSpec(kind="random_ablate", k=10, positions=_ALL).model_dump()],
    "random_k25": [InterventionSpec(kind="random_ablate", k=25, positions=_ALL).model_dump()],
    "matched_k5": [InterventionSpec(kind="random_matched", k=5, positions=_ALL).model_dump()],
    "matched_k10": [InterventionSpec(kind="random_matched", k=10, positions=_ALL).model_dump()],
    "matched_k25": [InterventionSpec(kind="random_matched", k=25, positions=_ALL).model_dump()],
    "jspace_peer_span": [_topk(10, positions=PositionSelector(
        scope="spans", labels=["peer_last"])).model_dump()],
    "jspace_generated": [_topk(10, positions=PositionSelector(scope="generated")).model_dump()],
    # size- and position-matched controls for the two Exp. 4 conditions
    "matched_generated": [InterventionSpec(
        kind="random_matched", k=10,
        positions=PositionSelector(scope="generated")).model_dump()],
    "matched_peer_span": [InterventionSpec(
        kind="random_matched", k=10,
        positions=PositionSelector(scope="spans", labels=["peer_last"])).model_dump()],
    "jspace_early": [_topk(10, layers="early").model_dump()],
    "jspace_late": [_topk(10, layers="late").model_dump()],
    "halve_jspace": [InterventionSpec(kind="scale_jspace", k=10, strength=0.5,
                                      positions=_ALL).model_dump()],
    "double_jspace": [InterventionSpec(kind="scale_jspace", k=10, strength=2.0,
                                       positions=_ALL).model_dump()],
    "suppress_peer": [InterventionSpec(kind="ablate_tokens", tokens=["{peer}"],
                                       positions=_ALL).model_dump()],
    "suppress_words": [InterventionSpec(kind="ablate_tokens", tokens=["{words}"],
                                        positions=_ALL).model_dump()],
    **{f"steer_{name}": [InterventionSpec(kind="steer", tokens=words,
                                          strength=STEER_STRENGTH[name],
                                          positions=_ALL).model_dump()]
       for name, words in STEER_WORDS.items()},
    "swap_peer_self": [InterventionSpec(kind="swap", source="{self}", target="{peer}",
                                        positions=_ALL).model_dump()],
}


def resolve(spec: str | list | None, *, peer: str = "", self_name: str = "",
            words: list[str] | None = None) -> list[InterventionSpec]:
    """Turn a preset name, or a list of preset names / raw spec dicts, into specs with
    placeholders filled."""
    if spec is None:
        return []
    items = [spec] if isinstance(spec, str) else list(spec)
    out: list[InterventionSpec] = []
    for item in items:
        raw_list = PRESETS[item] if isinstance(item, str) else [item]
        for raw in raw_list:
            raw = dict(raw)
            toks: list[str] = []
            for t in raw.get("tokens", []):
                if t == "{words}":
                    toks += list(words or [])
                else:
                    toks.append(t.replace("{peer}", peer).replace("{self}", self_name))
            raw["tokens"] = toks
            for key in ("source", "target"):
                if raw.get(key):
                    raw[key] = raw[key].replace("{peer}", peer).replace("{self}", self_name)
            out.append(InterventionSpec.model_validate(raw))
    return out
