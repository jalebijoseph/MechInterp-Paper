"""Causal experiments on one SAE feature: what causes it, and what it causes.

Upstream ("what caused this activation?"):
- `token_attribution`: mean-ablate each input token's embedding, one at a
  time, and measure how much the feature's activation at its peak
  position drops. Big drop = that token is a cause.
- `component_attribution`: mean-ablate each attention head and each MLP
  in layers 0..L, one at a time, and measure the same drop. Shows which
  parts of the model compute the feature.

Downstream ("what does this feature do?"):
- `output_effects`: ablate the feature (or steer along its decoder
  direction) at its layer and measure the change in the model's
  next-token predictions (KL divergence, most promoted/suppressed
  tokens), against random features as a control.
- `decoder_logit_lens`: the tokens the feature's decoder direction
  points at through the unembedding (cheap, rough, no forward pass).

All edits keep the SAE's reconstruction error: they subtract or add only
the feature's own contribution (activation × decoder row) instead of
replacing the residual stream with the SAE reconstruction, so the
unedited baseline is exactly the original model.

Ablations use the prompt's own mean (over non-BOS positions), not zero,
to stay closer to the model's normal activation range; the BOS position
is never edited.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Pure helpers (numpy only; unit-tested without a model)
# ---------------------------------------------------------------------------


def firing_frequency(token_acts: list) -> np.ndarray:
    """Fraction of (non-BOS) tokens on which each feature is active."""
    n_features = token_acts[0].n_features
    counts = np.zeros(n_features, dtype=np.int64)
    n_tokens = 0
    for acts in token_acts:
        n_tokens += len(acts.tokens)
        np.add.at(counts, acts.features, 1)
    return counts / max(n_tokens, 1)


def matched_random_features(
    freq: np.ndarray, target: int, n: int, rng: np.random.Generator, exclude: set[int] = frozenset()
) -> list[int]:
    """`n` random features whose firing frequency is close to `target`'s.

    Samples from the 5n features nearest in frequency (excluding `target`,
    `exclude`, and features that never fire), so the control features are
    about as active as the one being tested.
    """
    candidates = np.array(
        [i for i in range(len(freq)) if i != target and i not in exclude and freq[i] > 0]
    )
    if len(candidates) == 0:
        return []
    order = np.argsort(np.abs(freq[candidates] - freq[target]))
    pool = candidates[order[: max(5 * n, n)]]
    return [int(i) for i in rng.choice(pool, size=min(n, len(pool)), replace=False)]


def components_for_layer(layer: int, n_heads: int) -> list[tuple[int, str, int | None]]:
    """Every attention head and MLP in layers 0..layer, as (layer, kind, head)."""
    comps: list[tuple[int, str, int | None]] = []
    for lyr in range(layer + 1):
        comps.extend((lyr, "head", h) for h in range(n_heads))
        comps.append((lyr, "mlp", None))
    return comps


def component_name(comp: tuple[int, str, int | None]) -> str:
    lyr, kind, head = comp
    return f"L{lyr}H{head}" if kind == "head" else f"L{lyr}MLP"


# ---------------------------------------------------------------------------
# Model-dependent pieces
# ---------------------------------------------------------------------------


def _sae_dtype(sae):
    return next(sae.parameters()).dtype


def _decode_token(model, tok_id: int) -> str:
    if getattr(model, "tokenizer", None) is None:
        return str(tok_id)
    return model.tokenizer.decode([tok_id])


def feature_acts(model, sae, tokens, layer: int, feature_id: int, fwd_hooks=()) -> np.ndarray:
    """Feature activation at every position: (batch, pos). Stops at `layer`."""
    import torch

    hook_name = f"blocks.{layer}.hook_resid_post"
    store = {}

    def capture(resid, hook):
        store["resid"] = resid

    with torch.inference_mode():
        model.run_with_hooks(tokens, fwd_hooks=[*fwd_hooks, (hook_name, capture)], stop_at_layer=layer + 1)
        resid = store["resid"].to(_sae_dtype(sae))
        return sae.encode(resid)[..., feature_id].float().cpu().numpy()


@dataclass
class TokenAttribution:
    prompt: str
    target_pos: int
    target_token: str
    clean_act: float
    tokens: list[str]  # source tokens 1..target_pos
    drops: list[float]  # clean_act - ablated act, per source token


def token_attribution(model, sae, layer: int, feature_id: int, prompt: str) -> TokenAttribution | None:
    """Which input tokens cause the feature's peak activation in `prompt`.

    Returns None if the feature doesn't fire on the prompt.
    """
    import torch

    tokens = model.to_tokens(prompt)
    store = {}

    def capture_embed(x, hook):
        store["embed"] = x.clone()

    clean = feature_acts(model, sae, tokens, layer, feature_id, [("hook_embed", capture_embed)])[0]
    if len(clean) < 2 or clean[1:].max() <= 0:
        return None
    t = int(np.argmax(clean[1:])) + 1
    mean_embed = store["embed"][0, 1:].mean(dim=0)

    sources = list(range(1, t + 1))  # causal attention: only positions <= t can matter

    def ablate_embed(x, hook):
        for row, pos in enumerate(sources):
            x[row, pos] = mean_embed
        return x

    batch = tokens.repeat(len(sources), 1)
    ablated = feature_acts(model, sae, batch, layer, feature_id, [("hook_embed", ablate_embed)])[:, t]
    ids = tokens[0].tolist()
    return TokenAttribution(
        prompt=prompt,
        target_pos=t,
        target_token=_decode_token(model, ids[t]),
        clean_act=float(clean[t]),
        tokens=[_decode_token(model, ids[p]) for p in sources],
        drops=[float(clean[t] - a) for a in ablated],
    )


def _component_hooks(assignments, means):
    """Hooks that mean-ablate one component per batch row (BOS untouched)."""
    by_hook = defaultdict(list)
    for row, (lyr, kind, head) in assignments:
        name = f"blocks.{lyr}.attn.hook_z" if kind == "head" else f"blocks.{lyr}.hook_mlp_out"
        by_hook[name].append((row, head))

    hooks = []
    for name, items in by_hook.items():
        def fn(x, hook, items=items, name=name):
            for row, head in items:
                if head is None:
                    x[row, 1:] = means[name]
                else:
                    x[row, 1:, head] = means[name][head]
            return x

        hooks.append((name, fn))
    return hooks


def component_attribution(
    model, sae, layer: int, feature_id: int, prompt: str, batch_size: int = 32
) -> dict[str, float] | None:
    """Drop in the feature's peak activation when each head / MLP is mean-ablated.

    Positive = the component helps cause the activation; negative = it
    suppresses it. Returns None if the feature doesn't fire on the prompt.
    """
    tokens = model.to_tokens(prompt)
    comps = components_for_layer(layer, model.cfg.n_heads)
    names = {f"blocks.{l}.attn.hook_z" for l in range(layer + 1)} | {
        f"blocks.{l}.hook_mlp_out" for l in range(layer + 1)
    }
    store = {}

    def capture(x, hook):
        store[hook.name] = x[0, 1:].mean(dim=0).clone()  # mean over non-BOS positions

    clean = feature_acts(model, sae, tokens, layer, feature_id, [(n, capture) for n in names])[0]
    if len(clean) < 2 or clean[1:].max() <= 0:
        return None
    t = int(np.argmax(clean[1:])) + 1

    drops: dict[str, float] = {}
    for start in range(0, len(comps), batch_size):
        chunk = comps[start : start + batch_size]
        hooks = _component_hooks(list(enumerate(chunk)), store)
        ablated = feature_acts(model, sae, tokens.repeat(len(chunk), 1), layer, feature_id, hooks)[:, t]
        for comp, a in zip(chunk, ablated):
            drops[component_name(comp)] = float(clean[t] - a)
    return drops


def ablate_feature_hook(sae, feature_id: int):
    """Subtract the feature's own contribution (act × decoder row); keep SAE error."""
    w = sae.W_dec[feature_id]

    def fn(resid, hook):
        acts = sae.encode(resid.to(w.dtype))[..., feature_id : feature_id + 1]
        out = resid.clone()
        out[:, 1:] = resid[:, 1:] - (acts[:, 1:] * w).to(resid.dtype)
        return out

    return fn


def add_vector_hook(vec):
    """Add a fixed vector to the residual stream at every non-BOS position."""

    def fn(resid, hook):
        out = resid.clone()
        out[:, 1:] = resid[:, 1:] + vec.to(resid.dtype)
        return out

    return fn


@dataclass
class OutputEffect:
    kl: float  # mean KL(clean || edited) over non-BOS positions
    delta: np.ndarray  # mean change in log-prob per vocab token, over non-BOS positions


def output_effects(model, layer: int, prompt: str, hook_fns: dict) -> dict[str, OutputEffect]:
    """Effect of each residual-stream edit at `layer` on next-token predictions.

    `hook_fns` maps a name to a hook for blocks.{layer}.hook_resid_post;
    the clean forward pass is shared across all of them.
    """
    import torch

    tokens = model.to_tokens(prompt)
    hook_name = f"blocks.{layer}.hook_resid_post"
    out = {}
    with torch.inference_mode():
        clean = model(tokens)[0, 1:].float().log_softmax(-1)
        for name, fn in hook_fns.items():
            edited = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, fn)])[0, 1:].float().log_softmax(-1)
            out[name] = OutputEffect(
                kl=(clean.exp() * (clean - edited)).sum(-1).mean().item(),
                delta=(edited - clean).mean(0).cpu().numpy(),
            )
    return out


def top_tokens(model, scores: np.ndarray, top_n: int = 10, largest: bool = True) -> list[tuple[str, float]]:
    """Highest (or lowest) scoring vocab tokens, decoded."""
    order = np.argsort(scores)
    idx = order[::-1][:top_n] if largest else order[:top_n]
    return [(_decode_token(model, int(i)), float(scores[i])) for i in idx]


def decoder_logit_lens(model, sae, feature_id: int, top_n: int = 10) -> list[tuple[str, float]]:
    """Tokens the feature's decoder direction points at through the unembedding.

    Skips the final layer norm, so this is a rough guide — more meaningful
    for late layers than early ones.
    """
    import torch

    with torch.inference_mode():
        w = sae.W_dec[feature_id].to(model.W_U.dtype)
        scores = (w @ model.W_U).float()
        top = torch.topk(scores, top_n)
    return [(_decode_token(model, int(i)), float(v)) for v, i in zip(top.values, top.indices)]
