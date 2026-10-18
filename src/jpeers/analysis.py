"""Metrics over J-space traces.

Conventions (following the paper's experiment README): a word is a *hit* at a position
if it is the lens top-1 (rank 0) at any band layer; we also report softer rank
thresholds and the reciprocal rank, which degrade gracefully for small models.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

import numpy as np

from jpeers.trace import JSpaceTrace

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")
STOPWORDS = set("""a an the and or but if of to in on at for with by from as is are was were be
been being it its this that these those i you he she we they me him her us them my your our their
not no do does did have has had will would can could should may might must so than then there here
what which who whom whose when where why how all any each few more most other some such only own same
too very just also into over under about after before again further once up down out off""".split())


def norm_token(s: str) -> str:
    return s.strip().lower()


def content_words(text: str, *, min_len: int = 3) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if len(w) >= min_len
            and w.lower() not in STOPWORDS}


# --------------------------------------------------------------------------- watch-word ranks


LENSES = ("jlens", "logit", "output")


def word_ranks(trace: JSpaceTrace, word: str, mask: np.ndarray | None = None,
               lens: str = "jlens") -> np.ndarray:
    """Rank of ``word`` at each masked position -> ``[n_positions]``.

    ``lens``: ``jlens`` (best over band layers), ``logit`` (the same layers read without
    the Jacobian; baseline) or ``output`` (the model's own next-token prediction there;
    baseline). Baselines missing from the trace give NaN ranks."""
    w = trace.watch_index(word)
    if lens == "output":
        if trace.output_ranks is None:
            return np.full(int(mask.sum()) if mask is not None else len(trace.positions),
                           np.nan)
        ranks = trace.output_ranks[:, w]
        return ranks[mask] if mask is not None else ranks
    arr = trace.watch_ranks if lens == "jlens" else trace.watch_ranks_logit
    if arr is None:
        return np.full(int(mask.sum()) if mask is not None else len(trace.positions), np.nan)
    ranks = arr[:, :, w]
    if mask is not None:
        ranks = ranks[mask]
    return ranks.min(axis=1) if ranks.size else np.zeros(0, dtype=int)


def group_ranks(trace: JSpaceTrace, words: Iterable[str], mask: np.ndarray | None = None,
                lens: str = "jlens") -> np.ndarray:
    """Best rank of any word in ``words`` at each masked position."""
    words = [w for w in words if w in trace.watch]
    if not words:
        return np.zeros(0, dtype=int)
    return np.min(np.stack([word_ranks(trace, w, mask, lens) for w in words]), axis=0)


def upcoming_mask(trace: JSpaceTrace, words: Iterable[str], horizon: int = 3) -> np.ndarray:
    """``[P]`` bool: positions where one of ``words`` is among the next ``horizon`` actual
    tokens (prompt or generated). There, a high readout may just be the model about to
    say the word; ``mask & ~upcoming_mask(...)`` keeps the positions where it is not."""
    forms = {trace.vocab[i] for w in words if w in trace.watch
             for i in trace.watch_ids[trace.watch_index(w)] if i in trace.vocab}
    toks = trace.tokens
    out = np.zeros(len(trace.positions), dtype=bool)
    for j, p in enumerate(trace.positions):
        out[j] = any(t in forms for t in toks[p + 1:p + 1 + horizon])
    return out


def lens_mrrs(trace: JSpaceTrace, words: Iterable[str], mask: np.ndarray, prefix: str,
              horizon: int = 3) -> dict[str, float]:
    """MRR of the best of ``words`` over ``mask`` under every readout, plus the J-lens
    MRR restricted to positions where none of the words is about to be said:

        {prefix}           J-lens
        {prefix}_logit     logit-lens baseline (same layers, no Jacobian)
        {prefix}_output    output baseline (the model's own next-token prediction)
        {prefix}_beyond    J-lens, excluding positions where a word is upcoming

    The J-lens shows something beyond the output only if ``{prefix}`` clearly exceeds
    ``{prefix}_output`` and ``{prefix}_beyond`` stays well above chance."""
    words = list(words)
    beyond = mask & ~upcoming_mask(trace, words, horizon)
    out = {prefix: rank_summary(group_ranks(trace, words, mask))["mrr"]}
    for lens in ("logit", "output"):
        out[f"{prefix}_{lens}"] = rank_summary(group_ranks(trace, words, mask, lens))["mrr"]
    out[f"{prefix}_beyond"] = rank_summary(group_ranks(trace, words, beyond))["mrr"]
    return out


def rank_summary(ranks: np.ndarray, *, thresholds=(0, 9, 99)) -> dict[str, float]:
    """Summaries of a per-position rank vector (NaN ranks = baseline not recorded)."""
    if ranks.size == 0 or np.isnan(ranks.astype(float)).all():
        return {"n": 0, "best": float("nan"), "mrr": float("nan"),
                **{f"hit@{t + 1}": float("nan") for t in thresholds}}
    out = {"n": int(ranks.size), "best": int(ranks.min()),
           "mrr": float(np.mean(1.0 / (ranks + 1.0)))}
    for t in thresholds:
        out[f"hit@{t + 1}"] = float(np.mean(ranks <= t))
    out["any_hit@1"] = float(ranks.min() == 0)
    return out


def layer_profile(trace: JSpaceTrace, words: Iterable[str], mask: np.ndarray | None = None
                  ) -> dict[int, float]:
    """Mean reciprocal rank per layer (best word in group) -- where in depth the signal is."""
    words = [w for w in words if w in trace.watch]
    if not words:
        return {}
    idx = [trace.watch_index(w) for w in words]
    ranks = trace.watch_ranks[:, :, idx].min(axis=2)  # [P, L]
    if mask is not None:
        ranks = ranks[mask]
    return {layer: float(np.mean(1.0 / (ranks[:, i] + 1.0))) if len(ranks) else float("nan")
            for i, layer in enumerate(trace.layers)}


# --------------------------------------------------------------------------- bags of J-space tokens


def jspace_bag(trace: JSpaceTrace, mask: np.ndarray | None = None, k: int = 5) -> Counter:
    """Counts of normalised top-``k`` lens tokens over masked positions and band layers."""
    toks = (norm_token(t) for t in trace.top_tokens(mask, k=k))
    return Counter(t for t in toks if len(t) >= 2 and t not in STOPWORDS)


def bag_similarity(a: Counter, b: Counter) -> dict[str, float]:
    keys = set(a) | set(b)
    if not keys:
        return {"jaccard": float("nan"), "cosine": float("nan")}
    va = np.array([a.get(k, 0) for k in keys], dtype=float)
    vb = np.array([b.get(k, 0) for k in keys], dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return {"jaccard": len(set(a) & set(b)) / len(keys),
            "cosine": float(va @ vb / denom) if denom else float("nan")}


def _matches(token: str, words: set[str]) -> bool:
    if token in words:
        return True
    return len(token) >= 4 and any(w.startswith(token) for w in words)


def content_uptake(trace: JSpaceTrace, mask: np.ndarray, source_text: str,
                   exclude_text: str = "", k: int = 5) -> dict[str, float]:
    """How much of the J-space at ``mask`` is made of words that appear in ``source_text``
    (e.g. the peer's message) and not in ``exclude_text`` (the agent's own context)."""
    words = content_words(source_text) - content_words(exclude_text)
    bag = jspace_bag(trace, mask, k=k)
    total = sum(bag.values())
    hits = {t: c for t, c in bag.items() if _matches(t, words)}
    return {"uptake": (sum(hits.values()) / total) if total else float("nan"),
            "distinct": len(hits), "n_source_words": len(words),
            "top": sorted(hits, key=hits.get, reverse=True)[:10]}


# --------------------------------------------------------------------------- behaviour


def mentions(text: str, words: Iterable[str]) -> bool:
    low = text.lower()
    return any(re.search(rf"\b{re.escape(w.lower())}\b", low) for w in words)


def bootstrap_ci(values: Iterable[float], n: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(
        np.quantile(means, 1 - alpha / 2))


def paired_permutation_test(a: Iterable[float], b: Iterable[float], n: int = 5000,
                            seed: int = 0) -> float:
    """Two-sided sign-flip p-value for the mean of paired differences ``a[i] - b[i]``
    (e.g. the same fact or task under two conditions). Pairs with a NaN are dropped."""
    d = np.asarray(list(a), dtype=float) - np.asarray(list(b), dtype=float)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, d.size))
    count = int((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).sum())
    return (count + 1) / (n + 1)


def permutation_test(a: Iterable[float], b: Iterable[float], n: int = 5000, seed: int = 0
                     ) -> float:
    """Two-sided p-value for a difference in means."""
    a = np.asarray([v for v in a if not np.isnan(v)], dtype=float)
    b = np.asarray([v for v in b if not np.isnan(v)], dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    count = 0
    for _ in range(n):
        rng.shuffle(pooled)
        count += abs(pooled[: a.size].mean() - pooled[a.size:].mean()) >= obs
    return (count + 1) / (n + 1)
