"""J-space traces: what the lens read out during one agent turn.

Arrays (P = recorded positions, L = recorded layers, K = top-k, W = watch words):

    positions     [P]        absolute token positions (prompt and generated)
    topk_ids      [P, L, K]  top lens tokens at each (position, layer)
    topk_logits   [P, L, K]  their lens logits
    watch_ranks   [P, L, W]  0-based full-vocabulary rank of each watch word (best over
                             its tokenizations); 0 means the word is the lens top-1
    watch_ranks_logit [P, L, W]  same, read with the plain logit lens (baseline; optional)
    output_ranks  [P, W]     same, in the model's own next-token prediction at that
                             position (baseline; optional)
    model_top1    [P]        the model's own next-token argmax (optional)
    residuals     [P, L, D]  opt-in raw block-output residuals, respecting pre/post
    directions_jacobian / directions_logit [L, W, D] opt-in unit decoder rows
    direction_token_ids [W] exact token ids used for the direction rows
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field

import numpy as np

ARRAY_KEYS = ("positions", "topk_ids", "topk_logits", "watch_ranks", "watch_ranks_logit",
              "output_ranks", "model_top1", "residuals", "directions_jacobian",
              "directions_logit", "direction_token_ids")


@dataclass
class JSpaceTrace:
    positions: np.ndarray
    topk_ids: np.ndarray
    topk_logits: np.ndarray
    watch_ranks: np.ndarray
    watch_ranks_logit: np.ndarray | None = None
    output_ranks: np.ndarray | None = None
    model_top1: np.ndarray | None = None
    # metadata copied from the response
    layers: list[int] = field(default_factory=list)
    watch: list[str] = field(default_factory=list)
    watch_ids: list[list[int]] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    spans: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    vocab: dict[int, str] = field(default_factory=dict)
    prompt_len: int = 0
    # New optional fields follow the existing positional constructor arguments.
    residuals: np.ndarray | None = None  # [P, L, D], opt-in block-output capture
    directions_jacobian: np.ndarray | None = None  # [L, W, D], normalized rows
    directions_logit: np.ndarray | None = None
    direction_token_ids: np.ndarray | None = None  # [W], exact direction token ids
    capture_metadata: dict = field(default_factory=dict)
    causal_edits: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------ encoding

    def arrays(self) -> dict[str, np.ndarray]:
        out = {k: getattr(self, k) for k in ARRAY_KEYS if getattr(self, k) is not None}
        return out

    def encode(self, encoding: str = "npz_b64") -> dict | str:
        return encode_arrays(self.arrays(), encoding)

    @classmethod
    def from_response(cls, resp) -> JSpaceTrace | None:
        """Build from a :class:`~jpeers.schemas.GenerateResponse` (or its dict)."""
        r = resp if isinstance(resp, dict) else resp.model_dump()
        if r.get("trace") is None:
            return None
        arrays = decode_arrays(r["trace"], r["trace_encoding"])
        spans: dict[str, list[tuple[int, int]]] = {}
        for s in r["spans"]:
            spans.setdefault(s["label"], []).append((s["start"], s["end"]))
        return cls(
            **{k: arrays.get(k) for k in ARRAY_KEYS},
            layers=list(r["layers"]),
            watch=list(r["watch"]),
            watch_ids=[list(ids) for ids in r.get("watch_ids", [])],
            tokens=list(r["tokens"]),
            spans=spans,
            vocab={int(k): v for k, v in r["vocab"].items()},
            prompt_len=r["prompt_len"],
            capture_metadata=r.get("capture_metadata", {}),
            causal_edits=r.get("causal_edits", []),
        )

    def save(self, path: str) -> None:
        import json

        meta = {"layers": self.layers, "watch": self.watch, "watch_ids": self.watch_ids,
                "tokens": self.tokens,
                "spans": self.spans, "vocab": {str(k): v for k, v in self.vocab.items()},
                "prompt_len": self.prompt_len, "capture_metadata": self.capture_metadata,
                "causal_edits": self.causal_edits}
        np.savez_compressed(path, **self.arrays(), meta=np.array(json.dumps(meta)))

    @classmethod
    def load(cls, path: str) -> JSpaceTrace:
        import json

        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(
            **{k: (z[k] if k in z.files else None) for k in ARRAY_KEYS},
            layers=meta["layers"], watch=meta["watch"], tokens=meta["tokens"],
            watch_ids=meta.get("watch_ids", []),  # absent in traces saved before baselines
            spans={k: [tuple(s) for s in v] for k, v in meta["spans"].items()},
            vocab={int(k): v for k, v in meta["vocab"].items()},
            prompt_len=meta["prompt_len"],
            capture_metadata=meta.get("capture_metadata", {}),
            causal_edits=meta.get("causal_edits", []),
        )

    # ------------------------------------------------------------------ selection

    def mask(self, span: str | tuple[int, int] | None = None, *, generated: bool | None = None
             ) -> np.ndarray:
        """Boolean mask over recorded positions.

        ``span`` is a label (union of its occurrences), an explicit ``(start, end)``, or
        ``None`` for everything. ``generated=True/False`` further restricts to generated
        or prompt positions.
        """
        pos = self.positions
        m = np.ones(len(pos), dtype=bool)
        if span is not None:
            ranges = self.spans.get(span, []) if isinstance(span, str) else [span]
            sm = np.zeros(len(pos), dtype=bool)
            for a, b in ranges:
                sm |= (pos >= a) & (pos < b)
            m &= sm
        if generated is True:
            m &= pos >= self.prompt_len
        elif generated is False:
            m &= pos < self.prompt_len
        return m

    def top_tokens(self, mask: np.ndarray | None = None, k: int | None = None) -> list[str]:
        """Flattened top-k lens tokens (strings) over masked positions and all layers."""
        ids = self.topk_ids if mask is None else self.topk_ids[mask]
        if k is not None:
            ids = ids[..., :k]
        return [self.vocab.get(int(i), f"<{int(i)}>") for i in ids.reshape(-1)]

    def watch_index(self, word: str) -> int:
        return self.watch.index(word)

    @property
    def generated_text_tokens(self) -> list[str]:
        return self.tokens[self.prompt_len:]


def encode_arrays(arrays: dict[str, np.ndarray], encoding: str) -> dict | str:
    if encoding == "json":
        return {k: v.tolist() for k, v in arrays.items()}
    if encoding == "npz_b64":
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    raise ValueError(f"unknown trace encoding {encoding!r}")


def decode_arrays(payload: dict | str, encoding: str) -> dict[str, np.ndarray]:
    if encoding == "json":
        dtypes = {"positions": np.int64, "topk_ids": np.int64, "topk_logits": np.float32,
                  "watch_ranks": np.int64, "watch_ranks_logit": np.int64,
                  "output_ranks": np.int64, "model_top1": np.int64,
                  "residuals": np.float32, "directions_jacobian": np.float32,
                  "directions_logit": np.float32, "direction_token_ids": np.int64}
        return {k: np.asarray(v, dtype=dtypes.get(k)) for k, v in payload.items()}
    if encoding == "npz_b64":
        z = np.load(io.BytesIO(base64.b64decode(payload)), allow_pickle=False)
        return {k: z[k] for k in z.files}
    raise ValueError(f"unknown trace encoding {encoding!r}")
