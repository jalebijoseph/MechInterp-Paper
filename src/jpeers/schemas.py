"""Wire format shared by the inference server and every client backend.

One request = one forward/generation for one agent turn. Requests are stateless: the
client sends the full message history each turn, so the same request can be served
locally, by Azure ML / Foundry, by Vertex AI or by Cloud Run.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- positions


class PositionSelector(BaseModel):
    """Which token positions an intervention (or recording) applies to.

    ``scope``:
        all        -- every prompt and generated position
        prompt     -- prompt positions only
        generated  -- generated positions only
        spans      -- only tokens inside the labelled spans named in ``labels``
                      (see :class:`SpanQuery`), optionally plus generated tokens
        last       -- the final prompt position only (the "decision" position)
    """

    scope: Literal["all", "prompt", "generated", "spans", "last"] = "all"
    labels: list[str] = Field(default_factory=list)
    include_generated: bool = False
    skip_first: int = 0  # ignore attention-sink positions at the very start


class SpanQuery(BaseModel):
    """Ask the server to locate ``text`` inside the rendered prompt and report its token
    span under ``label`` (e.g. ``peer:Bob`` for the peer's last message)."""

    label: str
    text: str
    occurrence: Literal["first", "last", "all"] = "last"


# --------------------------------------------------------------------------- interventions


class InterventionSpec(BaseModel):
    """An edit to the residual stream, applied inside the forward pass on every step.

    kinds:
        ablate_topk    -- project out the ``k`` most active J-lens vectors at each position
                          (paper: "removing its most active contents"). k=5/10/25 are our
                          light/medium/heavy presets.
        ablate_tokens  -- project out the J-lens vectors of specific ``tokens``
        random_ablate  -- control: project out ``k`` random unit directions (fixed ``seed``)
        random_matched -- size-matched control for ``ablate_topk``: at each position,
                          project out one random direction that holds exactly as much of
                          ``h`` as the top-``k`` J-space does, so the edit has the same
                          size and leaves ``h`` the same length
        steer          -- h += strength * ||h|| * v_hat(token) for each of ``tokens``
                          (negative strength suppresses)
        swap           -- coordinate patch: exchange the coefficients of ``source`` and
                          ``target`` in the span of their J-lens vectors
        scale_jspace   -- multiply the component of h inside the top-k J-lens subspace by
                          ``strength`` (0 = ablate, 1 = no-op, >1 = amplify)
    """

    kind: Literal["ablate_topk", "ablate_tokens", "random_ablate", "random_matched", "steer",
                  "swap", "scale_jspace"]
    layers: str | list[int] = "band"  # "band" | "early" | "late" | "all" | explicit list
    positions: PositionSelector = Field(default_factory=PositionSelector)
    k: int = 10
    tokens: list[str] = Field(default_factory=list)
    source: str | None = None
    target: str | None = None
    strength: float = 1.0
    seed: int = 0
    word_like_only: bool = False  # restrict top-k selection to word-like vocab entries


# --------------------------------------------------------------------------- recording


class RecordSpec(BaseModel):
    """What J-space information to return.

    For every recorded (position, layer) the server returns the top-``top_k`` lens
    tokens and their logits, plus the full-vocabulary rank of every ``watch`` word.
    """

    enabled: bool = True
    layers: str | list[int] = "band"
    stride: int = 1  # record every n-th layer of the band
    positions: PositionSelector = Field(default_factory=PositionSelector)
    top_k: int = 10
    watch: list[str] = Field(default_factory=list)
    word_like_only: bool = True  # filter top-k display to word-like tokens
    when: Literal["post", "pre"] = "post"  # read after or before this layer's interventions
    max_positions: int = 4096  # cap on recorded prompt positions (keeps the latest)
    include_model_logits: bool = False
    lens: Literal["jacobian", "logit"] = "jacobian"  # "logit" = logit-lens baseline
    # Baselines recorded alongside every watch rank, at the same positions:
    #   logit  -- the same layer read without the Jacobian (does J add anything?)
    #   output -- rank in the model's own next-token prediction (is the readout just
    #             what the model is about to say?)
    baselines: list[Literal["logit", "output"]] = Field(
        default_factory=lambda: ["logit", "output"])


# --------------------------------------------------------------------------- request / response


class GenerateRequest(BaseModel):
    messages: list[dict] | None = None  # chat messages: {"role", "content"}
    prompt: str | None = None  # raw text alternative (base models, teacher forcing)
    assistant_prefill: str = ""
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    stop: list[str] = Field(default_factory=list)
    record: RecordSpec = Field(default_factory=RecordSpec)
    interventions: list[InterventionSpec] = Field(default_factory=list)
    spans: list[SpanQuery] = Field(default_factory=list)
    trace_encoding: Literal["json", "npz_b64"] = "npz_b64"


class TokenSpan(BaseModel):
    label: str
    start: int
    end: int  # exclusive


class GenerateResponse(BaseModel):
    model: str
    text: str
    prompt_len: int
    n_generated: int
    tokens: list[str]
    spans: list[TokenSpan]
    layers: list[int]
    watch: list[str]
    watch_ids: list[list[int]]
    vocab: dict[int, str]  # decoded strings for every token id appearing in the trace
    trace_encoding: str
    trace: dict | str | None  # see jpeers.trace
    interventions_applied: int
    timings: dict[str, float]
