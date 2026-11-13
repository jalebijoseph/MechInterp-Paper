"""Models with pre-fitted Jacobian lenses.

Pre-fitted lenses are hosted by Neuronpedia at ``neuronpedia/jacobian-lens`` on the
HuggingFace Hub (1000 WikiText prompts, fitted with the reference ``jlens`` code). Any
other HF decoder can be used by fitting a lens with ``scripts/fit_lens.py`` and pointing
``lens_path`` at the result.

``band`` is the workspace layer band as fractions of depth. The paper reindexes layers
to 0-100 and places the workspace at roughly 38-92 (early 38-58, late 62-92); we use
that as the default for every model and recommend checking it per model with
``scripts/lens_sanity.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

NEURONPEDIA_REPO = "neuronpedia/jacobian-lens"

DEFAULT_BAND = (0.38, 0.92)
BAND_PRESETS: dict[str, tuple[float, float]] = {
    "band": DEFAULT_BAND,
    "early": (0.38, 0.58),
    "late": (0.62, 0.92),
    "all": (0.0, 1.0),
}


@dataclass(frozen=True)
class ModelSpec:
    """A model and where its lens lives.

    Attributes:
        name: Short key used throughout configs.
        hf_id: HuggingFace model id.
        family: Model family label (used by the peer-selection family-bias experiment).
        params_b: Parameter count in billions (shown to orchestrators in Exp. 3).
        lens_repo / lens_file / lens_revision: Hub location of the fitted lens.
        lens_path: Local lens file; overrides the Hub location when set.
        chat: Whether the model has a chat template (instruction-tuned).
        band: Workspace band as (start, end) fractions of depth.
        min_gpu_gb: Rough bf16 memory needed to serve it.
        template_kwargs: Extra kwargs for ``apply_chat_template``.
    """

    name: str
    hf_id: str
    family: str
    params_b: float
    lens_file: str | None = None
    lens_repo: str = NEURONPEDIA_REPO
    lens_revision: str | None = None
    lens_path: str | None = None
    chat: bool = True
    band: tuple[float, float] = DEFAULT_BAND
    min_gpu_gb: float = 0.0
    template_kwargs: dict = field(default_factory=dict)


def _np(folder: str, stem: str) -> str:
    return f"{folder}/jlens/Salesforce-wikitext/{stem}_jacobian_lens.pt"


_QWEN3_KW = {"enable_thinking": False}

MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in [
        # --- Qwen ---
        ModelSpec("qwen3-1.7b", "Qwen/Qwen3-1.7B", "qwen", 1.7, _np("qwen3-1.7b", "Qwen3-1.7B"),
                  min_gpu_gb=5, template_kwargs=_QWEN3_KW),
        ModelSpec("qwen3-4b", "Qwen/Qwen3-4B", "qwen", 4, _np("qwen3-4b", "Qwen3-4B"),
                  min_gpu_gb=10, template_kwargs=_QWEN3_KW),
        ModelSpec("qwen3-8b", "Qwen/Qwen3-8B", "qwen", 8, _np("qwen3-8b", "Qwen3-8B"),
                  min_gpu_gb=20, template_kwargs=_QWEN3_KW),
        ModelSpec("qwen3-14b", "Qwen/Qwen3-14B", "qwen", 14, _np("qwen3-14b", "Qwen3-14B"),
                  min_gpu_gb=32, template_kwargs=_QWEN3_KW),
        ModelSpec("qwen3-32b", "Qwen/Qwen3-32B", "qwen", 32, _np("qwen3-32b", "Qwen3-32B"),
                  min_gpu_gb=70, template_kwargs=_QWEN3_KW),
        ModelSpec("qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B", "qwen", 0.8,
                  _np("qwen3.5-0.8b", "Qwen3.5-0.8B"), min_gpu_gb=3, template_kwargs=_QWEN3_KW),
        ModelSpec("qwen3.5-2b", "Qwen/Qwen3.5-2B", "qwen", 2, _np("qwen3.5-2b", "Qwen3.5-2B"),
                  min_gpu_gb=6, template_kwargs=_QWEN3_KW),
        ModelSpec("qwen3.5-4b", "Qwen/Qwen3.5-4B", "qwen", 4,
                  _np("qwen3.5-4b", "Qwen3.5-4B"), min_gpu_gb=10, template_kwargs=_QWEN3_KW),
        ModelSpec("qwen3.5-27b", "Qwen/Qwen3.5-27B", "qwen", 27,
                  _np("qwen3.5-27b", "Qwen3.5-27B"), min_gpu_gb=60, template_kwargs=_QWEN3_KW),
        ModelSpec("qwen2.5-7b-it", "Qwen/Qwen2.5-7B-Instruct", "qwen", 7,
                  _np("qwen2.5-7b-it", "Qwen2.5-7B-Instruct"), min_gpu_gb=18),
        # --- Llama (gated on the Hub: accept the license and set HF_TOKEN) ---
        ModelSpec("llama3.1-8b-it", "meta-llama/Llama-3.1-8B-Instruct", "llama", 8,
                  _np("llama3.1-8b-it", "Llama-3.1-8B-Instruct"), min_gpu_gb=20),
        ModelSpec("llama3.3-70b-it", "meta-llama/Llama-3.3-70B-Instruct", "llama", 70,
                  _np("llama3.3-70b-it", "Llama-3.3-70B-Instruct"), min_gpu_gb=150),
        # --- Gemma (gated on the Hub) ---
        ModelSpec("gemma-2-2b-it", "google/gemma-2-2b-it", "gemma", 2,
                  _np("gemma-2-2b-it", "gemma-2-2b-it"), min_gpu_gb=7),
        ModelSpec("gemma-2-9b-it", "google/gemma-2-9b-it", "gemma", 9,
                  _np("gemma-2-9b-it", "gemma-2-9b-it"), min_gpu_gb=22),
        ModelSpec("gemma-3-1b-it", "google/gemma-3-1b-it", "gemma", 1,
                  _np("gemma-3-1b-it", "gemma-3-1b-it"), min_gpu_gb=3),
        ModelSpec("gemma-3-4b-it", "google/gemma-3-4b-it", "gemma", 4,
                  _np("gemma-3-4b-it", "gemma-3-4b-it"), min_gpu_gb=10),
        ModelSpec("gemma-3-12b-it", "google/gemma-3-12b-it", "gemma", 12,
                  _np("gemma-3-12b-it", "gemma-3-12b-it"), min_gpu_gb=28),
        ModelSpec("gemma-3-27b-it", "google/gemma-3-27b-it", "gemma", 27,
                  _np("gemma-3-27b-it", "gemma-3-27b-it"), min_gpu_gb=60),
        # --- OpenAI open weights ---
        ModelSpec("gpt-oss-20b", "openai/gpt-oss-20b", "gpt-oss", 20,
                  _np("gpt-oss-20b", "gpt-oss-20b"), min_gpu_gb=48),
        # --- Base models, for lens sanity checks and CPU tests ---
        ModelSpec("gpt2-small", "openai-community/gpt2", "gpt2", 0.124,
                  _np("gpt2-small", "gpt2"), chat=False),
        ModelSpec("pythia-70m", "EleutherAI/pythia-70m-deduped", "pythia", 0.07,
                  _np("pythia-70m-deduped", "pythia-70m-deduped"), chat=False),
    ]
}


def get_model_spec(name: str, **overrides) -> ModelSpec:
    """Look up a registered model, or build an ad-hoc spec.

    ``name`` may be a registry key or a raw HF id; a raw HF id needs ``lens_path`` (or
    ``lens_file``) in ``overrides`` because no pre-fitted lens is known for it.
    """
    if name in MODELS:
        spec = MODELS[name]
    else:
        spec = ModelSpec(name=name.split("/")[-1].lower(), hf_id=name,
                         family=overrides.pop("family", "unknown"),
                         params_b=overrides.pop("params_b", 0.0))
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if "band" in overrides and isinstance(overrides["band"], str):
        overrides["band"] = BAND_PRESETS[overrides["band"]]
    return replace(spec, **overrides) if overrides else spec


def band_layers(n_layers: int, band: tuple[float, float] | str = DEFAULT_BAND) -> list[int]:
    """Block indices inside a fractional band, e.g. ``(0.38, 0.92)`` of a 36-layer model."""
    if isinstance(band, str):
        band = BAND_PRESETS[band]
    lo, hi = band
    first = max(0, int(round(lo * (n_layers - 1))))
    last = min(n_layers - 2, int(round(hi * (n_layers - 1))))  # final block has no lens
    return list(range(first, last + 1))
