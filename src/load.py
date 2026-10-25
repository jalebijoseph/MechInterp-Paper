"""Load the model and SAEs.

This is the one module that triggers large downloads: gemma-2-2b from
Hugging Face (requires accepting its license + an HF token) and the
Gemma Scope SAEs via sae_lens. Nothing in here runs on import — only
when these functions are actually called — so importing this module
elsewhere in the project (e.g. for type hints) is always safe.

Do not call these from an automated script without the user's
go-ahead; confirm first, per the project's "ask before downloading
large models or SAEs" rule.
"""

from __future__ import annotations

import logging

from src.config import LAYER_AVERAGE_L0, LAYERS, MODEL_NAME, SAE_RELEASE, SAE_WIDTH

logger = logging.getLogger("mech_interp_sae.load")


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str = MODEL_NAME, device: str | None = None, dtype=None):
    """Load gemma-2-2b into a transformer_lens HookedTransformer.

    Requires `huggingface-cli login` (or HF_TOKEN set) with access to
    the gated gemma-2-2b repo, and downloads several GB of weights the
    first time it's called.

    Defaults to bfloat16 on CPU: gemma-2-2b (~2.6B params) in float32 is
    ~10GB+ of weights alone, which can exceed RAM on a 16GB machine and
    get the process killed with no traceback (exactly what happened on
    this machine's first run). bfloat16 roughly halves that, and CPU
    inference supports bfloat16 reasonably well (unlike float16, whose
    CPU kernels are much slower). On CUDA/MPS, float32 is used unless a
    dtype is passed explicitly.

    Uses `from_pretrained_no_processing`: the default weight processing
    (fold_ln, center_writing_weights, ...) makes extra copies of every
    weight, which is what pushed peak RAM over the limit on this machine,
    and center_writing_weights also changes the residual stream itself —
    Gemma Scope SAEs were trained on the unprocessed residual stream, so
    processing would feed them slightly off-distribution inputs.
    """
    import torch
    from transformer_lens import HookedTransformer

    device = device or get_device()
    if dtype is None:
        dtype = torch.bfloat16 if device == "cpu" else torch.float32
    logger.info("Loading %s onto %s as %s", model_name, device, dtype)
    return HookedTransformer.from_pretrained_no_processing(model_name, device=device, dtype=dtype)


def sae_id_for_layer(layer: int, width: str = SAE_WIDTH) -> str:
    """Gemma Scope / sae_lens SAE id for a given residual-stream layer.

    Confirmed against the google/gemma-scope-2b-pt-res file tree on
    Hugging Face (see config.LAYER_AVERAGE_L0 for which variant was
    picked for each of our layers). Raises for any layer we haven't
    looked up, rather than silently guessing an L0 value.
    """
    if layer not in LAYER_AVERAGE_L0:
        raise ValueError(
            f"No confirmed average_l0 variant for layer {layer} — check "
            f"https://huggingface.co/google/gemma-scope-2b-pt-res/tree/main/layer_{layer}/width_{width} "
            "and add it to config.LAYER_AVERAGE_L0."
        )
    return f"layer_{layer}/width_{width}/average_l0_{LAYER_AVERAGE_L0[layer]}"


def load_sae(layer: int, release: str = SAE_RELEASE, width: str = SAE_WIDTH, device: str | None = None):
    """Load the Gemma Scope SAE for one residual-stream layer via sae_lens.

    Downloads the SAE weights the first time it's called for a given
    layer/width.
    """
    from sae_lens import SAE

    device = device or get_device()
    sae_id = sae_id_for_layer(layer, width)
    logger.info("Loading SAE release=%s id=%s onto %s", release, sae_id, device)
    sae, _cfg, _sparsity = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    return sae


def load_saes(layers: list[int] = LAYERS, **kwargs) -> dict[int, object]:
    return {layer: load_sae(layer, **kwargs) for layer in layers}
