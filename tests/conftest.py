"""A tiny randomly-initialised GPT-2 (4 layers, d=32) with a lens fitted on the spot.

Runs on CPU in seconds; only the GPT-2 tokenizer is downloaded (once, ~1 MB).
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def tiny_model():
    import jlens
    import transformers

    from jpeers.registry import ModelSpec
    from jpeers.runtime import LensedModel

    torch.manual_seed(0)
    tok = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2")
    cfg = transformers.GPT2Config(n_layer=4, n_embd=32, n_head=2, n_positions=512)
    hf = transformers.GPT2LMHeadModel(cfg).eval()
    lm = jlens.from_hf(hf, tok)
    prompts = ["The quick brown fox jumps over the lazy dog. " * 6,
               "Alice and Bob are working together on a research project today. " * 4]
    lens = jlens.fit(lm, prompts, dim_batch=8, max_seq_len=64, skip_first=4)
    spec = ModelSpec("tiny-gpt2", "tiny", "test", 0.0, chat=False, band=(0.0, 1.0))
    return LensedModel(spec, hf_model=hf, tokenizer=tok, lens=lens, device="cpu")
