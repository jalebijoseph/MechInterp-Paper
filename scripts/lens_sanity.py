"""Check that a model + lens pair behaves like the paper says before running experiments.

1. Multi-hop readout: on "The currency used in the country shaped like a boot is",
   the bridge entity ("Italy") should surface in mid layers before the answer.
2. Lens vs logit lens: the Jacobian lens should recover the model's final next-token
   prediction at earlier layers than the plain logit lens (mean rank per layer).
3. Band check: prints per-layer agreement so you can adjust the workspace band.

    python scripts/lens_sanity.py --model qwen3-1.7b [--lens-path my_lens.pt]
"""

from __future__ import annotations

import argparse

import numpy as np

PROMPTS = [
    "Fact: The currency used in the country shaped like a boot is the",
    "Fact: The capital of the country where the Eiffel Tower stands is",
    "The author of the play about a Danish prince named Hamlet was born in the town of",
    "Fact: The language spoken in the country whose capital is Lima is",
    "The largest planet in our solar system is named after the Roman god",
]
BRIDGES = ["Italy", "France", "Shakespeare", "Peru", "Jupiter"]


def main() -> None:
    import torch

    from jpeers.registry import get_model_spec
    from jpeers.runtime import LensedModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens-path")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    m = LensedModel(get_model_spec(args.model, lens_path=args.lens_path), device=args.device)
    layers = m.lens.source_layers
    band = set(m.band())
    jl_ranks, ll_ranks, bridge_ranks = [], [], []
    for prompt, bridge in zip(PROMPTS, BRIDGES):
        ids, _ = m.encode(prompt, True)
        from jlens import ActivationRecorder

        with torch.no_grad(), ActivationRecorder(m.lm.layers, at=layers + [m.n_layers - 1]) as rec:
            m.lm.forward(ids)
        final = m.lm.unembed(rec.activations[m.n_layers - 1][0, -1].float()).float()
        target = int(final.argmax())
        bridge_id = m.word_token_ids(bridge)
        jr, lr, br = [], [], []
        for layer in layers:
            h = rec.activations[layer][0, -1:].float()
            jl = m.lens_logits(layer, h)[0]
            ll = m.lens_logits(layer, h, use_jacobian=False)[0]
            jr.append(int((jl > jl[target]).sum()))
            lr.append(int((ll > ll[target]).sum()))
            br.append(min(int((jl > jl[b]).sum()) for b in bridge_id))
        jl_ranks.append(jr)
        ll_ranks.append(lr)
        bridge_ranks.append(br)
        print(f"{prompt!r} -> model says {m.vocab_strings[target]!r}; "
              f"best bridge rank {min(br)} at L{layers[int(np.argmin(br))]}")

    jl, ll, br = (np.array(x) for x in (jl_ranks, ll_ranks, bridge_ranks))
    print("\nlayer  band  J-lens rank(final)  logit-lens rank(final)  bridge rank")
    for i, layer in enumerate(layers):
        print(f"L{layer:<4} {'*' if layer in band else ' ':4} {np.median(jl[:, i]):>12.0f}"
              f" {np.median(ll[:, i]):>22.0f} {np.median(br[:, i]):>14.0f}")
    print(f"\nband = {sorted(band)}")


if __name__ == "__main__":
    main()
