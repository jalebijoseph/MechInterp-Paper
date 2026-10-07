"""Fit a Jacobian lens (build the J-space) for any HF decoder.

Wraps the reference implementation (``jlens.fit``): for each prompt, one forward pass
and ceil(d_model / dim_batch) backward passes give the per-layer Jacobian
dh_final/dh_l summed over target positions and averaged over source positions; the lens
is the mean over prompts. The paper used 1000 x 128-token WikiText sequences; quality
saturates by ~100-300.

Single machine:
    python scripts/fit_lens.py --model Qwen/Qwen3-8B --n-prompts 1000 --out lenses/qwen3-8b.pt

Sharded (run each shard on its own GPU / cloud job, then merge):
    python scripts/fit_lens.py --model ... --shard 0/4 --out lenses/part0.pt
    ...
    python scripts/fit_lens.py --merge lenses/part*.pt --out lenses/qwen3-8b.pt

Cost: per prompt ~ (d_model / dim_batch) backward passes. For an 8B model (d=4096) on one
A100-80GB with dim_batch=32 that is 128 backward passes per prompt, ~10-15 s/prompt, so
~3-4 GPU-hours for 1000 prompts. Prefer the pre-fitted Neuronpedia lenses when one exists
(``python -m jpeers.cli models``).
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HF model id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=300)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--dim-batch", type=int, default=16)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--layers", default=None, help="comma list of source layers (default: all)")
    ap.add_argument("--shard", default=None, help="i/n: fit prompts i::n only")
    ap.add_argument("--compile", action="store_true", help="torch.compile each block")
    ap.add_argument("--prompts-file", default=None, help="one prompt per line instead of WikiText")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--merge", nargs="*", help="merge these shard lenses into --out")
    args = ap.parse_args()

    import jlens
    import torch

    jlens.configure_logging()
    logging.getLogger().setLevel(logging.INFO)

    if args.merge:
        paths = sorted(p for pat in args.merge for p in glob.glob(pat))
        lens = jlens.JacobianLens.merge([jlens.JacobianLens.load(p) for p in paths])
        lens.save(args.out)
        print(f"merged {len(paths)} shards ({lens.n_prompts} prompts) -> {args.out}")
        return
    if not args.model:
        sys.exit("--model is required unless --merge")

    import transformers

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf, tok, compile=args.compile)

    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()][: args.n_prompts]
    else:
        from jlens.examples import load_wikitext_prompts

        prompts = load_wikitext_prompts(args.n_prompts)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        prompts = prompts[i::n]
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    lens = jlens.fit(model, prompts, source_layers=layers, dim_batch=args.dim_batch,
                     max_seq_len=args.max_seq_len,
                     checkpoint_path=args.checkpoint or args.out + ".ckpt",
                     checkpoint_every=10)
    lens.save(args.out)
    print(f"saved {lens} -> {args.out}")


if __name__ == "__main__":
    main()
