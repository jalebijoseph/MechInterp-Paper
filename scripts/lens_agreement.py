"""Does the WikiText-fitted lens read agent transcripts the way a transcript-fitted lens does?

The lens is an average Jacobian over a corpus (Neuronpedia: WikiText). Our experiments
read it on multi-agent chat, a different distribution. This checks the lens transfers:

1. ``prompts``: cut recorded transcripts (``runs/*/*/turns.jsonl``, rendered with the
   chat template) into token windows; write a fitting set and a held-out set (.jsonl).

       python scripts/lens_agreement.py prompts --model qwen3-1.7b --runs runs/collab runs/imposter

2. Fit a transcript lens on the fitting set with scripts/fit_lens.py
   (``--prompts-file lenses/agent_prompts.fit.jsonl``).

3. ``compare``: the registry (WikiText) lens vs the transcript lens, at each band layer:
       J_cosine      cosine between the two Jacobian matrices
       top1 / top10  on held-out transcript windows: share of positions where both lenses
                     give the same top-1 token / overlap of their top-10 sets
   The same readout agreement between the WikiText lens and the plain logit lens is
   printed as a reference: if the two J-lenses agree far more with each other than with
   the logit lens, the readout is a property of the model, not of the fitting corpus.

       python scripts/lens_agreement.py compare --model qwen3-1.7b --lens lenses/agent-1.7b.pt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def cmd_prompts(args) -> None:
    import transformers

    from jpeers.registry import get_model_spec

    spec = get_model_spec(args.model)
    tok = transformers.AutoTokenizer.from_pretrained(spec.hf_id)
    windows: dict[str, None] = {}
    for root in args.runs:
        for path in sorted(Path(root).glob("*/turns.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if not rec.get("messages"):
                    continue
                msgs = [*rec["messages"], {"role": "assistant", "content": rec["text"]}]
                text = tok.apply_chat_template(msgs, tokenize=False, **spec.template_kwargs)
                ids = tok(text, add_special_tokens=False).input_ids
                for start in range(0, max(1, len(ids) - args.min_len), args.window):
                    piece = ids[start:start + args.window]
                    if len(piece) >= args.min_len:
                        windows[tok.decode(piece)] = None  # dedupe: turns repeat history
    items = list(windows)
    random.Random(args.seed).shuffle(items)
    n_held = max(1, int(len(items) * args.heldout))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name, part in (("fit", items[n_held:]), ("heldout", items[:n_held])):
        p = out.with_suffix(f".{name}.jsonl")
        p.write_text("".join(json.dumps(w) + "\n" for w in part), encoding="utf-8")
        print(f"{name}: {len(part)} windows -> {p}")


def cmd_compare(args) -> None:
    import jlens
    import torch
    from jlens import ActivationRecorder

    from jpeers.registry import get_model_spec
    from jpeers.runtime import LensedModel

    m = LensedModel(get_model_spec(args.model))
    other = jlens.JacobianLens.load(args.lens)
    layers = [la for la in m.band() if la in other.jacobians]
    texts = [json.loads(line) for line in
             Path(args.heldout).read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = texts[: args.n_windows]

    stats = {la: {"top1_b": [], "top10_b": [], "top1_logit": [], "top10_logit": []}
             for la in layers}
    for text in texts:
        ids, _ = m.encode(text, True)
        with torch.no_grad(), ActivationRecorder(m.lm.layers, at=layers) as rec:
            m.lm.forward(ids)
        for la in layers:
            h = rec.activations[la][0][args.skip_first:].float()
            a = m.lens_logits(la, h)
            jb = other.jacobians[la].to(device=h.device, dtype=torch.float32)
            b = m.lm.unembed(h @ jb.T).float()
            lg = m.lens_logits(la, h, use_jacobian=False)
            ta = a.topk(10, dim=-1).indices
            for key, x in (("b", b), ("logit", lg)):
                tx = x.topk(10, dim=-1).indices
                stats[la][f"top1_{key}"].append((ta[:, 0] == tx[:, 0]).float().mean().item())
                overlap = (ta.unsqueeze(-1) == tx.unsqueeze(-2)).any(-1).float().mean(-1)
                stats[la][f"top10_{key}"].append(overlap.mean().item())

    print(f"{args.model}: registry lens vs {args.lens}, {len(texts)} held-out windows\n")
    print("layer  J_cosine  top1(J vs J')  top10(J vs J')  |  top1(J vs logit)  top10(J vs logit)")
    rows = []
    for la in layers:
        ja = m.lens.jacobians[la].float().flatten()
        jb = other.jacobians[la].float().flatten()
        cos = float(ja @ jb / (ja.norm() * jb.norm()))
        s = {k: sum(v) / len(v) for k, v in stats[la].items()}
        rows.append({"layer": la, "J_cosine": cos, **s})
        print(f"{la:5d}  {cos:8.3f}  {s['top1_b']:13.3f}  {s['top10_b']:14.3f}  |  "
              f"{s['top1_logit']:16.3f}  {s['top10_logit']:17.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--model", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="experiment dirs, e.g. runs/collab")
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--min-len", type=int, default=64)
    p.add_argument("--heldout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="lenses/agent_prompts")
    c = sub.add_parser("compare")
    c.add_argument("--model", required=True)
    c.add_argument("--lens", required=True, help="the transcript-fitted lens")
    c.add_argument("--heldout", default="lenses/agent_prompts.heldout.jsonl")
    c.add_argument("--n-windows", type=int, default=50)
    c.add_argument("--skip-first", type=int, default=4, help="attention-sink positions")
    c.add_argument("--out", default=None)
    args = ap.parse_args()
    {"prompts": cmd_prompts, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
