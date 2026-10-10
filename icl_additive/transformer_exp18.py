"""Exp 18: depth and LayerNorm ablations of the transformer's B-rescue.  Spec: docs/preregistration.md, section
"Locating the ingredient of the transformer's B-rescue: depth and LayerNorm (exp 18...)".
Reuses exp 8's run() with the `free` protocol unchanged; the exp-5 model is built by E5.TF(d) and then patched in place (exp 5 / exp 8 are not modified):
  D1 : model.blocks = ModuleList([model.blocks[0]])          (one block; block 0 keeps its seed-identical init)
  NL : ln1, ln2 of every block and lnf -> nn.Identity()       (no LayerNorm anywhere)
The exp-5 alignment proxy uses block0.ln2.weight and E5.norms uses blocks[1]; both are re-defined here for the patched models only
(NL: ln2.weight := 1; norms: the LAST block's fc2/attn-out/head norms).  Neither enters T_e.
One run : python3 -I icl_additive/transformer_exp18.py --run VARIANT N B SEED [--out results/exp18]
Jobs are launched by a bash claim-queue (mkdir claims) in the pre-registered order CELLS x seeds x VARIANTS; one log per run in results/exp18/logs/.
Graceful stop of all runs: touch results/exp18/STOP.
"""
import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn
import transformer_exp5 as E5
import transformer_exp8 as E8

FIELDS = ["variant"] + E8.FIELDS
# pre-registered launch order: N=256,B=64 controls, then N=16,B=1024, then N=16,B=64
CELLS = [(256, 64), (16, 1024), (16, 64)]
VARIANTS = ["D1", "NL"]


def patch(model, variant):
    if variant == "D1":
        model.blocks = nn.ModuleList([model.blocks[0]])
    elif variant == "NL":
        for b in model.blocks:
            b.ln1, b.ln2 = nn.Identity(), nn.Identity()
        model.lnf = nn.Identity()
    else:
        raise ValueError(variant)
    return model


def alignment(model, v):
    with torch.no_grad():
        b = model.blocks[0]
        g = b.ln2.weight[None, :] if hasattr(b.ln2, "weight") else 1.0
        W = (b.fc1.weight * g) @ model.emb.weight[:, :v.numel()]
        return ((W @ v) / (W.norm(dim=1) + 1e-12)).abs().max().item()


def norms(model):
    with torch.no_grad():
        b = model.blocks[-1]
        return [b.fc2.weight.norm().item(), b.attn.out.weight.norm().item(), model.head.weight.norm().item()]


def run_variant(variant, N, B, seed, out, max_steps=30000, stop_at=0.25):
    E8.TF = lambda d: patch(E5.TF(d), variant)      # exp 8's run() calls TF(D) right after nothing else touches the model
    E8.alignment = alignment
    E5.norms = norms                                # exp 8's run() calls E5.norms(model)
    vdir = os.path.join(out, variant)
    os.makedirs(os.path.join(vdir, "curves"), exist_ok=True)
    row = E8.run("free", N, B, seed, max_steps, vdir, stop_at, stop_file=os.path.join(out, "STOP"))
    row["variant"] = variant
    return row


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=4, metavar=("VARIANT", "N", "B", "SEED"))
    ap.add_argument("--variant", choices=VARIANTS, help="same as the first --run argument (checked if given)")
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--stop-at", type=float, default=0.25)
    ap.add_argument("--out", default="results/exp18")
    ap.add_argument("--count-params", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summ = os.path.join(a.out, "summary_raw.csv")
    if a.count_params:
        for v in [None] + VARIANTS:
            m = E5.TF(E8.D)
            if v:
                patch(m, v)
            print(v or "full", sum(p.numel() for p in m.parameters()))
        return
    torch.set_num_threads(1)
    if a.run:
        v, N, B, s = a.run[0], int(a.run[1]), int(a.run[2]), int(a.run[3])
        assert a.variant in (None, v)
        row = run_variant(v, N, B, s, a.out, a.max_steps, a.stop_at)
        append_row(summ, row); print("DONE", row, flush=True)


if __name__ == "__main__":
    main()
