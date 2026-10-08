"""Exp 4: does the finite-context effect survive in a small transformer?  (spec: docs/spec_exp4.md)

Data: x ~ N(0, I_d), teacher direction v (unit, fixed per run), task c ~ N(0,1) per prompt, y = c * sigma_k(<v,x>).
Tokens: [x_i ; y_i] (context), [x_q ; 0] (query, last position).  Loss/eval: MSE of the query prediction (E[y^2] = 1).

CLI (one cell):  python3 -I icl_additive/transformer_exp4.py --cell k d N attn seed [--inweight] [--max-steps 30000]
Grid driver:     python3 -I icl_additive/transformer_exp4.py --grid [--seeds 3]      (resumable via results/exp4/summary.csv)
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TOKENS_PER_STEP = 8192
EVAL_EVERY = 200
ALIGN_EVERY = 1000
N_EVAL = 2048
EVAL_SEED = 20261008
LR = 1e-3
D_MODEL, N_HEADS, D_FF = 64, 4, 256
FIELDS = ["k", "d", "N", "B", "attn", "inweight", "seed", "steps_run", "Te05", "Te025", "Te01",
          "reached05", "reached025", "reached01", "final_mse", "min_mse", "wall_s", "cpu_s",
          "final_align", "max_align", "diverged"]


# --- Hermite activations (same normalisation as icl_additive/hermite.py), torch version -------------------
def sigma_t(k, z):
    if k == 1:
        return z
    if k == 2:
        return (z * z - 1.0) / math.sqrt(2.0)
    raise ValueError(k)


# --- data ------------------------------------------------------------------------------------------------
def make_batch(d, N, B, v, k, gen, inweight):
    """Returns tokens (B, N+1, d+1) and query target y_q (B,)."""
    x = torch.randn(B, N + 1, d, generator=gen)
    z = x @ v
    if inweight:                      # fixed task c = 1, context labels zeroed
        c = torch.ones(B)
        y_ctx = torch.zeros(B, N)
    else:
        c = torch.randn(B, generator=gen)
        y_ctx = c[:, None] * sigma_t(k, z[:, :N])
    y_q = c * sigma_t(k, z[:, N])
    lab = torch.cat([y_ctx, torch.zeros(B, 1)], dim=1)
    return torch.cat([x, lab[..., None]], dim=-1), y_q


# --- model -----------------------------------------------------------------------------------------------
class Attn(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, h):
        B, T, _ = h.shape
        q, k, v = self.qkv(h).view(B, T, 3, N_HEADS, D_MODEL // N_HEADS).permute(2, 0, 3, 1, 4)
        if self.kind == "softmax":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:  # linear: (Q K^T)/N under the causal mask, no other normalisation (N = T-1 context tokens)
            s = (q @ k.transpose(-1, -2)) / (T - 1)
            s = s * torch.tril(torch.ones(T, T))
            o = s @ v
        return self.out(o.transpose(1, 2).reshape(B, T, D_MODEL))


class Block(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.attn = Attn(kind)
        self.fc1, self.fc2 = nn.Linear(D_MODEL, D_FF), nn.Linear(D_FF, D_MODEL)

    def forward(self, h):
        h = h + self.attn(self.ln1(h))
        return h + self.fc2(F.relu(self.fc1(self.ln2(h))))


class TF(nn.Module):
    def __init__(self, d, kind):
        super().__init__()
        self.emb = nn.Linear(d + 1, D_MODEL)
        self.blocks = nn.ModuleList([Block(kind), Block(kind)])
        self.lnf = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 1)

    def forward(self, tok):
        h = self.emb(tok)
        for b in self.blocks:
            h = b(h)
        return self.head(self.lnf(h[:, -1]))[:, 0]


def alignment(model, v):
    """max_j |cos(W_j, v)| with W_j = (fc1.weight[j] * ln2.weight) @ emb.weight[:, :d]  (block 1).
    Composition: row j of the block-1 MLP first layer, times the LN2 gain (elementwise), times the embedding
    restricted to the x-coordinates.  LN centring/scaling and the attention-branch contribution are ignored."""
    with torch.no_grad():
        b = model.blocks[0]
        d = v.numel()
        W = (b.fc1.weight * b.ln2.weight[None, :]) @ model.emb.weight[:, :d]     # (D_FF, d)
        cos = (W @ v) / (W.norm(dim=1) + 1e-12)
        return cos.abs().max().item()


@torch.no_grad()
def evaluate(model, d, N, v, k, inweight):
    gen = torch.Generator().manual_seed(EVAL_SEED)
    tot, ch = 0.0, 512
    for _ in range(N_EVAL // ch):
        tok, y = make_batch(d, N, ch, v, k, gen, inweight)
        tot += ((model(tok) - y) ** 2).sum().item()
    return tot / N_EVAL            # E[y^2] = 1 for both settings


def cell_name(k, d, N, attn, seed, inweight):
    return f"k{k}_d{d}_N{N}_{attn}_s{seed}" + ("_iw" if inweight else "")


def run_cell(k, d, N, attn, seed, inweight, max_steps, out, eval_every=EVAL_EVERY, align_every=ALIGN_EVERY):
    torch.set_num_threads(4)
    B = TOKENS_PER_STEP // N if not inweight else 128
    torch.manual_seed(seed); np.random.seed(seed)
    gv = torch.Generator().manual_seed(1000 + seed)
    v = torch.randn(d, generator=gv); v = v / v.norm()
    model = TF(d, attn)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(7919 * seed + 13)
    steps, mses, trainl, asteps, aligns = [], [], [], [], []
    te = {0.5: None, 0.25: None, 0.1: None}
    t0, c0 = time.time(), time.process_time()
    run_loss, diverged, step = 0.0, False, 0
    while True:
        if step % eval_every == 0:
            m = evaluate(model, d, N, v, k, inweight)
            steps.append(step); mses.append(m); trainl.append(run_loss / eval_every if step else float("nan")); run_loss = 0.0
            for th in te:
                if te[th] is None and m < th:
                    te[th] = step
            if step % align_every == 0:
                asteps.append(step); aligns.append(alignment(model, v))
            if step % max(2000, eval_every) == 0:
                print(f"[{cell_name(k, d, N, attn, seed, inweight)}] step {step} mse {m:.4f} align {aligns[-1]:.3f} "
                      f"t={time.time() - t0:.0f}s", flush=True)
            if m < 0.1 or step >= max_steps or not math.isfinite(m):
                diverged = not math.isfinite(m)
                break
        tok, y = make_batch(d, N, B, v, k, gen, inweight)
        loss = ((model(tok) - y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run_loss += loss.item()
        step += 1
    wall, cpu = time.time() - t0, time.process_time() - c0
    name = cell_name(k, d, N, attn, seed, inweight)
    np.savez(os.path.join(out, "curves", name + ".npz"), steps=np.array(steps), mse=np.array(mses),
             train_mse=np.array(trainl), align_steps=np.array(asteps), align=np.array(aligns),
             k=k, d=d, N=N, B=B, seed=seed)
    row = dict(k=k, d=d, N=N, B=B, attn=attn, inweight=int(inweight), seed=seed, steps_run=step,
               Te05=te[0.5], Te025=te[0.25], Te01=te[0.1],
               reached05=int(te[0.5] is not None), reached025=int(te[0.25] is not None), reached01=int(te[0.1] is not None),
               final_mse=mses[-1], min_mse=min(mses), wall_s=round(wall, 1), cpu_s=round(cpu, 1),
               final_align=aligns[-1], max_align=max(aligns), diverged=int(diverged))
    return row


def done_set(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(int(r["k"]), int(r["d"]), int(r["N"]), r["attn"], int(r["seed"]), int(r["inweight"])) for r in csv.DictReader(f)}


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def grid(seeds_by_prio):
    """Priority-ordered cell list (k, d, N, attn, inweight, prio)."""
    g = []
    g += [(2, 32, N, "softmax", False, 1) for N in (256, 64, 16)]          # P8
    g += [(1, 32, N, "softmax", False, 2) for N in (256, 64, 16)]          # P9 k*=1
    g += [(2, 32, 64, "softmax", True, 2)]                                 # P9 in-weight control (B=128)
    g += [(2, 32, N, "linear", False, 3) for N in (256, 16)]               # P10
    g += [(2, 16, N, "softmax", False, 4) for N in (256, 64, 16)]          # P8 at d=16
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", nargs=5, metavar=("k", "d", "N", "attn", "seed"))
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--seeds", type=int, default=3, help="seeds for priority 1 (and 2-3 unless --seeds-low)")
    ap.add_argument("--seeds-low", type=int, default=None, help="seeds for priorities 2-4 (default = --seeds)")
    ap.add_argument("--prio", type=int, default=4, help="max priority to run in --grid")
    ap.add_argument("--inweight", action="store_true")
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY, help="eval interval (supplementary fine-resolution k*=1 runs only)")
    ap.add_argument("--align-every", type=int, default=ALIGN_EVERY, help="alignment interval (multiple of eval-every; supplementary runs only)")
    ap.add_argument("--out", default="results/exp4/")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "curves"), exist_ok=True)
    summ = os.path.join(a.out, "summary.csv")
    cells = []
    if a.cell:
        k, d, N, attn, seed = a.cell
        cells = [(int(k), int(d), int(N), attn, a.inweight, int(seed))]
    elif a.grid:
        low = a.seeds_low if a.seeds_low is not None else a.seeds
        for pr in range(1, a.prio + 1):          # priority-major, then seed-major (all cells for seed 0 first)
            for s in range(a.seeds if pr == 1 else low):
                for (k, d, N, attn, iw, p_) in grid(None):
                    if p_ == pr:
                        cells.append((k, d, N, attn, iw, s))
    for (k, d, N, attn, iw, seed) in cells:
        if (k, d, N, attn, seed, int(iw)) in done_set(summ):
            print("skip", cell_name(k, d, N, attn, seed, iw), flush=True)
            continue
        row = run_cell(k, d, N, attn, seed, iw, a.max_steps, a.out, a.eval_every, a.align_every)
        append_row(summ, row)
        print("DONE", {kk: row[kk] for kk in ("k", "d", "N", "attn", "inweight", "seed", "steps_run", "Te05", "Te01", "final_mse", "wall_s", "cpu_s")}, flush=True)


if __name__ == "__main__":
    main()
