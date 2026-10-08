"""Exp 5: regime-matched transformer test (d=256, MLP width 32).  Spec: docs/spec_exp5.md.  Standalone copy of the exp-4 model/data/loop
with width flags (exp 4 files/results untouched).  Data: x ~ N(0,I_d), unit teacher v, c ~ N(0,1) per prompt, y = c*sigma_k(<v,x>);
tokens [x_i; y_i] (context) and [x_q; 0] (query, last).  Loss/eval = MSE of the query prediction (E[y^2]=1).

One cell:   python3 -I icl_additive/transformer_exp5.py --cell NAME SEED [--max-steps 30000] [--threads 2]
All cells:  python3 -I icl_additive/transformer_exp5.py --grid [--cells a,d,c,b,e,f] [--seeds 3]     (resumable via results/exp5/summary.csv)
Cells (name: k, N, B, inweight, n_seeds): see CELLS.  --B-override changes B for a cell (used for (d) if B=1024 is too expensive).
"""
import argparse, csv, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EVAL_EVERY = 200
ALIGN_EVERY = 200
NORM_EVERY = 1000
N_EVAL = 2048
EVAL_SEED = 20261008
LR = 1e-3
D, D_MODEL, N_HEADS, D_FF = 256, 64, 4, 32
CELLS = {  # name: (k, N, B, inweight, seeds)
    "a": (2, 256, 64, False, 3), "b": (2, 64, 64, False, 3), "c": (2, 16, 64, False, 3),
    "d": (2, 16, 1024, False, 3), "e": (1, 16, 64, False, 2), "f": (2, 64, 64, True, 2)}
FIELDS = ["cell", "k", "d", "N", "B", "inweight", "seed", "steps_run", "Te05", "Te025", "Te01", "reached05", "reached025", "reached01",
          "final_mse", "min_mse", "init_align", "final_align", "max_align", "wall_s", "cpu_s", "threads", "diverged"]


def sigma_t(k, z):
    return z if k == 1 else (z * z - 1.0) / math.sqrt(2.0)


def make_batch(d, N, B, v, k, gen, inweight):
    x = torch.randn(B, N + 1, d, generator=gen)
    z = x @ v
    if inweight:
        c, y_ctx = torch.ones(B), torch.zeros(B, N)
    else:
        c = torch.randn(B, generator=gen)
        y_ctx = c[:, None] * sigma_t(k, z[:, :N])
    y_q = c * sigma_t(k, z[:, N])
    lab = torch.cat([y_ctx, torch.zeros(B, 1)], dim=1)
    return torch.cat([x, lab[..., None]], dim=-1), y_q


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, h):
        B, T, _ = h.shape
        q, k, v = self.qkv(h).view(B, T, 3, N_HEADS, D_MODEL // N_HEADS).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(o.transpose(1, 2).reshape(B, T, D_MODEL))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.attn = Attn()
        self.fc1, self.fc2 = nn.Linear(D_MODEL, D_FF), nn.Linear(D_FF, D_MODEL)

    def forward(self, h):
        h = h + self.attn(self.ln1(h))
        return h + self.fc2(F.relu(self.fc1(self.ln2(h))))


class TF(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.emb = nn.Linear(d + 1, D_MODEL)
        self.blocks = nn.ModuleList([Block(), Block()])
        self.lnf = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 1)

    def forward(self, tok):
        h = self.emb(tok)
        for b in self.blocks:
            h = b(h)
        return self.head(self.lnf(h[:, -1]))[:, 0]


def alignment(model, v):
    """max_j |cos(W_j, v)|, W_j = (fc1.weight[j] * ln2.weight) @ emb.weight[:, :d]  (block 1; same definition as exp 4)."""
    with torch.no_grad():
        b = model.blocks[0]
        W = (b.fc1.weight * b.ln2.weight[None, :]) @ model.emb.weight[:, :v.numel()]
        return ((W @ v) / (W.norm(dim=1) + 1e-12)).abs().max().item()


def norms(model):
    with torch.no_grad():
        b2 = model.blocks[1]
        return [b2.fc2.weight.norm().item(), b2.attn.out.weight.norm().item(), model.head.weight.norm().item()]


@torch.no_grad()
def evaluate(model, d, N, v, k, inweight):
    gen = torch.Generator().manual_seed(EVAL_SEED)
    tot, ch = 0.0, 256
    for _ in range(N_EVAL // ch):
        tok, y = make_batch(d, N, ch, v, k, gen, inweight)
        tot += ((model(tok) - y) ** 2).sum().item()
    return tot / N_EVAL


def run_cell(cell, seed, max_steps, out, threads, B_override=None, stop_at=0.1):
    torch.set_num_threads(threads)
    k, N, B, iw, _ = CELLS[cell]
    B = B_override or B
    torch.manual_seed(seed); np.random.seed(seed)
    gv = torch.Generator().manual_seed(1000 + seed)
    v = torch.randn(D, generator=gv); v = v / v.norm()
    model = TF(D)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(7919 * seed + 13)
    steps, mses, trainl, aligns, nsteps, nrm = [], [], [], [], [], []
    te = {0.5: None, 0.25: None, 0.1: None}
    t0, c0 = time.time(), time.process_time()
    run_loss, diverged, step = 0.0, False, 0
    name = f"{cell}_s{seed}"
    while True:
        if step % EVAL_EVERY == 0:
            m = evaluate(model, D, N, v, k, iw)
            steps.append(step); mses.append(m); trainl.append(run_loss / EVAL_EVERY if step else float("nan")); run_loss = 0.0
            aligns.append(alignment(model, v))
            if step % NORM_EVERY == 0:
                nsteps.append(step); nrm.append(norms(model))
            for th in te:
                if te[th] is None and m < th:
                    te[th] = step
            if step == 0:
                print(f"[{name}] INIT align {aligns[0]:.4f} mse {m:.4f}", flush=True)
            if step % 1000 == 0:
                print(f"[{name}] step {step} mse {m:.4f} align {aligns[-1]:.3f} t={time.time() - t0:.0f}s", flush=True)
            if (stop_at and m < stop_at) or step >= max_steps or not math.isfinite(m):
                diverged = not math.isfinite(m)
                break
        tok, y = make_batch(D, N, B, v, k, gen, iw)
        loss = ((model(tok) - y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run_loss += loss.item()
        step += 1
    wall, cpu = time.time() - t0, time.process_time() - c0
    np.savez(os.path.join(out, "curves", name + ".npz"), steps=np.array(steps), mse=np.array(mses), train_mse=np.array(trainl),
             align=np.array(aligns), norm_steps=np.array(nsteps), norms=np.array(nrm),  # norms cols: blk2 fc2, blk2 attn.out, head
             k=k, d=D, N=N, B=B, seed=seed, cell=cell)
    return dict(cell=cell, k=k, d=D, N=N, B=B, inweight=int(iw), seed=seed, steps_run=step, Te05=te[0.5], Te025=te[0.25], Te01=te[0.1],
                reached05=int(te[0.5] is not None), reached025=int(te[0.25] is not None), reached01=int(te[0.1] is not None),
                final_mse=mses[-1], min_mse=min(mses), init_align=aligns[0], final_align=aligns[-1], max_align=max(aligns),
                wall_s=round(wall, 1), cpu_s=round(cpu, 1), threads=threads, diverged=int(diverged))


def done_set(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r["cell"], int(r["seed"])) for r in csv.DictReader(f)}


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", nargs=2, metavar=("NAME", "SEED"))
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--cells", default="a,d,c,b,e,f", help="priority order")
    ap.add_argument("--seeds", default=None, help="comma list of seeds to restrict to (default: all seeds of the cell)")
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--B-override", type=int, default=None)
    ap.add_argument("--stop-at", type=float, default=0.1, help="early stop when eval MSE < this (0 = never)")
    ap.add_argument("--out", default="results/exp5/")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "curves"), exist_ok=True)
    summ = os.path.join(a.out, "summary.csv")
    jobs = []
    if a.cell:
        jobs = [(a.cell[0], int(a.cell[1]))]
    elif a.grid:
        only = None if a.seeds is None else {int(s) for s in a.seeds.split(",")}
        for c in a.cells.split(","):
            for s in range(CELLS[c][4]):
                if only is None or s in only:
                    jobs.append((c, s))
    for (c, s) in jobs:
        if (c, s) in done_set(summ):
            print("skip", c, s, flush=True); continue
        row = run_cell(c, s, a.max_steps, a.out, a.threads, a.B_override, a.stop_at)
        append_row(summ, row)
        print("DONE", {kk: row[kk] for kk in ("cell", "B", "seed", "steps_run", "Te05", "Te01", "final_mse", "init_align", "wall_s", "cpu_s")}, flush=True)


if __name__ == "__main__":
    main()
