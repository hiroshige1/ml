"""Exp 8: transformer trap-vs-tradeoff test.  Spec: docs/preregistration.md, section "Transformer trap-vs-tradeoff test (exp 8)".
Imports the exp-5 model/data/eval/alignment code unchanged (icl_additive/transformer_exp5.py is not modified) and adds --readout:
  free        trainable readout (= exp 5)
  pinned1     final linear readout (weight AND bias) frozen at its initialisation (requires_grad=False, not in the optimizer)
  pinned_small  initial readout weight x0.1, then frozen (bias frozen at its init value)
One run : python3 -I icl_additive/transformer_exp8.py --run PROTO N B SEED [--stop-at 0.25]
Queue   : python3 -I icl_additive/transformer_exp8.py --queue   (priority-ordered job list; claim files make 2 workers safe)
Graceful stop of all runs: touch results/exp8/STOP; of one run: touch results/exp8/STOP_<name> (runs end as censored at the next eval).
"""
import argparse, csv, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import transformer_exp5 as E5
from transformer_exp5 import D, LR, EVAL_EVERY, NORM_EVERY, TF, make_batch, evaluate, alignment

K = 2
HEAD_LOG_EVERY = 1000
FIELDS = ["proto", "N", "B", "seed", "steps_run", "Te05", "Te025", "reached05", "reached025", "final_mse", "min_mse", "init_mse",
          "init_align", "final_align", "max_align", "head_norm_init", "head_norm_final", "censored_by", "wall_s", "cpu_s", "threads"]
# priority order: (proto, N, B)
PRIORITY = [("pinned1", 16, 1024), ("pinned_small", 16, 64), ("pinned1", 16, 64), ("pinned_small", 256, 64), ("pinned1", 256, 64),
            ("pinned_small", 16, 1024), ("free", 16, 64), ("free", 16, 1024), ("free", 256, 64)]


def run(proto, N, B, seed, max_steps, out, stop_at, threads=1, stop_file=None):
    torch.set_num_threads(threads)
    torch.manual_seed(seed); np.random.seed(seed)
    gv = torch.Generator().manual_seed(1000 + seed)
    v = torch.randn(D, generator=gv); v = v / v.norm()
    model = TF(D)                       # same init as exp 5 for the same seed
    if proto == "pinned_small":
        with torch.no_grad():
            model.head.weight.mul_(0.1)
    if proto != "free":
        for p in model.head.parameters():
            p.requires_grad_(False)
    head0 = model.head.weight.detach().clone(); bias0 = model.head.bias.detach().clone()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    gen = torch.Generator().manual_seed(7919 * seed + 13)
    name = f"{proto}_N{N}_B{B}_s{seed}"
    steps, mses, trainl, aligns, nsteps, nrm, hsteps, hnorm = [], [], [], [], [], [], [], []
    te = {0.5: None, 0.25: None}
    t0, c0 = time.time(), time.process_time()
    run_loss, step, censored = 0.0, 0, ""
    while True:
        if step % EVAL_EVERY == 0:
            m = evaluate(model, D, N, v, K, False)
            steps.append(step); mses.append(m); trainl.append(run_loss / EVAL_EVERY if step else float("nan")); run_loss = 0.0
            aligns.append(alignment(model, v))
            if step % NORM_EVERY == 0:
                nsteps.append(step); nrm.append(E5.norms(model))
            if step % HEAD_LOG_EVERY == 0 or step == 0:
                hsteps.append(step); hnorm.append(model.head.weight.norm().item())
            for th in te:
                if te[th] is None and m < th:
                    te[th] = step
            if step == 0:
                print(f"[{name}] INIT align {aligns[0]:.4f} mse {m:.4f} head_norm {hnorm[0]:.4f}", flush=True)
            if step % 1000 == 0:
                print(f"[{name}] step {step} mse {m:.4f} align {aligns[-1]:.3f} head {hnorm[-1]:.3f} t={time.time() - t0:.0f}s", flush=True)
            if m < stop_at or step >= max_steps or not math.isfinite(m):
                if step >= max_steps and m >= stop_at:
                    censored = "max_steps" if te[0.25] is None else ""
                break
            if stop_file and (os.path.exists(stop_file) or os.path.exists(f"{stop_file}_{name}")):
                censored = "STOP_file"; break
        tok, y = make_batch(D, N, B, v, K, gen, False)
        loss = ((model(tok) - y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run_loss += loss.item()
        step += 1
    if proto != "free":  # sanity: readout truly frozen
        assert torch.equal(model.head.weight, head0) and torch.equal(model.head.bias, bias0), "frozen readout changed"
    hsteps.append(step); hnorm.append(model.head.weight.norm().item())
    wall, cpu = time.time() - t0, time.process_time() - c0
    np.savez(os.path.join(out, "curves", name + ".npz"), steps=np.array(steps), mse=np.array(mses), train_mse=np.array(trainl),
             align=np.array(aligns), norm_steps=np.array(nsteps), norms=np.array(nrm), head_steps=np.array(hsteps),
             head_norm=np.array(hnorm), proto=proto, N=N, B=B, seed=seed)
    return dict(proto=proto, N=N, B=B, seed=seed, steps_run=step, Te05=te[0.5], Te025=te[0.25], reached05=int(te[0.5] is not None),
                reached025=int(te[0.25] is not None), final_mse=mses[-1], min_mse=min(mses), init_mse=mses[0], init_align=aligns[0],
                final_align=aligns[-1], max_align=max(aligns), head_norm_init=hnorm[0], head_norm_final=hnorm[-1], censored_by=censored,
                wall_s=round(wall, 1), cpu_s=round(cpu, 1), threads=threads)


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=4, metavar=("PROTO", "N", "B", "SEED"))
    ap.add_argument("--queue", action="store_true")
    ap.add_argument("--readout", choices=["free", "pinned1", "pinned_small"])
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--stop-at", type=float, default=0.25)
    ap.add_argument("--out", default="results/exp8/")
    ap.add_argument("--skip", default="", help="comma list of proto:N:B:seed to skip in the queue")
    ap.add_argument("--seeds1", default="", help="comma list of proto:N:B for which only seed 0 is run (budget cut)")
    ap.add_argument("--only-protos", default="")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "curves"), exist_ok=True); os.makedirs(os.path.join(a.out, "claims"), exist_ok=True)
    summ = os.path.join(a.out, "summary_raw.csv"); stop_file = os.path.join(a.out, "STOP")
    if a.run:
        p, N, B, s = a.run[0], int(a.run[1]), int(a.run[2]), int(a.run[3])
        assert a.readout in (None, p)
        row = run(p, N, B, s, a.max_steps, a.out, a.stop_at, stop_file=stop_file)
        append_row(summ, row); print("DONE", row, flush=True)
    elif a.queue:
        skip = set(a.skip.split(",")) - {""}; one = set(a.seeds1.split(",")) - {""}
        only = set(a.only_protos.split(",")) - {""}
        jobs = [(p, N, B, s) for (p, N, B) in PRIORITY for s in (0, 1)
                if (not only or p in only) and f"{p}:{N}:{B}:{s}" not in skip and not (s == 1 and f"{p}:{N}:{B}" in one)]
        for (p, N, B, s) in jobs:
            if os.path.exists(stop_file):
                break
            claim = os.path.join(a.out, "claims", f"{p}_N{N}_B{B}_s{s}")
            try:
                os.close(os.open(claim, os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                continue
            row = run(p, N, B, s, a.max_steps, a.out, a.stop_at, stop_file=stop_file)
            append_row(summ, row); print("DONE", row, flush=True)


if __name__ == "__main__":
    main()
