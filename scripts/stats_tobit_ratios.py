"""Tobit kappa_eff fits (exp 1b) and SGD/ODE absolute-time ratio tables (exp 1b, exp 6). Not committed.
Run: python scripts/stats_tobit_ratios.py   (from /home/user/ml)"""
import sys, numpy as np, pandas as pd
from scipy import optimize, stats
sys.path.insert(0, 'scripts')
from ode_kappa_exp1b import T05, rhs_free, rhs_tied
R = 'results/'
rng = np.random.default_rng(12345)
ETA = 1 / 32**2
ODE_KAPPA = {('free1', 32): 6.47, ('free1', 128): 5.46, ('free1', 512): 4.99, ('free1', 4096): 4.85,
             ('free10', 32): 7.02, ('free10', 128): 6.18, ('free10', 512): 5.32, ('free10', 4096): 4.92,
             ('tied', 32): 4.18, ('tied', 128): 4.15, ('tied', 512): 4.15, ('tied', 4096): 4.15}

# ---------------- data ----------------
s = pd.read_csv(R + 'exp1b/summary.csv')
k = s[s.phase.isin(['kappa', 'kappa2']) & (~s.superseded) & s.protocol.isin(['free1', 'free10', 'tied'])].copy()
assert (k.groupby(['protocol', 'N']).size() == 12).all()

def tobit_fit(x, y, cens, ycap):
    """y = a + b x + sigma*eps, right-censored at ycap where cens (y then unobserved, >= ycap)."""
    def nll(p):
        a, b, ls = p; sg = np.exp(ls); mu = a + b * x
        ll = np.where(cens, stats.norm.logsf((ycap - mu) / sg),
                      stats.norm.logpdf((y - mu) / sg) - ls)
        return -ll.sum()
    obs = ~cens
    b0, a0 = np.polyfit(x[obs], y[obs], 1)
    s0 = np.log(max(np.std(y[obs] - a0 - b0 * x[obs]), 0.05))
    best = None
    for start in ([a0, b0, s0], [a0, b0 * 1.3, s0], [a0, b0 * 0.8, s0 + 0.5]):
        r = optimize.minimize(nll, start, method='BFGS')
        r2 = optimize.minimize(nll, r.x, method='Nelder-Mead', options=dict(xatol=1e-9, fatol=1e-12, maxiter=20000))
        if best is None or r2.fun < best.fun: best = r2
    return best, nll

def num_hess(f, p, h=1e-4):
    n = len(p); H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            e_i = np.eye(n)[i] * h; e_j = np.eye(n)[j] * h
            H[i, j] = (f(p + e_i + e_j) - f(p + e_i - e_j) - f(p - e_i + e_j) + f(p - e_i - e_j)) / (4 * h * h)
    return H

rows = []
for (prot, N), g in k.groupby(['protocol', 'N'], sort=False):
    x = np.log(g.m0.values)
    cens = ~g.reached05.values
    cap = g.max_steps.values.astype(float)
    y = np.where(cens, np.log(cap * ETA), np.log(np.where(cens, 1, g.T05.values) * ETA))
    ycap = np.log(cap * ETA)
    fit, nll = tobit_fit(x, y, cens, ycap)
    a, b, ls = fit.x
    H = num_hess(nll, fit.x)
    cov = np.linalg.inv(H); se_b = np.sqrt(cov[1, 1])
    # parametric-free bootstrap of runs within the cell (resample the 12 runs)
    bb = []
    for _ in range(1000):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) < 2 or (~cens[idx]).sum() < 4: continue
        try:
            f2, _ = tobit_fit(x[idx], y[idx], cens[idx], ycap[idx]); bb.append(f2.x[1])
        except Exception: pass
    bb = np.array(bb)
    # OLS on reached only (same runs as README)
    o = ~cens; bo, ao = np.polyfit(x[o], y[o], 1)
    res = y[o] - ao - bo * x[o]; dof = o.sum() - 2
    se_o = np.sqrt(res @ res / dof / ((x[o] - x[o].mean())**2).sum())
    ok = ODE_KAPPA[(prot, N)]
    rows.append(dict(protocol=prot, N=N, n_runs=len(x), n_censored=int(cens.sum()),
                     censored_m0_scale=','.join(sorted(set(map(str, g.m0_scale.values[cens])))) or '-',
                     a_tobit=a, b_tobit=b, b_se_obsinfo=se_b, b_se_boot=bb.std(ddof=1), n_boot=len(bb),
                     b_boot_p16=np.percentile(bb, 16), b_boot_p84=np.percentile(bb, 84),
                     sigma_tobit=np.exp(ls), kappa_tobit=2 - b, kappa_ode=ok, diff_tobit_minus_ode=2 - b - ok,
                     diff_in_se=(2 - b - ok) / se_b,
                     b_ols_reached=bo, b_se_ols=se_o, kappa_ols_reached=2 - bo, diff_ols_minus_ode=2 - bo - ok))
kt = pd.DataFrame(rows)
kt.to_csv(R + 'exp1b/kappa_tobit.csv', index=False, float_format='%.5g')
pd.set_option('display.width', 250, 'display.max_columns', 50)
print(kt[['protocol', 'N', 'n_censored', 'b_tobit', 'b_se_obsinfo', 'b_se_boot', 'kappa_tobit', 'kappa_ode', 'diff_tobit_minus_ode',
          'kappa_ols_reached', 'b_se_ols', 'diff_ols_minus_ode', 'sigma_tobit']].round(3).to_string())
print('max |tobit-ode|', kt.diff_tobit_minus_ode.abs().max(), ' max |ols-ode|', kt.diff_ols_minus_ode.abs().max())

# ---------------- ODE check of ODE_KAPPA (script values) ----------------
d = 32; m0s = np.array([.5, .7, 1, 1.4]) / np.sqrt(d)
for lab, rhs, r in [('free1', rhs_free, 1.), ('free10', rhs_free, 10.), ('tied', rhs_tied, None)]:
    out = []
    for N in [32, 128, 512, 4096]:
        T = np.array([T05(rhs, m, .01, N, r) for m in m0s]); out.append(round(2 - np.polyfit(np.log(m0s), np.log(T), 1)[0], 2))
    print('ODE kappa recomputed', lab, out)

# ---------------- ratio table: exp 1b ----------------
def boot_med(r, nb=5000):
    r = np.asarray(r); m = np.median(rng.choice(r, (nb, len(r)), replace=True), axis=1)
    return np.percentile(m, 16), np.percentile(m, 84)

rr = []
cfg = {'free1': (rhs_free, 1.), 'free10': (rhs_free, 10.), 'tied': (rhs_tied, None)}
for (prot, N, ms), g in k.groupby(['protocol', 'N', 'm0_scale'], sort=False):
    rhs, r = cfg[prot]
    ode_steps = np.array([T05(rhs, m, .01, N, r) / ETA for m in g.m0.values])
    nreach = int(g.reached05.sum())
    ratios = np.where(g.reached05.values, g.T05.values / ode_steps, np.inf)   # censored -> +inf (lower bound ratio from cap below)
    ratio_lb = g.steps.values / ode_steps                                        # censored: ratio of cap to ODE (lower bound)
    if nreach == 3:
        lo, hi = boot_med(ratios); med = np.median(ratios)
        status = 'uncensored'
    else:
        lo = hi = np.nan; med = np.nan; status = f'censored({nreach}/3 reached)'
    rr.append(dict(experiment='exp1b', protocol=prot, N=N, scheme_or_m0scale=ms, B=int(g.B.iloc[0]), m0=g.m0.mean(), n_seeds=3, n_reached=nreach,
                   status=status, T_sgd_seeds='/'.join('%.0f' % t if np.isfinite(t) else 'cens>=%d' % c for t, c in zip(np.where(g.reached05, g.T05, np.nan), g.steps)),
                   T_sgd_median=np.median(g.T05.values) if nreach == 3 else np.nan, T_ode_steps=ode_steps.mean(),
                   ratio_median=med, ratio_boot68_lo=lo, ratio_boot68_hi=hi,
                   ratio_min=ratios.min() if nreach == 3 else np.nan, ratio_max=ratios.max() if nreach == 3 else np.nan,
                   ratio_cap_over_ode_lowerbound_for_censored=np.min(ratio_lb[~g.reached05.values]) if nreach < 3 else np.nan,
                   in_summary=(nreach == 3)))

# ---------------- ratio table: exp 6 ----------------
e = pd.read_csv(R + 'exp6/summary.csv')
e = e[np.isfinite(e.ode_steps)]
# drop censored 'main' free N=16 rows superseded by the 2.5e6 reruns ('ext'); keep every cell once
e = e[~((e.protocol == 'free') & (e.N == 16) & (e.variant == 'main'))]
for _, q in e.iterrows():
    t = np.array([float(v) for v in q.T05_seeds.split('/')])
    ratios = t / q.ode_steps
    lo, hi = boot_med(ratios)
    dup = (q.N == 64)   # tok and B64 are the same runs at N=64
    rr.append(dict(experiment='exp6', protocol={'fixed01': 'pinned0.1'}.get(q.protocol, q.protocol), N=q.N, scheme_or_m0scale=q.scheme, B=q.B, m0=0.125, n_seeds=3,
                   n_reached=int(q.n_reached), status='uncensored (rerun 2.5e6 cap)' if q.variant == 'ext' else 'uncensored',
                   T_sgd_seeds=q.T05_seeds, T_sgd_median=np.median(t), T_ode_steps=q.ode_steps,
                   ratio_median=np.median(ratios), ratio_boot68_lo=lo, ratio_boot68_hi=hi, ratio_min=ratios.min(), ratio_max=ratios.max(),
                   ratio_cap_over_ode_lowerbound_for_censored=np.nan, in_summary=True,
                   note='duplicate of the other scheme (identical runs at N=64)' if dup else ''))
rt = pd.DataFrame(rr)
rt.to_csv(R + 'ratios_sgd_vs_ode.csv', index=False, float_format='%.5g')

def summ(df, label):
    r = df.ratio_median; dev = (r - 1).abs()
    print(f'{label}: n={len(df)}  within10={int((dev<=.1).sum())} within20={int((dev<=.2).sum())} within50={int((dev<=.5).sum())}  '
          f'min={r.min():.3f} max={r.max():.3f} median={r.median():.3f}  seed-level min={df.ratio_min.min():.3f} max={df.ratio_max.max():.3f}')
S = rt[rt.in_summary]
summ(S, 'ALL'); summ(S[S.experiment == 'exp1b'], 'exp1b'); summ(S[S.experiment == 'exp6'], 'exp6')
summ(S[(S.experiment == 'exp6') & (S.note == '')], 'exp6 dedup (N=64 once)')
for p in ['free1', 'free10', 'tied']: summ(S[(S.experiment == 'exp1b') & (S.protocol == p)], 'exp1b ' + p)
for p in ['pinned0.1', 'free', 'tied']: summ(S[(S.experiment == 'exp6') & (S.protocol == p)], 'exp6 ' + p)
print(rt[~rt.in_summary][['protocol', 'N', 'scheme_or_m0scale', 'n_reached', 'T_sgd_seeds', 'T_ode_steps', 'ratio_cap_over_ode_lowerbound_for_censored']].to_string())
print(rt[rt.experiment == 'exp1b'][['protocol', 'N', 'scheme_or_m0scale', 'T_sgd_median', 'T_ode_steps', 'ratio_median', 'ratio_boot68_lo', 'ratio_boot68_hi', 'ratio_min', 'ratio_max']].round(3).to_string())
