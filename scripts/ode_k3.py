"""Population-ODE predictions for experiment 9 (backlog N3): sigma_3 (k*=3) sanity run, fixed BEFORE any SGD run.

Uses the general-k flows of scripts/verify_theorems.py (functions only; the checks are not re-run):
  g = m^3, V(m) = 1 + 18 m^2 + 54 m^4 + 20 m^6, L = 1 - 2 G g^2 + G^2((1-1/N) g^2 + V/N), s^2 = 1.
  pinned: dm/dt = -(1-m^2) dL/dm (Gamma fixed);  tied: dm/dt = -(1-m^2) dL/dm / rho,  drho/dt = -4 rho dL/drho.
Flow time tau -> SGD steps = tau/eta (eta = 2e-4 for every cell, so cross-d comparisons are not confounded by eta).
Run from /home/user/ml:  python scripts/ode_k3.py   -> results/exp9/ode_predictions.csv and stdout table.
"""
import ast, math, os, sys
import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
tree = ast.parse(open(os.path.join(HERE, 'verify_theorems.py')).read())
ns = {'np': np, 'math': math, 'solve_ivp': solve_ivp}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))], type_ignores=[]), 'vt', 'exec'), ns)
dLdm, run_tied, tau_ode_pinned, rho_trap, beta2 = (ns[k] for k in ['dLdm', 'run_tied', 'tau_ode_pinned', 'rho_trap', 'beta2'])
K, ETA, DS = 3, 2e-4, (8, 16, 32)

def mstar(g, N):
    ms = np.linspace(1e-4, 0.99, 200000); dr = -dLdm(ms, g, N, K)
    i = np.where(np.diff(np.sign(dr)) != 0)[0]
    return float(ms[i[0]]) if i.size else float('nan')

rows = []
def add(cell, N, d, tau, extra=""):
    steps = tau / ETA if np.isfinite(tau) else float('inf')
    rows.append(dict(cell=cell, N=N, d=d, m0=d ** -0.5, tau_flow=tau, steps_eta2e4=steps, note=extra))
    print(f"{cell:22s} N={N:4d} d={d:3d} m0={d**-0.5:.4f}  tau={tau:10.4g}  steps={steps:10.4g}  {extra}")

print("beta_{2i}^2 (k=3):", beta2(K), "  V(1) =", beta2(K).sum())
taus = {}
for N in (128, 16):
    for d in DS:
        m0 = d ** -0.5; s = run_tied(m0, 0.01, N, k=K, T=1e9)
        tau = s.t_events[0][0] if s.t_events[0].size else float('inf')
        rho_T = float(np.exp(s.y_events[0][0][1])) if s.t_events[0].size else float('nan')
        taus[(N, d)] = tau
        add(f"tied rho0=0.01", N, d, tau, f"rho_trap={rho_trap(K, N, m0):.4g} rho_at_T={rho_T:.4g}")
for N in (128, 16):
    for a, b in ((8, 16), (16, 32)):
        if np.isfinite(taus[(N, a)]) and np.isfinite(taus[(N, b)]):
            kap = 2 - math.log(taus[(N, b)] / taus[(N, a)]) / math.log(math.sqrt(a / b))
            print(f"   secant kappa_eff tied N={N} d={a}-{b}: {kap:.3f}"); rows.append(dict(cell=f"kappa tied N={N} d={a}-{b}", N=N, d=b, m0=float('nan'), tau_flow=float('nan'), steps_eta2e4=float('nan'), note=f"{kap:.4f}"))
for d in (8, 16):
    print(f"   ratio N=16/N=128 tied d={d}: {taus[(16, d)]/taus[(128, d)]:.3f}")
g, N = 0.2, 128; ms = mstar(g, N); print(f"pinned gamma={g} N={N}: m* = {ms:.4f}")
for d in DS:
    m0 = d ** -0.5; add(f"pinned gamma={g}", N, d, tau_ode_pinned(m0, g, N, k=K, T=1e9), f"m0/m*={m0/ms:.3f}")
os.makedirs(os.path.join(ROOT, 'results', 'exp9'), exist_ok=True)
import csv
with open(os.path.join(ROOT, 'results', 'exp9', 'ode_predictions.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
