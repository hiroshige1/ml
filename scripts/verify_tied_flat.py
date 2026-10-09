"""Numerical check of Theorem A(iii)(c) (tied readout: N-flat escape for small rho0), backlog N13.

Reuses the function definitions of scripts/verify_theorems.py (functions only; its checks are not re-run), s^2 = 1:
  G = m^{2k}, D = (1-1/N) m^{2k} + V(m)/N,  tied flow  dm/dt = (1-m^2)(2G' - rho D'),  drho/dt = 8 rho (G - rho D),
  rho*(m) = 2G'/D',  q = rho/rho*(m),  tau = first time m = 1/2,  X = (2k-2) 4k m0^{2k-2}  (so X*tau -> 1 as rho0, m0 -> 0).
Theorem A(iii)(c): if m0 in (0,1/2) and rho0 <= rho*(m0)/4, then until m = 1/2
  (R0) q < 1/3 (so m is increasing)
  (R1) rho >= rho0 / (1 + c1 rho0/(N m0^{2k-2})),       c1 = 2V(1/2)/(k(k-1))
  (R2) rho <= rho0 ((1-m0^2)/(1-m^2))^{3/(2k)} <= (4/3)^{3/(2k)} rho0
  (T1) X tau >= (1-(2m0)^{2k-2})(1 + rho_min(1-1/N)/2) + rho_min k (1-(2m0)^{4k-4}) / (2N m0^{2k-2})
  (T2) X tau <= 1 + (4/3) l_k m0^2 + rho_max + rho_max k vbar / (N m0^{2k-2}),  vbar = V'(1/2)/(2k^2),
       l_k = 2 ln(1/(2m0)) (k=2),  (k-1)(1-(2m0)^{2k-4})/(k-2) (k>=3)
  (T3) |X tau - 1| <= C_k (k rho0/(N m0^{2k-2}) + rho0 + lam_k m0^2),  C_k = max(4, (4/3)^{3/(2k)} vbar),
       lam_2 = 1 + ln(1/(2m0)),  lam_k = 1 (k>=3).
Run from /home/user/ml:  python3 -I scripts/verify_tied_flat.py   -> stdout and results/verify_tied_flat.txt
"""
import ast, math, os, io, contextlib
import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
tree = ast.parse(open(os.path.join(HERE, 'verify_theorems.py')).read())
ns = {'np': np, 'math': math, 'solve_ivp': solve_ivp}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))], type_ignores=[]), 'vt', 'exec'), ns)
run_tied, rho_star, Vfun, beta2 = (ns[k] for k in ['run_tied', 'rho_star', 'Vfun', 'beta2'])

def consts(k):
    c1 = 2 * Vfun(k, 0.5) / (k * (k - 1))
    vbar = Vfun(k, 0.5, 1) / (2 * k * k)
    up = (4 / 3) ** (3 / (2 * k))
    return c1, vbar, up, max(4.0, up * vbar)

# (k, d, rho0, N, source) -- every tied cell used in the paper
CELLS = ([(2, 64, r, N, 'E6/E7 d=64') for r in (0.01, 0.1) for N in (16, 64, 256)]
         + [(2, 128, 0.01, N, 'E7 d=128') for N in (16, 64, 256)]
         + [(2, 32, 1.0, N, 'E10 d=32 rho0=1') for N in (32, 128, 512, 2048)]
         + [(3, d, 0.01, N, 'E9 k=3') for d in (8, 16, 32) for N in (16, 128)])

out = io.StringIO()
def say(*a):
    print(*a); print(*a, file=out)

say("Theorem A(iii)(c) verification (tied flow, LSODA rtol=1e-10, s^2=1); constants:")
for k in (2, 3):
    c1, vbar, up, C = consts(k)
    say(f"  k={k}: c1=2V(1/2)/(k(k-1))={c1:.4f}  (4/3)^(3/2k)={up:.4f}  vbar=V'(1/2)/(2k^2)={vbar:.4f}  C_k={C:.4f}")
say("")
hdr = (f"{'k':>1} {'d':>4} {'rho0':>5} {'N':>5} {'q0':>6} {'hyp':>3} {'eps=krho0/Nm0^(2k-2)':>9} {'rho_min/rho0':>12} {'R1 low':>7} "
       f"{'rho_max/rho0':>12} {'R2 up':>6} {'max q':>6} {'X tau':>8} {'T1 low':>8} {'T2 up':>8} {'|Xt-1|':>7} {'T3 bnd':>7}  result")
say(hdr)
APRI = {}; n_pass = n_chk = 0; n_hyp = 0; fails = []
for (k, d, rho0, N, src) in CELLS:
    m0 = d ** -0.5; e = 2 * k - 2
    c1, vbar, up, C = consts(k)
    q0 = rho0 / rho_star(m0, N, k)
    hyp = (m0 < 0.5) and (q0 <= 0.25)
    sol = run_tied(m0, rho0, N, k=k, T=1e9, dense=True)
    eps = k * rho0 / (N * m0 ** e)
    if sol.t_events[0].size == 0:
        say(f"{k:>1} {d:>4} {rho0:>5g} {N:>5} {q0:6.3f} {'no':>3} {eps:9.3g}   no escape by t=1e9 (trapped)  [{src}; hypothesis violated]")
        continue
    tau = sol.t_events[0][0]
    tt = np.unique(np.concatenate([sol.t[sol.t <= tau], np.linspace(0, tau, 20001)]))
    Y = sol.sol(tt); mm, rr = np.exp(Y[0]), np.exp(Y[1])
    rmin, rmax = rr.min(), rr.max()
    qmax = np.max(rr / rho_star(mm, N, k))
    R1 = rho0 / (1 + c1 * rho0 / (N * m0 ** e))
    R2pt = rho0 * ((1 - m0 ** 2) / (1 - mm ** 2)) ** (3 / (2 * k))
    X = e * 4 * k * m0 ** e; Xt = X * tau
    lk = 2 * math.log(1 / (2 * m0)) if k == 2 else (k - 1) * (1 - (2 * m0) ** (2 * k - 4)) / (k - 2)
    lam = 1 + math.log(1 / (2 * m0)) if k == 2 else 1.0
    T1 = (1 - (2 * m0) ** e) * (1 + rmin * (1 - 1 / N) / 2) + rmin * k * (1 - (2 * m0) ** (2 * e)) / (2 * N * m0 ** e)
    T2 = 1 + (4 / 3) * lk * m0 ** 2 + rmax + rmax * k * vbar / (N * m0 ** e)
    T3 = C * (eps + rho0 + lam * m0 ** 2)
    # a-priori sandwich: T1 with rho_min -> R1, T2 with rho_max -> (4/3)^{3/2k} rho0 (no knowledge of the trajectory)
    T1a = (1 - (2 * m0) ** e) * (1 + R1 * (1 - 1 / N) / 2) + R1 * k * (1 - (2 * m0) ** (2 * e)) / (2 * N * m0 ** e)
    T2a = 1 + (4 / 3) * lk * m0 ** 2 + up * rho0 * (1 + k * vbar / (N * m0 ** e))
    APRI[(k, d, rho0, N)] = (T1a / X, T2a / X, hyp, tau)
    checks = {'R0 q<1/3': qmax < 1 / 3, 'R1': rmin >= R1, 'R2 pointwise': bool(np.all(rr <= R2pt * (1 + 1e-9))),
              'R2 global': rmax <= up * rho0, 'T1': Xt >= T1, 'T2': Xt <= T2, 'T1a,T2a': T1a <= Xt <= T2a, 'T3': abs(Xt - 1) <= T3}
    if hyp:
        n_hyp += 1; n_chk += len(checks); n_pass += sum(checks.values())
        bad = [c for c, v in checks.items() if not v]; fails += [(k, d, rho0, N, c) for c in bad]
        res = 'PASS 8/8' if not bad else 'FAIL ' + ','.join(bad)
    else:
        held = [c for c, v in checks.items() if v]
        res = f"hypothesis violated (q0>1/4{' or m0>=1/2' if m0 >= 0.5 else ''}); bounds holding anyway: {len(held)}/8" + \
              ('' if len(held) == 8 else ' [fail: ' + ','.join(c for c, v in checks.items() if not v) + ']')
    say(f"{k:>1} {d:>4} {rho0:>5g} {N:>5} {q0:6.3f} {'yes' if hyp else 'no':>3} {eps:9.3g} {rmin/rho0:12.4f} {R1/rho0:7.4f} "
        f"{rmax/rho0:12.4f} {up:6.4f} {qmax:6.3f} {Xt:8.4f} {T1:8.4f} {T2:8.4f} {abs(Xt-1):7.4f} {T3:7.4f}  {res}  [{src}]")
say("")
say(f"cells satisfying the hypothesis: {n_hyp}/{len(CELLS)};  checks passed on them: {n_pass}/{n_chk}")
if fails: say("FAILURES:", fails)
# N-ratios of the tied escape time in the covered blocks (the N-flatness the theorem bounds)
say("\nN-ratios tau(N_lo)/tau(N_hi) (exact flow) and the a-priori bracket [T1a(N_lo)/T2a(N_hi), T2a(N_lo)/T1a(N_hi)] from the theorem:")
for (k, d, rho0, Ns) in [(2, 64, 0.01, (16, 256)), (2, 64, 0.1, (16, 256)), (2, 128, 0.01, (16, 256)), (3, 8, 0.01, (16, 128)),
                         (3, 16, 0.01, (16, 128))]:
    m0 = d ** -0.5; (l1, u1, h1, t1), (l2, u2, h2, t2) = APRI[(k, d, rho0, Ns[0])], APRI[(k, d, rho0, Ns[1])]
    br = f"[{l1/u2:.3f}, {u1/l2:.3f}]" if (h1 and h2) else "(hypothesis fails at N_lo: no bracket)"
    say(f"  k={k} d={d:3d} rho0={rho0:g}: tau(N={Ns[0]})/tau(N={Ns[1]}) = {t1/t2:.4f}  {br}   k rho0/(N_lo m0^(2k-2)) = {k*rho0/(Ns[0]*m0**(2*k-2)):.3g}")
with open(os.path.join(ROOT, 'results', 'verify_tied_flat.txt'), 'w') as f:
    f.write(out.getvalue())
