"""Numerical/symbolic verification of every closed form in paper/theorems.tex.  Run: python3 -I scripts/verify_theorems.py
Conventions: s^2 = 1 (time scales as 1/s^2), sigma = sigma_k = He_k/sqrt(k!), g = m^k, V(m) = sum_i beta_{2i}^2 m^{2i}.
Population loss L(m,G) = 1 - 2 G m^{2k} + G^2 [ (1-1/N) m^{2k} + V(m)/N ].
"""
import math, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np, sympy as sp
from scipy.integrate import solve_ivp, quad
from numpy.polynomial import hermite_e as He

np.set_printoptions(precision=4)
ok_all = []
def check(name, cond, info=""):
    ok_all.append(bool(cond)); print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

# ---------------------------------------------------------------- Hermite algebra
def beta2(k):
    """beta_{2i}^2, i=0..k, for sigma_k^2 = sum_j beta_j sigma_j (linearisation formula), checked by quadrature below."""
    out = []
    for i in range(k + 1):
        r = k - i
        out.append((math.factorial(r) * math.comb(k, r) ** 2) ** 2 * math.factorial(2 * i) / math.factorial(k) ** 2)
    return np.array(out, float)

z, w = np.polynomial.hermite_e.hermegauss(80); w = w / w.sum()
def sig(k, x): return He.hermeval(x, [0] * k + [1]) / math.sqrt(math.factorial(k))
print("== Hermite coefficients of sigma_k^2 (quadrature vs formula)")
for k in (2, 3, 4):
    q = np.array([np.sum(w * sig(k, z) ** 2 * sig(2 * i, z)) ** 2 for i in range(k + 1)])
    b = beta2(k)
    check(f"k={k}: beta_2^2=2k^2, beta_2k^2=C(2k,k), all beta_2i^2", np.allclose(q, b) and np.isclose(b[1], 2 * k * k)
          and np.isclose(b[k], math.comb(2 * k, k)), f"beta_2i^2={b}")
    Vp1 = sum(2 * i * b[i] for i in range(k + 1)); E = np.sum(w * sig(k, z) ** 2 * sig(k - 1, z) ** 2)
    check(f"k={k}: V'(1) = 4k E[s_k^2 s_(k-1)^2]", np.isclose(Vp1, 4 * k * E), f"V'(1)={Vp1:.4g}")
    # V(m) by 2-D quadrature at m=0.6
    mm = 0.6; Z1, Z2 = np.meshgrid(z, z); W = np.outer(w, w)
    Vq = np.sum(W * sig(k, Z1) ** 2 * sig(k, mm * Z1 + math.sqrt(1 - mm * mm) * Z2) ** 2)
    check(f"k={k}: V(0.6) series vs 2-D quadrature", np.isclose(Vq, sum(b[i] * mm ** (2 * i) for i in range(k + 1)), rtol=1e-8))

def Vfun(k, m, d=0):
    b = beta2(k); c = np.zeros(2 * k + 1); c[0::2] = b
    p = np.polynomial.Polynomial(c)
    return (p.deriv(d) if d else p)(m)

def dLdm(m, G, N, k):
    return -4 * k * G * m ** (2 * k - 1) + G * G * (2 * k * (1 - 1 / N) * m ** (2 * k - 1) + Vfun(k, m, 1) / N)
def dLdG(m, G, N, k):
    return -2 * m ** (2 * k) + 2 * G * ((1 - 1 / N) * m ** (2 * k) + Vfun(k, m) / N)

# ---------------------------------------------------------------- Theorem A(i)
print("\n== Theorem A(i): pinned threshold")
mS, gS, NS = sp.symbols('m gamma N', positive=True)
L2 = 1 - 2 * gS * mS ** 4 + gS ** 2 * ((1 - 1 / NS) * mS ** 4 + (1 + 8 * mS ** 2 + 6 * mS ** 4) / NS)
drift = sp.expand(-sp.diff(L2, mS))
C = gS * (8 - 4 * gS - 20 * gS / NS); B = 16 * gS ** 2 / NS
check("k=2 drift = C m^3 - B m", sp.simplify(drift - (C * mS ** 3 - B * mS)) == 0)
check("k=2 m*^2 = B/C = 4g/(2N-(N+5)g)", sp.simplify(B / C - 4 * gS / (2 * NS - (NS + 5) * gS)) == 0)
check("k=2: m*<1 iff g < 2N/(N+9)", sp.simplify((C - B).factor() - gS * (8 * NS - 4 * gS * NS - 36 * gS) / NS) == 0)

def Pcoef(g, N, k):
    """coefficients (low->high in x=m^2) of -dL/dm / m."""
    b = beta2(k); a = 2 * k * g * (2 - g * (1 - 1 / N))
    c = np.array([-(g * g / N) * (2 * i + 2) * b[i + 1] for i in range(k)])
    c[k - 1] += a
    return c
for k in (2, 3, 4):
    bad = 0; n = 0
    for N in (4, 16, 128, 1024, 10 ** 5):
        Vp1 = sum(2 * i * beta2(k)[i] for i in range(k + 1)); g1 = 2 * N / (N - 1 + Vp1 / (2 * k))
        for g in np.linspace(0.01, 2.2, 60):
            r = np.polynomial.Polynomial(Pcoef(g, N, k)).roots()
            pos = [x.real for x in r if abs(x.imag) < 1e-9 and x.real > 0]
            n += 1
            inside = [x for x in pos if x < 1]
            if len(pos) > 1 or ((g < g1) != (len(inside) == 1)): bad += 1
    check(f"k={k}: <=1 positive root (Descartes) and root in (0,1) iff gamma<gamma_1", bad == 0, f"({n} (gamma,N) pairs)")
for k in (2, 3, 4):
    rat = []
    for N in (1e2, 1e4, 1e6, 1e8, 1e12):
        g = 0.5; r = np.polynomial.Polynomial(Pcoef(g, N, k)).roots()
        x = min(t.real for t in r if abs(t.imag) < 1e-9 and t.real > 0)
        rat.append(x ** (k - 1) / (2 * k * g / ((2 - g) * N)))
    check(f"k={k}: m*^(2(k-1)) / [2k g/((2-g)N)] -> 1 as N->inf (N=1e2..1e12)", abs(rat[-1] - 1) < 1e-2, f"{np.array(rat)}")
r = np.polynomial.Polynomial(Pcoef(0.1, 128, 3)).roots(); x3 = [t.real for t in r if abs(t.imag) < 1e-9 and t.real > 0][0]
print(f"   k=3, gamma=0.1, N=128: exact m* = {math.sqrt(x3):.4f}, leading order {(6*0.1/(1.9*128))**0.25:.4f}, d* = {1/x3:.1f}")

# ---------------------------------------------------------------- Theorem A(ii)
print("\n== Theorem A(ii): k=2 escape time, partial fractions vs LSODA")
def tau_pf(m0, g, N, me=0.5):
    Cn = g * (8 - 4 * g - 20 * g / N); Bn = 16 * g * g / N
    F = lambda u: -np.log(u) / Bn - np.log(1 - u) / (Cn - Bn) + Cn * np.log(Cn * u - Bn) / (Bn * (Cn - Bn))
    return 0.5 * (F(me ** 2) - F(m0 ** 2))
def tau_ode_pinned(m0, g, N, k=2, me=0.5, T=1e14):
    f = lambda t, y: [-(1 - np.exp(2 * y[0])) * dLdm(np.exp(y[0]), g, N, k) / np.exp(y[0])]
    ev = lambda t, y: y[0] - np.log(me); ev.terminal = True
    s = solve_ivp(f, (0, T), [np.log(m0)], method='LSODA', events=ev, rtol=1e-11, atol=1e-13)
    return s.t_events[0][0] if s.t_events[0].size else np.inf
for (d, N, g) in [(512, 128, 0.1), (512, 512, 0.1), (64, 64, 0.1), (64, 16, 0.01), (32, 128, 1.0), (16, 128, 1.0)]:
    m0 = d ** -0.5; ms2 = 4 * g / (2 * N - (N + 5) * g)
    if m0 ** 2 <= ms2:
        t = tau_ode_pinned(m0, g, N, T=1e7)
        check(f"d={d} N={N} g={g}: m0<m* -> no escape", not np.isfinite(t), f"(m0^2={m0**2:.4g} <= m*^2={ms2:.4g})"); continue
    a, b = tau_pf(m0, g, N), tau_ode_pinned(m0, g, N)
    check(f"d={d} N={N} g={g}", abs(a / b - 1) < 1e-7, f"closed form {a:.6g}  LSODA {b:.6g}  1/(2C m0^2)={1/(2*g*(8-4*g-20*g/N)*m0**2):.6g}")
m0 = math.sqrt(4 * 0.1 / (2 * 128 - 133 * 0.1) * (1 + 1e-4))
check("near threshold (u0 = x*(1+1e-4))", abs(tau_pf(m0, 0.1, 128) / tau_ode_pinned(m0, 0.1, 128) - 1) < 1e-6, f"tau={tau_pf(m0, 0.1, 128):.7g} LSODA {tau_ode_pinned(m0, 0.1, 128):.7g}")

# ---------------------------------------------------------------- Theorem A(iii)
print("\n== Theorem A(iii): tied readout")
rS = sp.symbols('rho', positive=True)
Lr = L2.subs(gS, rS)
mdot_tied = (1 - mS ** 2) * (4 * mS ** 2 * 2 * mS - rS * (2 * (1 - 1 / NS) * mS ** 2 * 2 * mS + (16 * mS + 24 * mS ** 3) / NS))
mdot_pin = -(1 - mS ** 2) * sp.diff(L2, mS)
check("identity mdot_tied(m,rho) = mdot_pinned(m; gamma=rho)/rho (sympy, k=2)", sp.simplify(mdot_tied - mdot_pin.subs(gS, rS) / rS) == 0)
# chain rule from u-space: rho = |u|^2, w = u/|u| gives  rho' = -4 rho dL/drho,  m' = -(1-m^2) dL/dm / rho
check("rho' = -4 rho dL/drho reproduces 8 rho (G - rho D)", sp.simplify(-4 * rS * sp.diff(Lr, rS) - 8 * rS * (mS ** 4 - rS * ((1 - 1 / NS) * mS ** 4 + (1 + 8 * mS ** 2 + 6 * mS ** 4) / NS))) == 0)

def tied_rhs(k, N):
    def f(t, y):
        m, rho = np.exp(y[0]), np.exp(y[1])
        return [-(1 - m * m) * dLdm(m, rho, N, k) / (rho * m), -4 * dLdG(m, rho, N, k)]
    return f
def run_tied(m0, rho0, N, k=2, T=1e8, me=0.5, dense=False):
    ev = lambda t, y: y[0] - np.log(me); ev.terminal = True
    return solve_ivp(tied_rhs(k, N), (0, T), [np.log(m0), np.log(rho0)], method='LSODA', events=ev, rtol=1e-10, atol=1e-12, dense_output=dense)
# time-change: integrate the pinned flow with gamma = rho(t) in the pinned clock t, carrying t' (dt' = rho dt)
d, N, rho0 = 64, 16, 0.1; m0 = d ** -0.5
st = run_tied(m0, rho0, N, dense=True)
def pin_tc(t, y):
    m, rho = np.exp(y[0]), np.exp(y[1])
    return [-(1 - m * m) * dLdm(m, rho, N, 2) / m, -4 * rho * dLdG(m, rho, N, 2), rho]
ev = lambda t, y: y[0] - np.log(0.5); ev.terminal = True
sp_ = solve_ivp(pin_tc, (0, 1e9), [np.log(m0), np.log(rho0), 0.0], method='LSODA', events=ev, rtol=1e-11, atol=1e-13)
check("tied(t') == pinned with gamma=rho(t), dt'=rho dt (escape-time match)", abs(sp_.y_events[0][0][2] / st.t_events[0][0] - 1) < 1e-6,
      f"tied tau'={st.t_events[0][0]:.6g}, int rho dt={sp_.y_events[0][0][2]:.6g}")
print("   frozen-rho consequence tau_tied ~ rho0 tau_pinned(rho0) (d=64, rho0=gamma=0.01):")
for N in (16, 64, 256):
    tt = run_tied(1 / 8, 0.01, N).t_events[0][0]; tp = tau_pf(1 / 8, 0.01, N)
    print(f"   N={N:4d}: tau_tied={tt:9.5g}  0.01*tau_pinned={0.01*tp:9.5g}  ratio={tt/(0.01*tp):.4f}  [rho0 d/N={0.01*64/N:.3g}]")
# permanent trap and asymptotics at the E5 cell (d=64, N=16, rho0=0.3)
N = 16
s = solve_ivp(tied_rhs(2, N), (0, 1e10), [np.log(1 / 8), np.log(0.3)], method='LSODA', rtol=1e-11, atol=1e-13,
              t_eval=[1e2, 1e4, 1e6, 1e8, 1e10])
mt, rt = np.exp(s.y[0]), np.exp(s.y[1])
rs = 2 * N * mt ** 2 / (4 + (N + 5) * mt ** 2)
print("   t        m           rho        m t^2       rho t*8/N   rho/rho*(m)")
for t, a, b, c in zip(s.t, mt, rt, rs): print(f"   {t:7.0e}  {a:10.3e}  {b:10.3e}  {a*t*t:10.5g}  {8*b*t/N:10.6f}  {b/c:9.3e}")
check("trap: m t^2 -> const, 8 rho t/N -> 1", abs(mt[-1] * 1e20 / (mt[-2] * 1e16) - 1) < 1e-3 and abs(8 * rt[-1] * 1e10 / N - 1) < 1e-6)
# the trap boundary is rho0 = c_k rho*(m0), c_k = 1 + 1/(k^3-k^2-1), not rho0 = rho*(m0)
def rho_star(m, N, k):
    return 4 * k * m ** (2 * k - 1) / (2 * k * (1 - 1 / N) * m ** (2 * k - 1) + Vfun(k, m, 1) / N)
for (k, m0, N) in [(2, 1e-3, 16), (2, 1e-2, 64), (3, 1e-2, 16), (2, 1 / 8, 16)]:
    lo, hi = 1.0, 3.0
    for _ in range(40):
        q = 0.5 * (lo + hi); sol = run_tied(m0, q * rho_star(m0, N, k), N, k=k, T=1e30)
        trapped = sol.t_events[0].size == 0
        lo, hi = (lo, q) if trapped else (q, hi)
    ck = 1 + 1 / (k ** 3 - k ** 2 - 1)
    tol = 2e-2 if m0 < 0.05 else 1.0
    check(f"k={k} m0={m0:g} N={N}: trap boundary rho0/rho*(m0)", abs(lo / ck - 1) < tol, f"= {lo:.4f} (small-m prediction c_k={ck:.4f})")
# explicit sufficient trap condition (eq. trapcond): rho0 > rho_trap(m0) = k(k-1) m0^(2k-2) / (D(m0)(a-1)), a = k^2(k-1)(1-m0^2)/(N D(m0)) > 1
def rho_trap(k, N, m0):
    D0 = (1 - 1 / N) * m0 ** (2 * k) + Vfun(k, m0) / N; a = k * k * (k - 1) * (1 - m0 * m0) / (N * D0)
    return k * (k - 1) * m0 ** (2 * k - 2) / (D0 * (a - 1)) if a > 1 else np.inf
for (k, N, m0) in [(2, 16, 1 / 8), (2, 16, 1e-2), (2, 128, 1e-2), (3, 16, 1e-2), (2, 16, 1e-3)]:
    rt = rho_trap(k, N, m0); ck = 1 + 1 / (k ** 3 - k ** 2 - 1)
    sol = run_tied(m0, 1.0001 * rt, N, k=k, T=1e30, dense=False)
    trapped = sol.t_events[0].size == 0; mono = np.all(np.diff(sol.y[0]) <= 1e-12)
    check(f"k={k} N={N} m0={m0:.4g}: rho0=1.0001 rho_trap traps, m non-increasing", trapped and mono,
          f"rho_trap={rt:.4g} = {rt/rho_star(m0, N, k):.4f} rho*(m0)  [c_k={ck:.4f}]")
check("E5 cell (d=64, N=16): rho0=0.3 > rho_trap", 0.3 > rho_trap(2, 16, 1 / 8), f"rho_trap={rho_trap(2, 16, 1/8):.4f}, rho*={rho_star(1/8, 16, 2):.4f}")

# ---------------------------------------------------------------- Theorem A(iv)
print("\n== Theorem A(iv): B does not enter the mean minibatch gradient (MC, k=2, d=8, N=8, gamma=0.5)")
rng = np.random.default_rng(0); dd, NN, gg = 8, 8, 0.5
v = np.zeros(dd); v[0] = 1; mtrue = 0.4; wv = np.zeros(dd); wv[0] = mtrue; wv[1] = math.sqrt(1 - mtrue ** 2)
s2 = lambda x: (x * x - 1) / math.sqrt(2)
def prompt_grad_m(nprompt):
    c = rng.standard_normal(nprompt)[:, None]
    X = rng.standard_normal((nprompt, NN, dd)); xq = rng.standard_normal((nprompt, dd))
    y = c * s2(X @ v); A = np.mean(y * s2(X @ wv), 1); yq = c[:, 0] * s2(xq @ v)
    yhat = gg * s2(xq @ wv) * A; res = yhat - yq
    dA = np.mean((y * (math.sqrt(2) * (X @ wv)))[..., None] * X, 1)
    gw = 2 * res[:, None] * gg * (math.sqrt(2) * (xq @ wv)[:, None] * xq * A[:, None] + s2(xq @ wv)[:, None] * dA)
    return gw @ v - (gw @ wv) * mtrue  # tangential component along v
gall = np.concatenate([prompt_grad_m(32768) for _ in range(64)])
for Bsz in (1, 16, 256):
    gm = gall.reshape(-1, Bsz).mean(1)
    print(f"   B={Bsz:4d}: mean dm-gradient {gm.mean():+.5f} +- {gm.std()/math.sqrt(gm.size):.5f}, per-step sd {gm.std():.4f}")
pop = (1 - mtrue ** 2) * dLdm(mtrue, gg, NN, 2)
print(f"   population (1-m^2) dL/dm = {pop:+.5f}")
check("MC mean gradient = population gradient (|z|<3), sd ~ B^-1/2", abs(gall.mean() - pop) < 3 * gall.std() / math.sqrt(gall.size))

# ---------------------------------------------------------------- Theorem B(i)
print("\n== Theorem B(i): free readout, fixed N, d -> inf; tau_LSODA / tau_formula")
def free_rhs(k, N, r):
    def f(t, y):
        m, G = np.exp(y[0]), np.exp(y[1])
        return [-(1 - m * m) * dLdm(m, G, N, k) / m, -r * dLdG(m, G, N, k) / G]
    return f
class _Budget(Exception): pass
def _stage(k, N, r, y, target, span, method, budget=30000):
    f0 = free_rhs(k, N, r); cnt = [0]
    def f(t, yy):
        cnt[0] += 1
        if cnt[0] > budget: raise _Budget
        return f0(t, yy)
    ev = lambda t, yy: yy[0] - np.log(target); ev.terminal = True; ev.direction = 1
    s = solve_ivp(f, (0, span), y, method=method, events=ev, rtol=1e-11, atol=1e-13)
    return (s.t_events[0][0], s.y_events[0][0]) if s.t_events[0].size else None
def tau_free(m0, G0, N, r, k=2, me=0.5):
    """escape time; integrated in stages that restart the clock whenever m doubles (absolute times reach 1e22, where
    float spacing would otherwise exceed the O(1) time scale of the final escape). LSODA occasionally stalls for some
    spans, so each stage has an evaluation budget and falls back to other spans / Radau."""
    y, total, target = np.array([np.log(m0), np.log(G0)]), 0.0, min(2 * m0, me)
    while True:
        for span, meth in [(1e30, 'LSODA'), (1e26, 'LSODA'), (1e22, 'LSODA'), (1e30, 'Radau')]:
            try:
                out = _stage(k, N, r, y, target, span, meth)
                if out is not None: break
            except (_Budget, ValueError):
                out = None
        if out is None: raise RuntimeError("integration failed")
        total += out[0]; y = out[1]
        if target >= me: return total
        target = min(2 * target, me)
def tau_B(d, G0, N, r, k=2):
    b2 = 2 * k * k
    return math.exp((4 * k - 2) * b2 * G0 ** 2 / (2 * r)) * d ** (2 * k - 1) / (4 * k * (4 * k - 2) * N)
ds = [256, 1024, 4096, 16384, 65536]
for (k, N, r, G0) in [(2, 32, 1, 0.01), (2, 32, 1, 0.3), (2, 128, 1, 0.3), (2, 32, 10, 0.5), (2, 8, 0.3, 0.2), (2, 32, 1, 1.0), (3, 16, 1, 0.1), (3, 64, 3, 0.3)]:
    dl = ds if k == 2 else [64, 256, 1024, 4096]
    rat = [tau_free(d ** -0.5, G0, N, r, k) / tau_B(d, G0, N, r, k) for d in dl]
    check(f"k={k} N={N:3d} r={r:4} G0={G0:4}", abs(rat[-1] - 1) < 0.03 and abs(rat[-1] - 1) <= abs(rat[0] - 1) + 1e-3,
          "ratios at d=" + "/".join(map(str, dl)) + ": " + " ".join(f"{x:.4f}" for x in rat))

# ---------------------------------------------------------------- Theorem B(ii)-(iii)
print("\n== Proposition B(ii): joint scaling N = lam d^k, r = rt d; tau d^(1-k) -> T(mu0; lam, rt, G0)")
def T_limit(mu0, lam, rt, G0, k=2, big=1e2):
    """blow-up time of the limit system: integrate to mu = big, then add the tail int_big^inf dmu/(2k G(2-G) mu^(2k-1)) at the
    current G (G is within O(big^-2k) of its limit 1 there, so the tail error is negligible)."""
    def f(s, y):
        mu, G = np.exp(y[0]), np.exp(y[1]); il = 0.0 if lam == np.inf else 1 / lam
        return [4 * k * G * (1 - G / 2) * mu ** (2 * k - 2), 2 * rt * (mu ** (2 * k) * (1 - G) - G * il) / G]
    y, total, target = np.array([np.log(mu0), np.log(max(G0, 1e-300))]), 0.0, min(2 * mu0, big)
    while True:  # stages restart the clock each time mu doubles (see tau_free)
        for span in (1e30, 1e20, 1e12):
            ev = lambda s, y, tg=target: y[0] - np.log(tg); ev.terminal = True; ev.direction = 1
            try:
                s = solve_ivp(f, (0, span), y, method='LSODA', events=ev, rtol=1e-11, atol=1e-13)
                if s.t_events[0].size: break
            except ValueError:
                pass
        total += s.t_events[0][0]; y = s.y_events[0][0]
        if target >= big: break
        target = min(2 * target, big)
    G = np.exp(y[1])
    return total + big ** (2 - 2 * k) / ((2 * k - 2) * 2 * k * G * (2 - G))
for (lam, rt, G0) in [(1 / 32, 1 / 32, 0.01), (4.0, 1 / 32, 0.01), (1 / 32, 10 / 32, 0.01)]:
    T = T_limit(1.0, lam, rt, G0)
    vals = [tau_free(d ** -0.5, G0, lam * d * d, rt * d) / d for d in (32, 128, 512, 2048, 8192)]
    check(f"lam={lam:.4g} rt={rt:.4g} G0={G0}", abs(vals[-1] / T - 1) < 0.01,
          f"tau/d at d=32..8192: " + " ".join(f"{x:.4f}" for x in vals) + f"  limit T={T:.4f}")
print("\n== Proposition B(iii): corner exponents kappa = 2 - dlog T/dlog mu0 of the limit system (k=2)")
def kappa(mu0, *a, **kw):
    h = 1e-3; return 2 - (np.log(T_limit(mu0 * (1 + h), *a, **kw)) - np.log(T_limit(mu0 * (1 - h), *a, **kw))) / (np.log(1 + h) - np.log(1 - h))
for k in (2, 3):
    kl = kappa(1.0, 1e-4, 1.0, 0.01, k=k)
    k2 = kappa(1e-3, np.inf, 1.0, 0.3, k=k)
    k3 = kappa(1e-3, np.inf, 1.0, 1e-9, k=k)
    check(f"k={k}: lam->0 gives 4k", abs(kl - 4 * k) < 0.02, f"kappa={kl:.4f}")
    check(f"k={k}: lam=inf, G0 >> sqrt(rt) mu0 gives 2k", abs(k2 - 2 * k) < 0.02, f"kappa={k2:.4f}")
    check(f"k={k}: lam=inf, G0 << sqrt(rt) mu0 << 1 gives 2k+1", abs(k3 - 2 * k - 1) < 0.02, f"kappa={k3:.4f}")
# explicit constants of the corners
mu0, lam = 1.0, 1e-5
check("lam->0 constant: T*4k(4k-2)lam*mu0^(4k-2) -> 1 (k=2)", abs(T_limit(mu0, lam, 1.0, 0.01) * 48 * lam - 1) < 1e-2, f"{T_limit(mu0, lam, 1.0, 0.01)*48*lam:.4f}")
mu0 = 1e-3; I = quad(lambda x: 1 / (math.sqrt(x * x - 1) * x ** 3), 1, np.inf)[0]
Tc = I / (4 * 2 * math.sqrt(1.0 / 4)) * mu0 ** -3
check("2k+1 corner constant (k=2, rt=1)", abs(T_limit(mu0, np.inf, 1.0, 1e-12) / Tc - 1) < 1e-2, f"T/T_pred={T_limit(mu0, np.inf, 1.0, 1e-12)/Tc:.4f}")
check("Wallis: int_1^inf dx/(x^3 sqrt(x^2-1)) = pi/4", abs(I - math.pi / 4) < 1e-10)
mu0, G0 = 1e-3, 0.3
Tb = mu0 ** -2 / (2 * 4 * G0 * (2 - G0))
check("2k corner constant (k=2): mu0^(2k-2) T -> 1/((2k-2) 2k G0 (2-G0))", abs(T_limit(mu0, np.inf, 1.0, G0) / Tb - 1) < 1e-2, f"T/T_pred={T_limit(mu0, np.inf, 1.0, G0)/Tb:.4f}")
print(f"\n{sum(ok_all)}/{len(ok_all)} checks passed")
