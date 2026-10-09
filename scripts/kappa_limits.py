"""d -> infinity values of the secant exponent kappa(lambda, r~, Gamma_0) of the joint-scaling limit system (Prop. joint, paper/theorems.tex),
for the exp-1b free cells: d=32, N in {32,128,512,4096} -> lambda = N/d^2, r = eta_Gamma/eta = 1 -> r~ = 1/32, Gamma_0 = 0.01, secant over
mu_0 = m_0 sqrt(d) in [0.5, 1.4] (the pre-registered m_0 window). Reproduces the paper's "(numerical; 6.45/5.40/4.95/4.83)".
Run from /home/user/ml:  python3 scripts/kappa_limits.py > results/kappa_limits.txt"""
import ast, math, numpy as np
from scipy.integrate import solve_ivp, quad
tree = ast.parse(open('scripts/verify_theorems.py').read()); ns = {'np': np, 'math': math, 'solve_ivp': solve_ivp, 'quad': quad}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))], type_ignores=[]), 'vt', 'exec'), ns)
T_limit = ns['T_limit']
d, G0, r = 32, 0.01, 1.0
print("secant kappa = 2 - [ln T(1.4) - ln T(0.5)]/[ln 1.4 - ln 0.5] of the limit system, k=2, Gamma_0=0.01, r~=r/d")
for N in (32, 128, 512, 4096):
    lam, rt = N / d ** 2, r / d
    Ta, Tb = T_limit(0.5, lam, rt, G0), T_limit(1.4, lam, rt, G0)
    kap = 2 - (math.log(Tb) - math.log(Ta)) / (math.log(1.4) - math.log(0.5))
    print(f"N={N:5d} (lambda={lam:.4g}, r~={rt:.4g}): kappa_limit = {kap:.3f}")
