"""Population-ODE predictions for experiment 10 (backlog N2): the three regimes at d=128 (sigma_2), fixed BEFORE any SGD run.
Pinned: closed-form escape time (Theorem A) with the analytic trap test m0 <= m*; tied: LSODA with a trap event (drift turns negative);
free: tau_free from verify_theorems.py. eta = 1/d^2. Run from /home/user/ml: python3 scripts/ode_exp10.py > results/exp10/ode_predictions.txt"""
import ast, math, numpy as np
from scipy.integrate import solve_ivp
tree=ast.parse(open('/home/user/ml/scripts/verify_theorems.py').read())
ns={'np':np,'math':math,'solve_ivp':solve_ivp}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Import,ast.ImportFrom))],type_ignores=[]),'vt','exec'),ns)
tau_pf, tied_rhs, tau_free, dLdm, dLdG = ns['tau_pf'], ns['tied_rhs'], ns['tau_free'], ns['dLdm'], ns['dLdG']
import csv, os
rows=[]
d=128; m0=d**-0.5; eta=1/d**2
print(f"d={d} m0={m0:.4f} eta={eta:.3e}")
for g in (1.0, 0.1):
    for N in (16,64,256):
        ms2=4*g/(2*N-(N+5)*g)
        if ms2<=0 or m0*m0<=ms2: print(f"pinned g={g} N={N}: TRAPPED (m*={math.sqrt(ms2) if ms2>0 else float('nan'):.4f} >= m0)"); continue
        t=tau_pf(m0,g,N); print(f"pinned g={g} N={N}: tau={t:.4g} steps={t/eta:.3g}  m*={math.sqrt(ms2):.4f}")
def run_tied(m0,rho0,N):
    f=tied_rhs(2,N)
    ev=lambda t,y: y[0]-math.log(0.5); ev.terminal=True
    ev2=lambda t,y: f(t,y)[0]; ev2.terminal=True; ev2.direction=-1   # drift turns negative -> trap
    s=solve_ivp(f,(0,1e7),[math.log(m0),math.log(rho0)],method='LSODA',events=[ev,ev2],rtol=1e-10,atol=1e-12)
    return (s.t_events[0][0], math.exp(s.y_events[0][0][1])) if s.t_events[0].size else (float('inf'), float('nan'))
for N in (16,64,256):
    t,r=run_tied(m0,0.01,N); print(f"tied rho0=0.01 N={N}: tau={t:.4g} steps={t/eta:.3g} rho_T={r:.4g}")
for N in (16,64,256):
    t=tau_free(m0,0.01,N,1.0,k=2); print(f"free G0=0.01 r=1 N={N}: tau={t:.4g} steps={t/eta:.3g}")
