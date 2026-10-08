"""Population gradient flow, Model A, k*=2, with the readout TIED to the feature norm (Gamma = rho = |w|^2),
i.e. the 2-homogeneous parameterisation of Ren et al.  Direction m and norm rho:
  dm/dt   = (1-m^2) [ 4 g g' - rho (2(1-1/N) g g' + V'/N) ]        (attraction independent of rho!)
  drho/dt = -4 rho dL/drho = -4 rho [ -2 g^2 + 2 rho ((1-1/N) g^2 + V/N) ]
Compare with the free-readout ODE (ode_two_timescale.py)."""
import numpy as np
def run(d, N, rho0, mt=0.5):
    m, rho, t = d**-0.5, rho0, 0.0
    while m < mt and t < 1e12:
        g, gp = m**2, 2*m; V, Vp = 1+8*m**2+6*m**4, 16*m+24*m**3
        dm = (1-m**2)*(4*g*gp - rho*(2*(1-1/N)*g*gp + Vp/N))
        drho = -4*rho*(-2*g**2 + 2*rho*((1-1/N)*g**2 + V/N))
        if dm <= 0: return np.inf, rho
        h = 0.01*m/dm
        if drho != 0: h = min(h, 0.01*rho/abs(drho))
        m += h*dm; rho += h*drho; t += h
    return t, rho
ds = np.array([16,32,64,128,256,512,1024])
for N in [128, 1024, 1e18]:
    for rho0 in [0.01, 1.0]:
        res = [run(d, N, rho0) for d in ds]; T = np.array([r[0] for r in res]); R=[r[1] for r in res]
        ok = np.isfinite(T); sl = np.polyfit(np.log(ds[ok]), np.log(T[ok]), 1)[0] if ok.sum()>1 else np.nan
        print(f"N={N:<6g} rho0={rho0:<5} slope={sl:5.2f} T=" + " ".join(f"{x:9.3g}" for x in T) + "  rho_end=" + " ".join(f"{x:7.3g}" for x in R))
