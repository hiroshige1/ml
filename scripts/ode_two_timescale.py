"""Population gradient flow for the single-skill, single-neuron in-context model (Model A), k*=2,
using the closed-form loss  L(m,G) = 1 - 2 G g^2 + G^2 ((1-1/N) g^2 + V/N),  g = m^2, V = 1+8m^2+6m^4.
Spherical flow for m: dm/dt = -(1-m^2) dL/dm ;  readout: dG/dt = -r * dL/dG  (r = relative readout speed).
Question: how does the time to reach m=0.5 scale with d (m0 = d^-1/2) for fixed G vs trainable G?"""
import numpy as np, sys

def rhs(m, G, N, r, trainG):
    g, gp = m**2, 2*m
    V, Vp = 1 + 8*m**2 + 6*m**4, 16*m + 24*m**3
    dLdm = -4*G*g*gp + G**2*((1-1/N)*2*g*gp + Vp/N)
    dLdG = -2*g**2 + 2*G*((1-1/N)*g**2 + V/N)
    dm = -(1-m**2)*dLdm
    dG = -r*dLdG if trainG else 0.0
    return dm, dG

def time_to(m_target, d, N, G0, r, trainG, dt=None, tmax=1e12):
    m, G, t = d**-0.5, G0, 0.0
    # adaptive dt: step so that relative change in m is <= 1%
    while m < m_target and t < tmax:
        dm, dG = rhs(m, G, N, r, trainG)
        if dm <= 0:
            return np.inf, G  # stuck (repulsive origin)
        h = 0.01 * m / dm
        if trainG and dG != 0:
            h = min(h, 0.01 * max(G, 1e-6) / abs(dG))
        m += h*dm; G += h*dG; t += h
    return t, G

def slope(ds, Ts):
    ok = np.isfinite(Ts)
    return np.polyfit(np.log(ds[ok]), np.log(Ts[ok]), 1)[0] if ok.sum() > 1 else np.nan

ds = np.array([16, 32, 64, 128, 256, 512, 1024])
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
print(f"N={N}")
for label, G0, r, trainG in [("fixed G=0.1", 0.1, 0, False), ("fixed G=1.0", 1.0, 0, False),
                             ("train G, G0=0.01, r=1", 0.01, 1.0, True), ("train G, G0=0.01, r=10", 0.01, 10.0, True),
                             ("train G, G0=0.01, r=0.1", 0.01, 0.1, True), ("train G, G0=1e-4, r=1", 1e-4, 1.0, True)]:
    Ts, Gs = [], []
    for d in ds:
        T, G = time_to(0.5, d, N, G0, r, trainG); Ts.append(T); Gs.append(G)
    Ts = np.array(Ts)
    print(f"{label:28s} slope={slope(ds, Ts):5.2f}  T=" + " ".join(f"{T:9.3g}" for T in Ts) + "  G_end=" + " ".join(f"{G:6.3f}" for G in Gs))
# N -> infinity reference (no repulsion): expect slope (kappa-2)/2 = 1 for kappa=4
print("N=inf reference")
for label, G0, r, trainG in [("fixed G=1", 1.0, 0, False), ("train G, G0=0.01, r=1", 0.01, 1, True)]:
    Ts = np.array([time_to(0.5, d, 1e18, G0, r, trainG)[0] for d in ds])
    print(f"{label:28s} slope={slope(ds, Ts):5.2f}  T=" + " ".join(f"{T:9.3g}" for T in Ts))
