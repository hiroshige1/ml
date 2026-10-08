"""Population-ODE prediction of kappa_eff at the exact exp1b parameters (d=32, m0 = c/sqrt(d), c in {0.5,0.7,1.0,1.4}),
for the free-readout model (ode_two_timescale.py equations) and the tied-readout model (ode_tied.py equations).
No free parameters. Written after exp1b results were seen; the equations themselves were committed before (see git log)."""
import numpy as np
def rhs_free(m,G,N,r):
    g,gp=m**2,2*m; V,Vp=1+8*m**2+6*m**4,16*m+24*m**3
    dLdm=-4*G*g*gp+G**2*((1-1/N)*2*g*gp+Vp/N); dLdG=-2*g**2+2*G*((1-1/N)*g**2+V/N)
    return -(1-m**2)*dLdm, -r*dLdG
def rhs_tied(m,rho,N,r=None):
    g,gp=m**2,2*m; V,Vp=1+8*m**2+6*m**4,16*m+24*m**3
    return (1-m**2)*(4*g*gp-rho*(2*(1-1/N)*g*gp+Vp/N)), -4*rho*(-2*g**2+2*rho*((1-1/N)*g**2+V/N))
def T05(rhs,m0,G0,N,r):
    m,G,t=m0,G0,0.0
    while m<0.5 and t<1e9:
        dm,dG=rhs(m,G,N,r)
        if dm<=0: return np.inf
        h=0.01*m/dm
        if dG!=0: h=min(h,0.01*max(abs(G),1e-9)/abs(dG))
        m+=h*dm; G+=h*dG; t+=h
    return t
if __name__=="__main__":
    d=32; cs=np.array([0.5,0.7,1.0,1.4]); m0s=cs/np.sqrt(d)
    for label,rhs,G0,r in [("free, eta_G=eta",rhs_free,0.01,1.0),("free, eta_G=10eta",rhs_free,0.01,10.0),("tied, rho0=0.01",rhs_tied,0.01,None)]:
        row=[]
        for N in [32,128,512,4096]:
            T=np.array([T05(rhs,m0,G0,N,r) for m0 in m0s]); ok=np.isfinite(T)
            b=np.polyfit(np.log(m0s[ok]),np.log(T[ok]),1)[0]
            row.append(f"N={N}: {2-b:4.2f}")
        print(f"{label:20s} "+" | ".join(row))
