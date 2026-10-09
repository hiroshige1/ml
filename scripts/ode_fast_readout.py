"""Free-readout flow at d=64 as a function of the readout learning-rate ratio r = eta_Gamma/eta (Gamma_0 = 0.01), and the Theorem-B
estimate of the SGD cost of the adiabatic 1/N law at d=256.  Produces the Limitations numbers of the paper (the flow caps tau(16)/tau(256) at
12.8 as r -> infinity at d=64; ~1e9 steps at d=256) and the "E2 at N=16 is adiabatic to 1%" statement (tau(N=16): 444.2 at r=1 vs 440.6 at r=256).
Run from /home/user/ml:  python3 scripts/ode_fast_readout.py > results/ode_fast_readout.txt"""
import ast, math, numpy as np
from scipy.integrate import solve_ivp
tree=ast.parse(open('/home/user/ml/scripts/verify_theorems.py').read()); ns={'np':np,'math':math,'solve_ivp':solve_ivp}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Import,ast.ImportFrom))],type_ignores=[]),'vt','exec'),ns)
tau_free=ns['tau_free']; d=64; m0=d**-0.5; eta=1/d**2
print("d=64 m0=0.125 eta=2.44e-4; Theorem B adiabatic: tau ~ mu0^-6 d^3/(48 N) * exp(6*8*G0^2/(2r)):", [round(d**3/(48*N),1) for N in (16,64,256)])
for r in (1,8,32,64,128,256):
    ts=[tau_free(m0,0.01,N,float(r),k=2) for N in (16,64,256)]
    print(f"r={r:4d}: tau={ts[0]:.1f}/{ts[1]:.1f}/{ts[2]:.1f}  steps={ts[0]/eta:.3g}/{ts[1]/eta:.3g}/{ts[2]/eta:.3g}  ratio16/256={ts[0]/ts[2]:.2f}  ratio16/64={ts[0]/ts[1]:.2f}")
for B in (1024,4096):
    for N in (16,64,256):
        print(f"SGD-transfer bound eta_G << B N^2 d^-4: B={B} N={N}: {B*N*N/d**4:.4f}  (eta*r for r=32: {eta*32:.4f}, r=64: {eta*64:.4f})")

for dd in (128, 256):
    print(f"Theorem-B estimate at d={dd}, Gamma_0->0: tau ~ d^3/(48 N) flow units; steps at eta=1/d^2 = d^5/(48N):", {N: f"{dd**5/(48*N):.2g}" for N in (16, 64, 256)})
