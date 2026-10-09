| cell | N | T_0.5 per seed (steps) | reached | median steps | median flow T*eta | ODE flow | ODE steps | median/ODE | readout at T_0.5 (seeds) | final \|m\| |
|---|---|---|---|---|---|---|---|---|---|---|
| a pinned g=1 | 16 | CENS/CENS | 0/2 | inf | inf | trapped | trapped | - | - | 0.014/0.005 |
| a pinned g=1 | 64 | CENS/CENS | 0/2 | inf | inf | trapped | trapped | - | - | 0.002/0.002 |
| a pinned g=1 | 256 | CENS/CENS | 0/2 | inf | inf | trapped | trapped | - | - | 0.004/0.006 |
| b pinned g=0.1 | 16 | CENS/CENS/CENS | 0/3 | inf | inf | trapped | trapped | - | - | 0.029/0.033/0.022 |
| b pinned g=0.1 | 64 | 1783191/1896305/1722126 | 3/3 | 1783191 | 108.8 | 110.1 | 1.8e+06 | 0.989 | - | 0.500/0.500/0.500 |
| b pinned g=0.1 | 256 | 1466097/1493848/1445353 | 3/3 | 1466097 | 89.5 | 89.0 | 1.46e+06 | 1.006 | - | 0.500/0.500/0.500 |
| c tied rho0=0.01 | 16 | 162187/135030/160529 | 3/3 | 160529 | 9.8 | 8.8 | 1.44e+05 | 1.118 | 0.01063/0.01077/0.01099 | 0.500/0.500/0.500 |
| c tied rho0=0.01 | 64 | 131662/127407/130146 | 3/3 | 130146 | 7.9 | 8.2 | 1.34e+05 | 0.969 | 0.01139/0.01138/0.01142 | 0.500/0.501/0.500 |
| c tied rho0=0.01 | 256 | 137243/134382/128556 | 3/3 | 134382 | 8.2 | 8.1 | 1.32e+05 | 1.017 | 0.01145/0.01145/0.0115 | 0.500/0.500/0.500 |
| d free G0=0.01 etaG=eta | 256 | 5458606/5121944 | 2/2 | 5290275 | 322.9 | 327.9 | 5.37e+06 | 0.985 | 0.2371/0.2375 | 0.500/0.500 |

Censoring (no |m|=0.5 within cap):

| cell | N=16 | N=64 | N=256 | cap |
|---|---|---|---|---|
| a pinned g=1 | 2/2 | 2/2 | 2/2 | 1000000 |
| b pinned g=0.1 | 3/3 | 0/3 | 0/3 | 2500000 |
| c tied rho0=0.01 | 0/3 | 0/3 | 0/3 | 1000000 |
| d free G0=0.01 etaG=eta | - | - | 0/2 | 12000000 |

Tied N-ratio T(16)/T(256) = 1.195 (ODE 1.086, pre-reg 1.09); T(16)/T(64) = 1.233, T(64)/T(256) = 0.968

Free secant kappa = 2 - ln(tau128/tau64)/ln(2^-1/2):

| numerator (d=128) | denominator (d=64) | kappa |
|---|---|---|
| ODE 327.9 | ODE 50.1 (pre-registered baseline) | 7.42 |
| ODE 327.9 | ODE 101.0 (exp 6 free eta_Gamma=eta, N=256, B=64) | 5.40 |
| SGD median 322.9 flow | SGD exp 6 median 100.4 flow | 5.37 |
| SGD median | 50.1 (pre-registered baseline) | 7.38 |

Verdict detail:

- P23a: passed=True default_triggered=False; N=16 censored 2/2; N=64 censored 2/2; N=256 censored 2/2
- P23b: passed=True default_triggered=False; N=16 censored 3/3 (need >=2/3); N=64 median/ODE=0.989 ok; N=256 median/ODE=1.006 ok
- P23c: passed=True default_triggered=False; N=16 median/ODE=1.118 ok; N=64 median/ODE=0.969 ok; N=256 median/ODE=1.017 ok; ratio T(16)/T(256)=1.195 (pre-reg 1.09 +-0.15; ODE 1.086) ok
- P23d: passed=False default_triggered=False; n seeds run 2/2, censored 0; median/ODE=0.985 ok; SGD secant (exp 6 d=64 median 100.4 flow) = 5.37 (need 7.4 +-0.5) OUT
