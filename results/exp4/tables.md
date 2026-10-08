| setting | k* | d | N | B | seeds | T_e(0.5) median | T_e(0.5) per seed | T_e(0.25) median | T_e(0.25) per seed | T_e(0.1) per seed | reached 0.5 | alignment proxy at last logged step (per seed; step 0 only if run < 1000 steps) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| softmax | 2 | 32 | 16 | 512 | 3 | 1000 | [800, 1000, 1200] | 1200 | [1000, 1400, 1200] | [1200, 1600, 1400] | 3/3 | 0.99, 0.95, 0.94 |
| softmax | 2 | 32 | 64 | 128 | 3 | 1400 | [1400, 3200, 1000] | 1800 | [1800, 3400, 1200] | [2400, 4000, 1600] | 3/3 | 0.99, 0.96, 0.94 |
| softmax | 2 | 32 | 256 | 32 | 3 | 2800 | [2600, 3200, 2800] | 3600 | [4000, 3600, 3400] | [6200, 4600, 5400] | 3/3 | 0.97, 0.99, 0.99 |
| softmax | 2 | 16 | 16 | 512 | 2 | 400 | [400, 400] | 400 | [400, 400] | [600, 600] | 2/2 | 0.83, 0.59 |
| softmax | 2 | 16 | 64 | 128 | 2 | 500 | [600, 400] | 700 | [800, 600] | [1200, 1000] | 2/2 | 0.99, 0.99 |
| softmax | 2 | 16 | 256 | 32 | 2 | 800 | [1000, 600] | 1300 | [1400, 1200] | [3400, 2800] | 2/2 | 0.99, 0.98 |
| softmax | 1 | 32 | 16 | 512 | 3 | 200 | [200, 200, 200] | 200 | [200, 200, 200] | [200, 200, 200] | 3/3 | 0.43, 0.55, 0.45 |
| softmax | 1 | 32 | 64 | 128 | 3 | 200 | [200, 200, 200] | 200 | [200, 200, 200] | [200, 200, 200] | 3/3 | 0.43, 0.55, 0.45 |
| softmax | 1 | 32 | 256 | 32 | 3 | 200 | [200, 200, 200] | 200 | [200, 200, 200] | [200, 200, 200] | 3/3 | 0.43, 0.55, 0.45 |
| linear | 2 | 32 | 16 | 512 | 2 | 2600 | [2200, 3000] | 2600 | [2200, 3000] | [2200, 3200] | 2/2 | 0.95, 0.99 |
| linear | 2 | 32 | 256 | 32 | 2 | inf | [n/r, n/r] | inf | [n/r, n/r] | [n/r, n/r] | 0/2 | 0.44, 0.53 |
| in-weight control softmax | 2 | 32 | 64 | 128 | 3 | 200 | [200, 200, 200] | 200 | [200, 200, 200] | [200, 200, 200] | 3/3 | 0.43, 0.55, 0.45 |

- P8 (d=16): median T_e N=16/64/256 = 400 / 500 / 800; ratio T_e(16)/T_e(256) = 0.50 (P8 requires >= 3); ratio T_e(256)/T_e(16) = 2.00
- P8 (d=32): median T_e N=16/64/256 = 1000 / 1400 / 2800; ratio T_e(16)/T_e(256) = 0.36 (P8 requires >= 3); ratio T_e(256)/T_e(16) = 2.80
- P9a (k*=1, d=32): median T_e N=16/64/256 = 200 / 200 / 200; max/min = 1.00 (P9 requires < 1.5)
- P9b (in-weight control k*=2, d=32, B=128, N=64 prompts of zeroed labels): T_e per seed [200, 200, 200] (median 200); in-context medians N=16: 1000, N=64: 1400, N=256: 2800; faster than every in-context cell (median-vs-median): True; faster than every in-context *run* (max iw seed < min over all in-context seeds): True
- P10 (d=32, N=16): softmax median 1000, linear median 2600 (per-seed linear [2200, 3000])
- P10 (d=32, N=256): softmax median 2800, linear median inf (per-seed linear [n/r, n/r])
- P10 ratio T_e(256)/T_e(16): linear inf, softmax 2.80
- Total: 31 runs, CPU time 15804 s = 4.39 CPU-h (process time summed over 4 threads), wall 1.12 h; diverged runs: 0

## Fine-resolution supplementary runs

| setting | N | B | T_e(0.5) per seed (eval every 20) | median |
|---|---|---|---|---|
| k*=1 softmax | 16 | 512 | [60, 60, 60] | 60 |
| k*=1 softmax | 64 | 128 | [60, 60, 80] | 60 |
| k*=1 softmax | 256 | 32 | [120, 100, 120] | 120 |
| in-weight control k*=2 | 64 | 128 | [40, 40, 60] | 40 |

- P9a at fine resolution: median T_e(k*=1) N=16/64/256 = 60 / 60 / 120; max/min = 2.00

## Zero-predictor MSE on the fixed eval sets

| k | d | N | seed | zero-predictor MSE on eval set |
|---|---|---|---|---|
| 2 | 32 | 16 | 0 | 0.975 |
| 2 | 32 | 16 | 1 | 1.152 |
| 2 | 32 | 16 | 2 | 1.006 |
| 2 | 32 | 256 | 0 | 1.145 |
| 2 | 32 | 256 | 1 | 0.969 |
| 2 | 32 | 256 | 2 | 1.120 |
| 2 | 32 | 64 | 0 | 1.120 |
| 2 | 32 | 64 | 1 | 0.842 |
| 2 | 32 | 64 | 2 | 0.932 |
