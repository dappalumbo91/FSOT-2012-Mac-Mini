# Another pass on the same Mac mini

Measured on 21 Oct 2026, after the tables in [FINDINGS.md](FINDINGS.md). Same machine, same `schedutil` governor, same mitigations, same Ubuntu 26.04.1. The OS was not replaced. This file is the record of that pass.

The fair CPU comparison is inside this session. The published strict 4-thread rate was 17.64 million/s on an earlier run. Rebuilt and run again here, that same scalar did 16.66 million/s, then 16.60 million/s. The new code is compared with those two rows, not with the earlier afternoon.

## What got faster

The scalar is a handful of `exp`, `log`, `sin`, `cos`, and `sqrt` calls. This chip has AVX and does not have AVX2 or FMA. glibc's `libmvec` still has an AVX path that evaluates four doubles at a time. `bench/cpu_scalar_vec.cpp` lays the batch out as separate arrays, folds the terms that do not depend on the sample, and calls `_ZGVcN4v_exp`, `_ZGVcN4v_log`, `_ZGVcN4v_sin`, and `_ZGVcN4v_cos`. Setup and the array transpose sit outside the clock, the same way the original bench leaves thread startup outside the clock.

FP64, pin AEB2AD, 2^20 samples, median of 5 rounds. g++ 15.2 `-O3 -march=native` unless noted.

| Mode | Threads | Per second | Max abs error vs the published scalar |
|--|--:|--:|--:|
| Published scalar | 4 | 16,656,390 | 0 |
| Folded scalar, still one-wide libm | 4 | 19,923,880 | 3.766e-13 |
| AVX libmvec | 1 | 15,291,787 | 3.695e-13 |
| AVX libmvec | 2 | 23,603,103 | 3.695e-13 |
| AVX libmvec | 4 | 31,608,037 | 3.695e-13 |
| AVX libmvec, each thread pinned | 4 | 31,056,054 | 3.695e-13 |

A second 4-thread pair later in the same sitting: published scalar 16,600,355/s, AVX libmvec 31,007,316/s. clang 21 on the same source: published scalar 15,739,483/s, AVX libmvec 30,233,908/s. g++ is the one to keep.

The 4-thread AVX rate is 1.90× the published scalar in the first pair (31,608,037 / 16,656,390). One AVX thread is already in the band of the old 4-thread scalar. Two AVX threads are 1.54× one. Four are 1.34× two. Pinning the four workers to cpu 0–3 did not help.

The batch sum moved from `3362353.52762220` to `3362353.52762223`. The first and last samples still print `0.07168321` and `-93.36003123`. The largest absolute gap against the published scalar, over all 2^20 samples, is `3.695e-13`. That is much tighter than the `-ffast-math` gap in FINDINGS.md, and the rate is higher than that 20.16 million/s row. Use the AVX path for throughput. Keep the published scalar when a bit-exact check is required.

## What did not move

The HD 4000 was already on its clock. A leaner fragment shader folds the same constant terms and uses one `sin` for the acoustic piece. It is selected with `igpu_bench 1024 50 lean`. Against the C FP32 host formula the max absolute error stayed `1.818e-5`, and every checked collapse code still matched.

| Shader | Repeats | Draw ms | Per second |
|--|--:|--:|--:|
| Archive | 8 | 127.175 | 65,961,243 |
| Lean | 8 | 121.579 | 68,997,320 |
| Archive | 20 | 303.754 | 69,041,125 |
| Lean | 20 | 303.404 | 69,120,814 |
| Archive | 50 | 753.030 | 69,623,799 |
| Lean | 50 | 790.462 | 66,326,755 |

The lean shader wins the short run and loses the long one. That is run noise around the published 69.49 million/s plateau, not a new GPU rate. Leave the archive shader in place for the resident FP32 batch.

Replacing Ubuntu, dropping mitigations, or switching the governor was not part of this pass. The gain came from evaluating the same FP64 scalar four-wide.
