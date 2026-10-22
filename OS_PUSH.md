# Another pass on the same Mac mini

Measured on 21 Oct 2026, after the tables in [FINDINGS.md](FINDINGS.md). Same machine, same `schedutil` governor, same mitigations, same Ubuntu 26.04.1. The OS was not replaced. The newest pass is first. The sections under it are the earlier passes, with their recorded numbers left as they were.

## Gemma 4, four threads, and the epoch-7 adapter

Measured on 22 Oct 2026. Same Mac mini, same `schedutil` governor, same mitigations, same Q4_K_M file as the section below. llama.cpp 0.7.0, build `b1-620d222`, CPU only. Across the sweep the cores stayed between 2893 and 2897 MHz. Across the two proposals they stayed at 2893 MHz. `schedutil` was the governor at the start and at the end.

llama-bench loaded the model once. A 256-token prompt and a 32-token generation were timed apart, two repetitions each, batch 512.

| Threads | ubatch | Prompt t/s | Generation t/s |
|--:|--:|--:|--:|
| 2 | 128 | 7.20 | 3.62 |
| 4 | 128 | 7.78 | 3.59 |
| 2 | 512 | 7.65 | 3.60 |
| 4 | 512 | 7.92 | 3.61 |

Four threads and ubatch 512 are the fast prompt row, 7.92 t/s beside 7.20 t/s at two threads and ubatch 128. Generation stays inside 3.59 to 3.62 t/s. The standard deviation on those generation means is 0.004 to 0.023 t/s. The eight tests shared one load, so the weights were already resident.

The proposal check is llama-cli at four threads, batch 512, ubatch 512, context 1024, temperature 0, thinking off, up to 128 new tokens. Temperature 0 is the dial's greedy setting. The prompt is the fine-structure row, measured 0.0072973525643. That training prompt prints `PUBLISHED_UNCERTAINTY_PCT: 1.5e-08` and `BAR_SOURCE: sigma_equivalent`.

Base weights: `PROPOSAL: $\frac{e^2}{4\pi\epsilon_0 \hbar c}$`. Prompt 6.5 t/s, generation 3.0 t/s. The hub gate rejected `$`, `\`, `{`, and `}`. The engine computed no value.

The epoch-7 adapter is a separate GGUF, 43,604,480 bytes, sha256 `af9aabf659472b274504d5aecda0a996e346d916f00bb98ee12f4b49de129bab`. It is the math-si `adapter-epoch7` LoRA, rank 16, alpha 32, on attention q, k, v, and output. The file is not in this repo. The scored run passed `--lora` and exited 0. A following load of that same file logged 336 tensors, alpha 32, and a 41.56 MiB CPU buffer.

With the adapter the greedy line was `PROPOSAL: 1/(e^2*pi^2/3)`. Prompt 6.7 t/s, generation 2.8 t/s. The gate accepted the line. The pinned engine computed 0.0411369932583139. The percent error against 0.0072973525643 is 463.7. `kill_metric` treats the printed 1.5e-08 as a published bar, and z is 3.09e10, so the row is killed. The stored training completion is `1/(e^3*phi^4 - psi_con - (POOF*SUCTION)^2*(C_factor^2/P_base))`. Nothing was sent and nothing was promoted.

llama-cli generation after this prompt is 3.0 t/s on the base weights. The section below recorded 3.0 and 3.1 t/s on its own prompts. The bench generation column above is the empty-cache measurement.

## One polynomial, and two skips the formula already had

`sin` and `cos` now reduce by `π`. The remainder stays in `[−π/2, π/2]`, and each call evaluates one series: `sin` through `1/21!`, `cos` through `1/22!`. The other value is a sign, `(−1)^n`. The `π/2` split evaluated both polynomials, because those quadrants swap them.

Mode `push` is that series plus two skips the folded scalar already had. The growth argument on this pin is about `2.88e-4`, so when every lane sits inside `[−0.02, 0.02]` the `exp` stops at `x^8/8!`. A wider lane uses the full `exp`. The observed-state cosine runs only when some lane has `obs > 0.5`. The timed batch sets `obs` to 0, so that factor is 1.

`sqrt` is still the AVX instruction. Pin AEB2AD is unchanged. Mode `ours` keeps the full growth `exp` and always evaluates the observed-state cosine, so the `π` reduction can be timed on its own.

Same 2^20 batch, median of 5 rounds, g++ `-O3 -march=native`. The published scalar and a rerun of the previous series binary were in this same sitting.

| Mode | Threads | Per second | Max abs error vs the published scalar |
|--|--:|--:|--:|
| Published scalar | 4 | 16,500,538 | 0 |
| Previous FSOT series, same sitting | 4 | 34,327,666 | 3.766e-13 |
| `push` | 4 | 44,719,517 | 2.629e-13 |

A published-scalar run just after that `push` landed at 16,461,131/s. The pair above is the comparison: 44,719,517 / 16,500,538 = 2.71×. Against the previous series binary between those two scalar runs, 44,719,517 / 34,327,666 = 1.30×.

In the sitting just before, `push` landed at 47,231,436/s and then 44,418,708/s. One thread did 22,114,390/s and two threads did 39,975,976/s. Mode `ours`, the `π` reduction without the two skips, did 38,064,477/s beside the previous series at 34,260,644/s.

The batch sum moved from `3362353.52762220` to `3362353.52762222`. Endpoints still print `0.07168321` and `-93.36003123`.

On `[−64, 64]` the new `sin` and `cos` stay within `2.61e-15` of libm. The wider grid, the same 449,064 samples as the section below, has a largest absolute gap of `9.895e-10`, at `D = 0.001`, `delta_psi = 7.8`, `delta_theta = −8`, `rho = 3`, `obs = 0`, `hits = 0.3`. The published scalar there is `−349,943.07`, so that gap is a relative `2.8e-15`. Mode `push` matches mode `ours` on that grid.

Use mode `push` for throughput. Keep the published scalar when the check has to be bit-exact.

## Gemma 4 E4B on this CPU

math-si has Gemma 4 propose and the pinned engine compute. The Windows dial loads an epoch-7 LoRA on top of `google/gemma-4-E4B-it`. A GGUF cannot carry that adapter, so this run is the base model. The file is `gemma-4-E4B-it-Q4_K_M.gguf` from `unsloth/gemma-4-E4B-it-GGUF`, 4,977,171,584 bytes, sha256 `85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87`. The weights are not in this repo.

llama.cpp 0.7.0, build `b1-620d222`, compiled here with `-march=native`. That is AVX, not AVX2, and not FMA. The binary has no GPU backend. Four threads. `schedutil` was left alone. The iGPU was not used.

A one-line check, context 512, 32 new tokens, temperature 0, thinking off, answered `hello from this Mac mini.` The client printed prompt 4.9 t/s and generation 3.1 t/s.

The math-si smoke prompt for Air_20C, context 1024, temperature 0.4, thinking off, up to 128 new tokens, came back as one line: `PROPOSAL: $\text{Air\_20C}^1$`. Prompt 6.7 t/s, generation 3.0 t/s. The local hub gate rejected it. `$`, `\`, `{`, and `}` are not in the allowed set, so the pinned engine did not compute a value. That smoke row also has no published uncertainty, so the try is not an inside-bar pass. Nothing was sent and nothing was promoted.

## FSOT's own exp, log, sin, and cos

The first AVX pass called glibc's `libmvec`. Those entry points are gone from the hot path. `bench/fsot_avx_math.hpp` evaluates the series itself: Taylor `exp` after a split by `ln2`, `log` as `2·atanh` of the reduced mantissa, and `sin`/`cos` after a split by `π/2`. The only floating-point instruction taken from the chip, rather than from a series, is AVX `sqrt`. The scalar is still pin AEB2AD, `S = K·(T1+T2+T3)`.

Same 2^20 batch, median of 5 rounds, g++ `-O3 -march=native`. The published-scalar row was run in this same sitting.

| Mode | Threads | Per second | Max abs error vs the published scalar |
|--|--:|--:|--:|
| Published scalar | 4 | 17,051,668 | 0 |
| FSOT series, AVX | 1 | 17,139,900 | 3.766e-13 |
| FSOT series, AVX | 2 | 31,435,686 | 3.766e-13 |
| FSOT series, AVX | 4 | 35,266,749 | 3.766e-13 |
| glibc libmvec, AVX, same sitting | 4 | 31,634,851 | 3.695e-13 |

Two later 4-thread runs of the FSOT series landed at 36,564,523/s and 35,815,091/s. One later published-scalar run landed at 15,995,205/s. The side-by-side pair above is the comparison: 35,266,749 / 17,051,668 = 2.07×. The FSOT series also beat glibc's vector path on the run next to it, 35,266,749 against 31,634,851.

The batch sum moved from `3362353.52762220` to `3362353.52762223`. Endpoints still print `0.07168321` and `-93.36003123`.

A wider grid, 449,064 samples, varies `D` from 0.001 to 400, `rho` from −2 to 3, `delta_psi` from −8 to 8, `delta_theta` from −8 to 8, and both observed states. The largest absolute gap was `1.281e-9`, at `D = 0.001`, `delta_psi = 7.8`, `delta_theta = −8`, `rho = 3`. The published scalar there is about −349,993, so that gap is a relative `3.7e-15`.

Use mode `ours`. Keep the published scalar when the check has to be bit-exact. The libmvec numbers below are the earlier pass, kept so the two paths can be compared.

## Earlier pass, glibc libmvec

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
