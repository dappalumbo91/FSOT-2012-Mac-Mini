# What this Mac mini can do with FSOT

Measured on 21 Oct 2026. Nothing here is copied from another machine's benchmark table.

The fastest strict path is the C++ FP64 scalar on 4 threads, built with `-O3 -march=native`: **17.64 million evaluations per second** in the head-to-head sweep. Two later runs of that same binary landed at 16.37 and 15.48 million/s while the core sat near 2.89 GHz. Hyper-threading is worth using. `-march=native` by itself is not: it beat `-O2` by 1.7% at 4 threads, because this chip has AVX and SSE4.2 and does not have AVX2 or FMA, and the loop is libm.

`-ffast-math` reached **20.16 million/s** in the same sweep, 14.3% above the strict 4-thread binary. The batch sum moved from `3362353.5276222` to `3362353.5276225`. The first and last samples matched at 8 decimal places (`0.07168321` and `-93.36003123`). That is safe for this input batch. The authority path stays the strict binary.

The HD 4000 is the throughput path for a large FP32 batch when a GL context stays open. A sustained 1024×1024 fragment-shader run did **69.49 million evaluations per second** across 50 draws (68.29 million/s across 20). That is FP32, not FP64. Against the shader's own C FP32 formula the max absolute error was `1.818e-5`, and every checked collapse code matched. Draw time grew with the repeat count (22.9 ms, 40.1 ms, 66.9 ms, 127.7 ms for 1, 2, 4, and 8 draws), so the repeats are real work. The chip's render clocks hit the 1100 MHz boost during those draws.

Reading the image back costs about 25 ms per 1024×1024 frame. One draw from the 50-repeat run is 15.09 ms, so a draw-plus-readback round trip is about **25.2 million FP32 evaluations per second**, still above the strict CPU rate. A fresh process is the wrong way to get there: a 32×32 job took 67.6 ms of wall time for 4.7 ms of drawing. At 1024×1024 with 8 draws, wall time included, the process still delivered 28.5 million/s.

Python 3.14 on the same formula did **272 thousand/s**, 65 times slower than the strict 4-thread C++ path. Keep it off the hot loop.

Leave the governor on `schedutil`, leave Spectre/Meltdown mitigations on, and do not chase RAM or the SSD. This run never left the Ivy Bridge libm and the iGPU ALUs. Package temperature stayed in the 50s and low 60s °C.

## Machine

| | |
|--|--|
| Product | Macmini6,1, board Mac-031AEE4D24BFF0B1, Apple Inc. |
| CPU | Intel Core i5-3210M, 2 cores / 4 threads, 1.2–3.1 GHz, L3 3 MiB |
| ISA | AVX, SSE4.2, AES, F16C. No AVX2. No FMA |
| Governor | `schedutil`, unchanged |
| GPU | Intel HD Graphics 4000, PCI 8086:0166, Mesa 26.0.8, crocus, GL 4.2 Core |
| GPU clocks | 350 MHz floor, 1100 MHz max and boost |
| Memory | 15,751,880 KiB visible (15.02 GiB), swap 4,194,300 KiB |
| Disk | SanDisk SDSSDH3 1T02, 953.9 G, SATA, rotational 0 |
| OS | Ubuntu 26.04.1 LTS, Linux 7.0.0-38-generic |

Mitigations that were on during the run: Meltdown PTI, Spectre v2 retpolines, MDS buffer clears, L1TF PTE inversion. `lscpu` reports SRBDS as Vulnerable. Microcode was not changed.

Under the 4-thread loop, samples of `cpu0` `scaling_cur_freq` sat near 2893 MHz, with one sample at 3078 MHz. The `cpu0_mhz` field inside `study.json` is taken after the timed section and sometimes already shows the 1200 MHz floor. It is not the clock the loop ran at.

## CPU sweep

FP64, pin AEB2AD, `S = K·(T1+T2+T3)`, unclamped. Threads stay up. Each cell is the median of 5 rounds of 2^20 samples. Rates are evaluations per second.

| Build | 1 thread | 2 threads | 4 threads |
|--|--:|--:|--:|
| `-O2` | 7,711,518 | 14,408,602 | 17,335,332 |
| `-O3 -march=native` | 7,747,147 | 14,503,152 | 17,636,675 |
| `-O3 -march=native -ffast-math` | 8,764,790 | 16,225,851 | 20,157,041 |

Two threads are 1.87× one thread on the strict native build. Four threads are 1.22× two threads. The second hardware thread on each core is a real gain and a long way from another core.

Strict sums match across `-O2` and `-O3` at every thread count. The fast-math sum differs by `3e-7`.

## HD 4000 sweep

FP32 fragment shader, RGBA32F, crocus, renderer `Mesa Intel(R) HD Graphics 4000 (IVB GT2)`. The rate is draw plus `glFinish`. Readback is not in the rate. Collapse checks compare the shader with the C FP32 formula. Code mismatches: 0 at every size.

| Side | Samples | Repeats | Draw ms | Readback ms | Per second | Max abs error |
|--|--:|--:|--:|--:|--:|--:|
| 32 | 1,024 | 80 | 4.675 | 0.053 | 17,522,470 | 1.705e-5 |
| 64 | 4,096 | 40 | 8.254 | 0.102 | 19,850,689 | 1.794e-5 |
| 128 | 16,384 | 20 | 14.644 | 0.487 | 22,377,048 | 1.812e-5 |
| 256 | 65,536 | 12 | 19.260 | 1.626 | 40,831,402 | 1.818e-5 |
| 512 | 262,144 | 8 | 43.590 | 5.738 | 48,110,480 | 1.818e-5 |
| 768 | 589,824 | 4 | 44.205 | 12.527 | 53,371,142 | 1.818e-5 |
| 1024 | 1,048,576 | 3 | 49.578 | 23.171 | 63,450,694 | 1.818e-5 |

There is no OpenGL compute stage on this GPU, and current Intel compute stacks do not drive Gen7. The fragment shader is the graphics path the chip actually has. Below a 256-pixel side the draw rate is in the same band as the 4-thread CPU, and a new process wipes that out. Use the GPU for a resident batch, 256² and up.

## Official C++ engine

`FSOT-2.1-Cpp` `bench_cpp`, this machine, `-O3 -march=native`. These are single-call hot loops, not the 2^20 batch above. The double call also takes the `observed = true` branch; the batch sets `observed` to 0.

| Type | One scalar | Evaluations/s | Notes |
|--|--:|--:|--|
| double | 124 ns | 8,072,739 | init 5.9 µs, 35 domains 6.1 µs |
| long double | 831 ns | 1,202,956 | |
| `__float128` | 12.6 µs | 79,174 | init 0.896 s |
| `cpp_bin_float<169>` | 140 µs | 7,152 | 1,129× the double call |
| trit, 4096×256 | ref 5.55 ms, packed 66.6 µs | | packed is 83.4× the reference |

Stay on double for the engine. Use the packed trit similarity when that kernel runs.

## Plant fold on this host

Pin D1D38A, fold `plant_host_v1`, the bridge from `fsot-pc-monitor`. `N = 4`, `P = φ`, `D_eff = φ²π = 8.2248`, `S` clamped to `[-3, 3]`, `observed = true`. The idle sample is V0. The GPU channel is `gt_act / gt_max`, because sysfs has no utilization counter. `K = 0.42022`. An excursion is `|S − S_ref| > K`. Collapse code 1 means `|S| ≤ Θ`, and `Θ = 0.91747`.

| State | CPU util | GPU clock | Package °C | S | \|S − S_ref\| | Excursion | Code |
|--|--:|--:|--:|--:|--:|--|--:|
| Idle = V0 | 11.7% | 350 / 1100 | 53 | 0.64540 | 2.9e-5 | no | 1 |
| 4-thread scalar | 91.1% | 350 / 1100 | 53 coretemp | 0.24020 | 0.4052 | no | 1 |
| iGPU scalar | 2.9% | 1100 / 1100 | 60 | 0.20805 | 0.4374 | yes | 1 |

`S_ref = 0.64543`. Both loads pull `S` down and leave it inside the collapse band. The iGPU load crosses the excursion threshold. The CPU load does not, when the bridge is fed coretemp's package reading.

At that same CPU-load instant, `thermal_zone0` (`x86_pkg_temp`) read 65 °C while coretemp's package still read 53 °C. Recomputing the bridge with 65 °C gives `S = 0.11384` and `|S − S_ref| = 0.5316`, which is an excursion. The collapse code is still 1. The two sensors disagreed on a short run, so the CPU-load excursion call depends on which package sensor is used. Both results are in `results/plant_bridge.json`.

`results/study.json` also has `plant_idle` and `plant_after_load`. Those used a 45 °C stand-in, so a healthy idle near 59 °C looked thermally busy. They are not the fold result.

Memory pressure never showed up. At the idle plant sample, 14,023,688 KiB was still available.

## What not to do on this PC

- Do not expect `-march=native` to move this scalar. The 1.7% gap to `-O2` is the measurement.
- Do not spawn `igpu_bench` per batch. Keep the context.
- Do not treat the GPU rate as FP64. The authority numbers are the strict C++ column.
- Do not drop mitigations to chase the last of a libm-bound loop. These rates already include them.
- Do not run the engine at 169-bit precision on the hot path. It is about a thousand times slower than double here.
