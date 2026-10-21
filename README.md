# FSOT on 2012 Mac mini hardware

Measured FSOT scalar throughput and a `plant_host_v1` reading on this machine: Apple Macmini6,1 (Late 2012), Intel Core i5-3210M, Intel HD Graphics 4000, Ubuntu 26.04.1 LTS.

The numbers are in [FINDINGS.md](FINDINGS.md). Raw records:

- `results/study.json` — CPU, iGPU, Python, and the official `FSOT-2.1-Cpp` bench
- `results/validation.json` — repeated HD 4000 draws, to show the rate scales with work
- `results/plant_bridge.json` — pin D1D38A plant fold on a measured idle anchor

Throughput uses the unclamped archive scalar, pin AEB2AD. The plant fold uses pin D1D38A and clamps `S` to `[-3, 3]`.

## Reproduce

```bash
python3 scripts/run_study.py
```

That builds three CPU binaries and the iGPU bench, then rewrites `results/study.json`. It expects `g++`, Mesa's `libGL.so.1` and `libEGL.so.1`, and `/dev/dri` open for the desktop user. The iGPU path forces the crocus driver and refuses a software rasterizer. `plant_bridge.json` is a separate live sample, not a product of that script.

The official engine rows are included when `/tmp/fsot-cpp-build/bench_cpp` is already built from `FSOT-2.1-Cpp` (`-O3 -march=native`).

Apache-2.0, same as the other FSOT repositories.
