#!/usr/bin/env python3
"""Measure the FSOT scalar on this Late 2012 Mac mini and write results/study.json."""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
BUILD = ROOT / "build"
RES = ROOT / "results"
LIBS = [
    "/usr/lib/x86_64-linux-gnu/libEGL.so.1",
    "/usr/lib/x86_64-linux-gnu/libGL.so.1",
]


def read(path: str, default: str = "") -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def cpu_percent(sample_s: float = 0.2) -> float:
    def snap():
        parts = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        vals = [int(x) for x in parts]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        return sum(vals), idle

    t0, i0 = snap()
    time.sleep(sample_s)
    t1, i1 = snap()
    dt = t1 - t0
    if dt <= 0:
        return 0.0
    return 100.0 * (1.0 - (i1 - i0) / dt)


def mem_fraction() -> float:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        info[k] = int(v.strip().split()[0])
    total = info["MemTotal"]
    avail = info.get("MemAvailable", info["MemFree"])
    return (total - avail) / total


def plant_scalar(cpu_frac: float, mem_frac: float, gpu_frac: float, temp_c: float) -> dict:
    """plant_host_v1 bridge from fsot-pc-monitor BRIDGE.md, pin constants AEB2AD for S."""
    n_ch = 4.0
    f_c = min(1.0, max(0.0, cpu_frac))
    f_m = min(1.0, max(0.0, mem_frac))
    f_g = min(1.0, max(0.0, gpu_frac))
    f_t = min(1.0, max(0.0, abs(temp_c - 45.0) / 25.0))
    load = 0.25 * (f_c + f_m + f_g + f_t)
    # Live route. V0 is the idle anchor measured by the caller and passed as zeros
    # here; this function is the instantaneous map, not a fitted model.
    return {
        "f_cpu": f_c,
        "f_mem": f_m,
        "f_gpu": f_g,
        "f_temp": f_t,
        "load": load,
        "delta_psi": max(-1.0, min(1.0, load)),
        "delta_theta": max(-1.0, min(1.0, (temp_c - 45.0) / 25.0)),
        "recent_hits": max(0.0, min(n_ch, load * n_ch)),
        "rho": f_m,
        "temp_c": temp_c,
    }


def host_s(N, P, D, dp, hits, rho, dth, scale, amp, trend, observed) -> float:
    alpha = 8.082937414140405e-4
    psi_con = 0.6321205588285577
    eta_eff = 0.46694220692425986
    beta = 2.620866911333223e-17
    chaos = -0.33102418261048183
    theta_s = 0.29089654054517305
    poof = 0.1534822148944508
    c_eff = 0.9577480213378242
    p_var = 0.9579871226722757
    b_in = 0.7879784892100858
    a_in = 1.6668538450045732
    a_bleed = 1.046973630587551
    suction = 0.14703398542810284
    p_new = 0.30030227667037146
    c_factor = 0.28761391128429215
    k = 0.4201087636498879
    phi = 1.618033988749895
    gamma = 0.5772156649015329
    pi = math.pi
    N = max(N, 1e-6)
    D = max(D, 1e-6)
    growth = math.exp(alpha * (1.0 - hits / N) * gamma / phi)
    base = (N * P / math.sqrt(D)) * math.cos((psi_con + dp) / eta_eff) * math.exp(-alpha * hits / N + rho + b_in * dp) * (1.0 + growth * c_eff)
    t1 = base * (1.0 + p_new * math.log(D / 25.0))
    if observed > 0.5:
        t1 *= math.exp(c_factor * p_var) * math.cos(dp + p_var)
    t2 = scale * amp + trend
    valve = beta * math.cos(dp) * (N * P / math.sqrt(D)) * (1.0 + chaos * (D - 25.0) / 25.0) * (1.0 + poof * math.cos(theta_s + pi) + suction * math.sin(theta_s))
    acoustic = 1.0 + (a_bleed * math.sin(dth) ** 2) / phi + (a_in * math.cos(dth) ** 2) / phi
    return k * (t1 + t2 + valve * acoustic * phase(b_in, p_var))


def phase(b_in, p_var) -> float:
    return 1.0 + b_in * p_var


def sh(cmd: list[str]) -> str:
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return p.stdout.strip()


def build() -> None:
    BUILD.mkdir(exist_ok=True)
    flags = {
        "o2": ["-O2"],
        "o3_native": ["-O3", "-march=native"],
        "o3_native_fast": ["-O3", "-march=native", "-ffast-math"],
    }
    for name, fl in flags.items():
        out = BUILD / f"cpu_scalar_{name}"
        subprocess.run(
            ["g++", "-std=c++20", *fl, str(BENCH / "cpu_scalar.cpp"), "-o", str(out), "-pthread"],
            check=True,
        )
    subprocess.run(
        [
            "gcc", "-O2", "-std=c11", str(BENCH / "igpu_bench.c"), "-o", str(BUILD / "igpu_bench"),
            *LIBS, "-lm", "-ldl",
        ],
        check=True,
    )


def run_json(cmd: list[str]) -> dict:
    p = subprocess.run(cmd, capture_output=True, text=True)
    text = p.stdout.strip()
    if p.returncode != 0 or not text:
        return {
            "ok": False,
            "code": p.returncode,
            "stdout": text[-2000:],
            "stderr": p.stderr[-2000:],
        }
    line = text.splitlines()[-1]
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return {"ok": False, "stdout": text[-2000:], "stderr": p.stderr[-2000:]}
    return row


def python_rate(n: int = 200_000) -> dict:
    t0 = time.perf_counter()
    acc = 0.0
    for i in range(n):
        luma = 0.15 + 0.85 * (i % 997) / 996.0
        acc += host_s(luma * 4, luma, 25, (i % 64) / 63.0 * math.tau, luma * 0.3, 1, 1, luma, 1, 0, 0)
    dt = time.perf_counter() - t0
    return {"n": n, "seconds": dt, "scalar_per_s": n / dt, "sum": acc}


def main() -> None:
    RES.mkdir(exist_ok=True)
    build()
    temp = int(read("/sys/class/thermal/thermal_zone0/temp", "0")) / 1000.0
    gpu_cur = int(read("/sys/class/drm/card1/gt_cur_freq_mhz", "0"))
    gpu_max = int(read("/sys/class/drm/card1/gt_max_freq_mhz", "1"))
    idle_cpu = cpu_percent(0.4)
    idle_mem = mem_fraction()
    hardware = {
        "product": read("/sys/class/dmi/id/product_name"),
        "board": read("/sys/class/dmi/id/board_name"),
        "vendor": read("/sys/class/dmi/id/sys_vendor"),
        "cpu": "Intel Core i5-3210M",
        "cpu_max_mhz": int(read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "0")) // 1000,
        "cores": 2,
        "threads": 4,
        "l3": "3 MiB",
        "isa": "AVX, SSE4.2, no AVX2, no FMA",
        "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "gpu": "Intel HD Graphics 4000 (IVB GT2)",
        "gpu_mhz": {"min": int(read("/sys/class/drm/card1/gt_min_freq_mhz", "0")), "max": gpu_max},
        "ram_kib": int(read("/proc/meminfo").splitlines()[0].split()[1]),
        "disk": "SanDisk SDSSDH3 1TB SATA SSD",
        "kernel": platform.platform(),
        "os": read("/etc/os-release").split("PRETTY_NAME=")[-1].strip().strip('"') if False else "",
    }
    osrel = Path("/etc/os-release").read_text()
    for line in osrel.splitlines():
        if line.startswith("PRETTY_NAME="):
            hardware["os"] = line.split("=", 1)[1].strip().strip('"')
    cpu_runs = []
    for flags in ("o2", "o3_native", "o3_native_fast"):
        binp = BUILD / f"cpu_scalar_{flags}"
        for threads in (1, 2, 4):
            row = run_json([str(binp), str(threads), str(1 << 20), "5"])
            row["flags"] = flags
            cpu_runs.append(row)
            rate = row.get("scalar_per_s")
            print("cpu", flags, threads, round(rate) if isinstance(rate, (int, float)) else row, flush=True)
    igpu_runs = []
    for side, reps in ((32, 80), (64, 40), (128, 20), (256, 12), (512, 8), (768, 4), (1024, 3)):
        row = run_json([str(BUILD / "igpu_bench"), str(side), str(reps)])
        igpu_runs.append(row)
        rate = row.get("scalar_per_s")
        print("igpu", side, round(rate) if isinstance(rate, (int, float)) else row, flush=True)
    py = python_rate()
    print("python", round(py["scalar_per_s"]), flush=True)
    temp_after = int(read("/sys/class/thermal/thermal_zone0/temp", "0")) / 1000.0
    cpu_max = hardware["cpu_max_mhz"] or 1
    cpu_mhz_peak = max((int(r.get("cpu0_mhz") or 0) for r in cpu_runs), default=0)
    gpu_act_peak = max((int(r.get("gt_act_mhz") or 0) for r in igpu_runs), default=0)
    gpu_cur_peak = max((int(r.get("gt_cur_mhz") or 0) for r in igpu_runs), default=0)
    gpu_frac_idle = gpu_cur / gpu_max if gpu_max else 0.0
    gpu_frac_load = (gpu_act_peak or gpu_cur_peak) / gpu_max if gpu_max else 0.0
    # Clock ratio, not /proc/stat. The bench has already returned by the time
    # a new idle sample would run, so the in-run cpu0 frequency is the load signal.
    study = {
        "pin": "AEB2AD",
        "law": "S = K*(T1+T2+T3)",
        "hardware": hardware,
        "plant_idle": plant_scalar(idle_cpu / 100.0, idle_mem, gpu_frac_idle, temp),
        "plant_after_load": plant_scalar(cpu_mhz_peak / cpu_max, mem_fraction(), gpu_frac_load, temp_after),
        "clocks_under_load": {
            "cpu0_mhz_peak": cpu_mhz_peak,
            "gt_act_mhz_peak": gpu_act_peak,
            "gt_cur_mhz_peak": gpu_cur_peak,
            "temp_c_after": temp_after,
        },
        "cpu": cpu_runs,
        "igpu": igpu_runs,
        "python": py,
        "notes": [
            "CPU numbers are FP64 archive evaluations per second. Threads stay up; the median is 5 rounds of 2^20 samples and does not include thread startup.",
            "iGPU numbers are FP32 fragment-shader evaluations per second, draw plus finish, readback excluded.",
            "The plant GPU channel is a clock ratio (gt_act or gt_cur over gt_max). This iGPU has no utilization counter in sysfs.",
            "plant_after_load f_cpu is peak cpu0 clock over cpuinfo_max, sampled at the end of each CPU round. It is not a /proc/stat utilization.",
            "Spectre/Meltdown mitigations were left on.",
        ],
    }
    official = Path("/tmp/fsot-cpp-build/bench_cpp")
    if official.exists():
        try:
            p = subprocess.run([str(official)], capture_output=True, text=True, timeout=300)
            lines = [json.loads(x) for x in p.stdout.splitlines() if x.startswith("{")]
            study["official_bench_cpp"] = lines
            if p.returncode != 0:
                study["official_bench_cpp_error"] = p.stderr[-2000:]
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
            study["official_bench_cpp_error"] = str(exc)
    (RES / "study.json").write_text(json.dumps(study, indent=2) + "\n")
    print("wrote", RES / "study.json")


if __name__ == "__main__":
    main()
