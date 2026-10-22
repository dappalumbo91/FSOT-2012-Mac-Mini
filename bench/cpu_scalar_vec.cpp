// Same AEB2AD scalar as cpu_scalar.cpp.
// mode ref   — published scalar, libm, one sample at a time
// mode tight — same math with loop-invariant terms folded
// mode vec   — tight math, AVX x4 through glibc libmvec
// mode ours  — tight math, AVX x4 through FSOT's own exp/log/sin/cos
// mode pin   — vec, each worker pinned to cpu 0..threads-1
// The clock starts after the pool is up and after AoS data is transposed.
// Governor and mitigations are not touched.
#include <algorithm>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include "fsot_avx_math.hpp"
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <vector>

struct In {
    double N, P, D, dp, hits, rho, dth, scale, amp, trend, obs;
};

static double scalar_ref(const In& a) {
    constexpr double PI = 3.14159265358979323846;
    constexpr double PHI = 1.618033988749895;
    constexpr double GAMMA = 0.5772156649015329;
    constexpr double ALPHA = 8.082937414140405e-4;
    constexpr double PSI_CON = 0.6321205588285577;
    constexpr double ETA_EFF = 0.46694220692425986;
    constexpr double BETA = 2.620866911333223e-17;
    constexpr double CHAOS = -0.33102418261048183;
    constexpr double THETA_S = 0.29089654054517305;
    constexpr double POOF = 0.1534822148944508;
    constexpr double C_EFF = 0.9577480213378242;
    constexpr double P_VAR = 0.9579871226722757;
    constexpr double B_IN = 0.7879784892100858;
    constexpr double A_IN = 1.6668538450045732;
    constexpr double A_BLEED = 1.046973630587551;
    constexpr double SUCTION = 0.14703398542810284;
    constexpr double P_NEW = 0.30030227667037146;
    constexpr double C_FACTOR = 0.28761391128429215;
    constexpr double K = 0.4201087636498879;
    double N = a.N < 1e-6 ? 1e-6 : a.N;
    double D = a.D < 1e-6 ? 1e-6 : a.D;
    double growth = std::exp(ALPHA * (1.0 - a.hits / N) * GAMMA / PHI);
    double base = (N * a.P / std::sqrt(D))
        * std::cos((PSI_CON + a.dp) / ETA_EFF)
        * std::exp(-ALPHA * a.hits / N + a.rho + B_IN * a.dp)
        * (1.0 + growth * C_EFF);
    double t1 = base * (1.0 + P_NEW * std::log(D / 25.0));
    if (a.obs > 0.5)
        t1 *= std::exp(C_FACTOR * P_VAR) * std::cos(a.dp + P_VAR);
    double t2 = a.scale * a.amp + a.trend;
    double valve = BETA * std::cos(a.dp) * (N * a.P / std::sqrt(D))
        * (1.0 + CHAOS * (D - 25.0) / 25.0)
        * (1.0 + POOF * std::cos(THETA_S + PI) + SUCTION * std::sin(THETA_S));
    double acoustic = 1.0
        + (A_BLEED * std::sin(a.dth) * std::sin(a.dth)) / PHI
        + (A_IN * std::cos(a.dth) * std::cos(a.dth)) / PHI;
    double phase = 1.0 + B_IN * P_VAR;
    return K * (t1 + t2 + valve * acoustic * phase);
}

// Folded once. sin(dth)^2 replaces a second sin and both cos calls.
// cos(THETA_S+PI) is -cos(THETA_S), evaluated once.
struct Fold {
    double inv_eta, agp, inv_phi, phase, valve_wave, obs_gain;
    double alpha, b_in, c_eff, p_var, p_new, beta, chaos, k, a_in, a_bleed, psi;
};

static Fold make_fold() {
    constexpr double PI = 3.14159265358979323846;
    constexpr double PHI = 1.618033988749895;
    constexpr double GAMMA = 0.5772156649015329;
    constexpr double ALPHA = 8.082937414140405e-4;
    constexpr double PSI_CON = 0.6321205588285577;
    constexpr double ETA_EFF = 0.46694220692425986;
    constexpr double BETA = 2.620866911333223e-17;
    constexpr double CHAOS = -0.33102418261048183;
    constexpr double THETA_S = 0.29089654054517305;
    constexpr double POOF = 0.1534822148944508;
    constexpr double C_EFF = 0.9577480213378242;
    constexpr double P_VAR = 0.9579871226722757;
    constexpr double B_IN = 0.7879784892100858;
    constexpr double A_IN = 1.6668538450045732;
    constexpr double A_BLEED = 1.046973630587551;
    constexpr double SUCTION = 0.14703398542810284;
    constexpr double P_NEW = 0.30030227667037146;
    constexpr double C_FACTOR = 0.28761391128429215;
    constexpr double K = 0.4201087636498879;
    Fold f;
    f.inv_eta = 1.0 / ETA_EFF;
    f.agp = ALPHA * GAMMA / PHI;
    f.inv_phi = 1.0 / PHI;
    f.phase = 1.0 + B_IN * P_VAR;
    f.valve_wave = 1.0 + POOF * std::cos(THETA_S + PI) + SUCTION * std::sin(THETA_S);
    f.obs_gain = std::exp(C_FACTOR * P_VAR);
    f.alpha = ALPHA;
    f.b_in = B_IN;
    f.c_eff = C_EFF;
    f.p_var = P_VAR;
    f.p_new = P_NEW;
    f.beta = BETA;
    f.chaos = CHAOS;
    f.k = K;
    f.a_in = A_IN;
    f.a_bleed = A_BLEED;
    f.psi = PSI_CON;
    return f;
}

static double scalar_tight(const In& a, const Fold& f) {
    double N = a.N < 1e-6 ? 1e-6 : a.N;
    double D = a.D < 1e-6 ? 1e-6 : a.D;
    double h = a.hits / N;
    double growth = std::exp(f.agp * (1.0 - h));
    double inv_sqrt = 1.0 / std::sqrt(D);
    double geom = N * a.P * inv_sqrt;
    double base = geom
        * std::cos((f.psi + a.dp) * f.inv_eta)
        * std::exp(-f.alpha * h + a.rho + f.b_in * a.dp)
        * (1.0 + growth * f.c_eff);
    double t1 = base * (1.0 + f.p_new * std::log(D / 25.0));
    if (a.obs > 0.5)
        t1 *= f.obs_gain * std::cos(a.dp + f.p_var);
    double t2 = a.scale * a.amp + a.trend;
    double s = std::sin(a.dth);
    double acoustic = 1.0 + (f.a_in + (f.a_bleed - f.a_in) * s * s) * f.inv_phi;
    double valve = f.beta * std::cos(a.dp) * geom
        * (1.0 + f.chaos * (D - 25.0) / 25.0)
        * f.valve_wave;
    return f.k * (t1 + t2 + valve * acoustic * f.phase);
}

// Same folded scalar, with FSOT's own exp, log, sin, and cos.
static double scalar_ours(const In& a, const Fold& f) {
    double N = a.N < 1e-6 ? 1e-6 : a.N;
    double D = a.D < 1e-6 ? 1e-6 : a.D;
    double h = a.hits / N;
    double growth = fsot_exp_d(f.agp * (1.0 - h));
    double inv_sqrt = 1.0 / std::sqrt(D);
    double geom = N * a.P * inv_sqrt;
    double base = geom
        * fsot_cos_d((f.psi + a.dp) * f.inv_eta)
        * fsot_exp_d(-f.alpha * h + a.rho + f.b_in * a.dp)
        * (1.0 + growth * f.c_eff);
    double t1 = base * (1.0 + f.p_new * fsot_log_d(D / 25.0));
    if (a.obs > 0.5)
        t1 *= f.obs_gain * fsot_cos_d(a.dp + f.p_var);
    double t2 = a.scale * a.amp + a.trend;
    double s = fsot_sin_d(a.dth);
    double acoustic = 1.0 + (f.a_in + (f.a_bleed - f.a_in) * s * s) * f.inv_phi;
    double valve = f.beta * fsot_cos_d(a.dp) * geom
        * (1.0 + f.chaos * (D - 25.0) / 25.0)
        * f.valve_wave;
    return f.k * (t1 + t2 + valve * acoustic * f.phase);
}

extern "C" {
__m256d _ZGVcN4v_cos(__m256d);
__m256d _ZGVcN4v_exp(__m256d);
__m256d _ZGVcN4v_log(__m256d);
__m256d _ZGVcN4v_sin(__m256d);
}

static void fill(std::vector<In>& v) {
    for (size_t i = 0; i < v.size(); ++i) {
        double luma = 0.15 + 0.85 * double(i % 997) / 996.0;
        v[i] = In{luma * 4.0, luma, 25.0, double(i % 64) / 63.0 * 6.283185307179586,
                  luma * 0.3, 1.0, 1.0, luma, 1.0, 0.0, 0.0};
    }
}

struct Soa {
    std::vector<double> N, P, D, dp, hits, rho, dth, scale, amp, trend, obs;
    explicit Soa(const std::vector<In>& in) {
        size_t n = in.size();
        N.resize(n); P.resize(n); D.resize(n); dp.resize(n); hits.resize(n);
        rho.resize(n); dth.resize(n); scale.resize(n); amp.resize(n);
        trend.resize(n); obs.resize(n);
        for (size_t i = 0; i < n; ++i) {
            N[i] = in[i].N; P[i] = in[i].P; D[i] = in[i].D; dp[i] = in[i].dp;
            hits[i] = in[i].hits; rho[i] = in[i].rho; dth[i] = in[i].dth;
            scale[i] = in[i].scale; amp[i] = in[i].amp; trend[i] = in[i].trend;
            obs[i] = in[i].obs;
        }
    }
};

static void work_ref(const In* in, double* out, size_t b, size_t e) {
    for (size_t i = b; i < e; ++i)
        out[i] = scalar_ref(in[i]);
}

static void work_tight(const In* in, double* out, size_t b, size_t e, const Fold& f) {
    for (size_t i = b; i < e; ++i)
        out[i] = scalar_tight(in[i], f);
}

template <bool Ours>
static inline __m256d vexp(__m256d x) {
    if constexpr (Ours) return fsot_exp(x);
    else return _ZGVcN4v_exp(x);
}
template <bool Ours>
static inline __m256d vlog(__m256d x) {
    if constexpr (Ours) return fsot_log(x);
    else return _ZGVcN4v_log(x);
}
template <bool Ours>
static inline __m256d vsin(__m256d x) {
    if constexpr (Ours) return fsot_sin(x);
    else return _ZGVcN4v_sin(x);
}
template <bool Ours>
static inline __m256d vcos(__m256d x) {
    if constexpr (Ours) return fsot_cos(x);
    else return _ZGVcN4v_cos(x);
}

template <bool Ours>
static void work_vec(const Soa& s, double* out, size_t b, size_t e, const Fold& f) {
    const __m256d vmin = _mm256_set1_pd(1e-6);
    const __m256d vone = _mm256_set1_pd(1.0);
    const __m256d v25 = _mm256_set1_pd(25.0);
    const __m256d vhalf = _mm256_set1_pd(0.5);
    const __m256d vagp = _mm256_set1_pd(f.agp);
    const __m256d vinv_eta = _mm256_set1_pd(f.inv_eta);
    const __m256d vpsi = _mm256_set1_pd(f.psi);
    const __m256d valpha = _mm256_set1_pd(f.alpha);
    const __m256d vb_in = _mm256_set1_pd(f.b_in);
    const __m256d vc_eff = _mm256_set1_pd(f.c_eff);
    const __m256d vp_var = _mm256_set1_pd(f.p_var);
    const __m256d vp_new = _mm256_set1_pd(f.p_new);
    const __m256d vbeta = _mm256_set1_pd(f.beta);
    const __m256d vchaos = _mm256_set1_pd(f.chaos);
    const __m256d vk = _mm256_set1_pd(f.k);
    const __m256d va_in = _mm256_set1_pd(f.a_in);
    const __m256d va_bleed = _mm256_set1_pd(f.a_bleed);
    const __m256d vinv_phi = _mm256_set1_pd(f.inv_phi);
    const __m256d vphase = _mm256_set1_pd(f.phase);
    const __m256d vwave = _mm256_set1_pd(f.valve_wave);
    const __m256d vobs = _mm256_set1_pd(f.obs_gain);
    size_t i = b;
    for (; i + 4 <= e; i += 4) {
        __m256d N = _mm256_max_pd(_mm256_loadu_pd(&s.N[i]), vmin);
        __m256d P = _mm256_loadu_pd(&s.P[i]);
        __m256d D = _mm256_max_pd(_mm256_loadu_pd(&s.D[i]), vmin);
        __m256d dp = _mm256_loadu_pd(&s.dp[i]);
        __m256d hits = _mm256_loadu_pd(&s.hits[i]);
        __m256d rho = _mm256_loadu_pd(&s.rho[i]);
        __m256d dth = _mm256_loadu_pd(&s.dth[i]);
        __m256d scale = _mm256_loadu_pd(&s.scale[i]);
        __m256d amp = _mm256_loadu_pd(&s.amp[i]);
        __m256d trend = _mm256_loadu_pd(&s.trend[i]);
        __m256d obs = _mm256_loadu_pd(&s.obs[i]);
        __m256d h = _mm256_div_pd(hits, N);
        __m256d growth = vexp<Ours>(_mm256_mul_pd(vagp, _mm256_sub_pd(vone, h)));
        __m256d inv_sqrt = _mm256_div_pd(vone, _mm256_sqrt_pd(D));
        __m256d geom = _mm256_mul_pd(_mm256_mul_pd(N, P), inv_sqrt);
        __m256d base = _mm256_mul_pd(
            _mm256_mul_pd(geom, vcos<Ours>(_mm256_mul_pd(_mm256_add_pd(vpsi, dp), vinv_eta))),
            _mm256_mul_pd(
                vexp<Ours>(_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), valpha), h), rho), _mm256_mul_pd(vb_in, dp))),
                _mm256_add_pd(vone, _mm256_mul_pd(growth, vc_eff))));
        __m256d t1 = _mm256_mul_pd(base, _mm256_add_pd(vone, _mm256_mul_pd(vp_new, vlog<Ours>(_mm256_div_pd(D, v25)))));
        __m256d quirk = _mm256_mul_pd(vobs, vcos<Ours>(_mm256_add_pd(dp, vp_var)));
        __m256d mask = _mm256_cmp_pd(obs, vhalf, _CMP_GT_OQ);
        t1 = _mm256_mul_pd(t1, _mm256_blendv_pd(vone, quirk, mask));
        __m256d t2 = _mm256_add_pd(_mm256_mul_pd(scale, amp), trend);
        __m256d sn = vsin<Ours>(dth);
        __m256d acoustic = _mm256_add_pd(vone, _mm256_mul_pd(
            _mm256_add_pd(va_in, _mm256_mul_pd(_mm256_sub_pd(va_bleed, va_in), _mm256_mul_pd(sn, sn))),
            vinv_phi));
        __m256d valve = _mm256_mul_pd(
            _mm256_mul_pd(_mm256_mul_pd(vbeta, vcos<Ours>(dp)), geom),
            _mm256_mul_pd(_mm256_add_pd(vone, _mm256_mul_pd(vchaos, _mm256_div_pd(_mm256_sub_pd(D, v25), v25))), vwave));
        __m256d S = _mm256_mul_pd(vk, _mm256_add_pd(_mm256_add_pd(t1, t2), _mm256_mul_pd(_mm256_mul_pd(valve, acoustic), vphase)));
        _mm256_storeu_pd(out + i, S);
    }
    for (; i < e; ++i) {
        In a{s.N[i], s.P[i], s.D[i], s.dp[i], s.hits[i], s.rho[i], s.dth[i],
             s.scale[i], s.amp[i], s.trend[i], s.obs[i]};
        out[i] = Ours ? scalar_ours(a, f) : scalar_tight(a, f);
    }
}

static void pin_self(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "vec";
    int threads = argc > 2 ? std::atoi(argv[2]) : 4;
    size_t n = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : (1u << 20);
    int rounds = argc > 4 ? std::atoi(argv[4]) : 5;
    if (threads < 1) threads = 1;
    if (threads > 16) threads = 16;
    if (n < 1024) n = 1024;
    bool ours = std::strcmp(mode, "ours") == 0 || std::strcmp(mode, "ours-pin") == 0;
    bool vec = ours || std::strcmp(mode, "vec") == 0 || std::strcmp(mode, "pin") == 0;
    bool pin = std::strcmp(mode, "pin") == 0 || std::strcmp(mode, "ours-pin") == 0;
    bool tight = std::strcmp(mode, "tight") == 0 || vec;
    Fold fold = make_fold();
    if (std::strcmp(mode, "probe") == 0) {
        const double Ds[] = {1e-3, 0.1, 1, 10, 25, 80, 400};
        const double rhos[] = {-2, 0, 1, 3};
        const double obs[] = {0, 1};
        const double hits_scale[] = {0, 0.3, 2};
        double max_abs = 0, at_d = 0, at_dp = 0, at_dth = 0, at_rho = 0, at_obs = 0;
        long long count = 0;
        for (double D : Ds) {
            for (double rho : rhos) {
                for (double ob : obs) {
                    for (double hs : hits_scale) {
                        for (int idp = -40; idp <= 40; ++idp) {
                            for (int idt = -16; idt <= 16; ++idt) {
                                double dp = idp * 0.2;
                                double dth = idt * 0.5;
                                In a{2.0, 0.5, D, dp, hs, rho, dth, 0.5, 1.0, 0.0, ob};
                                double err = std::fabs(scalar_ours(a, fold) - scalar_ref(a));
                                if (err > max_abs) {
                                    max_abs = err;
                                    at_d = D;
                                    at_dp = dp;
                                    at_dth = dth;
                                    at_rho = rho;
                                    at_obs = ob;
                                }
                                ++count;
                            }
                        }
                    }
                }
            }
        }
        std::printf("{\"mode\":\"probe\",\"count\":%lld,\"max_abs_err\":%.3e,"
                    "\"at_D\":%.6g,\"at_dp\":%.6g,\"at_dth\":%.6g,\"at_rho\":%.6g,\"at_obs\":%.0f}\n",
                    count, max_abs, at_d, at_dp, at_dth, at_rho, at_obs);
        return 0;
    }
    std::vector<In> in(n);
    std::vector<double> out(n);
    fill(in);
    Soa soa(in);
    std::barrier gate(static_cast<std::ptrdiff_t>(threads + 1));
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            if (pin) pin_self(t % 4);
            size_t chunk = (n + size_t(threads) - 1) / size_t(threads);
            if (vec) chunk = (chunk + 3) & ~size_t(3);
            size_t b = size_t(t) * chunk;
            size_t e = std::min(n, b + chunk);
            if (t == threads - 1) e = n;
            auto run = [&] {
                if (b >= n) return;
                if (ours) work_vec<true>(soa, out.data(), b, e, fold);
                else if (vec) work_vec<false>(soa, out.data(), b, e, fold);
                else if (tight) work_tight(in.data(), out.data(), b, e, fold);
                else work_ref(in.data(), out.data(), b, e);
            };
            run();
            gate.arrive_and_wait();
            for (int r = 0; r < rounds; ++r) {
                gate.arrive_and_wait();
                run();
                gate.arrive_and_wait();
            }
        });
    }
    gate.arrive_and_wait();
    using clk = std::chrono::steady_clock;
    std::vector<double> secs;
    secs.reserve(rounds);
    for (int r = 0; r < rounds; ++r) {
        auto t0 = clk::now();
        gate.arrive_and_wait();
        gate.arrive_and_wait();
        secs.push_back(std::chrono::duration<double>(clk::now() - t0).count());
    }
    for (auto& th : pool) th.join();
    std::sort(secs.begin(), secs.end());
    double med = secs[secs.size() / 2];
    double sum = 0, ref = 0, max_abs = 0;
    for (size_t i = 0; i < n; ++i) {
        double r = scalar_ref(in[i]);
        ref += r;
        sum += out[i];
        double ad = std::fabs(out[i] - r);
        if (ad > max_abs) max_abs = ad;
    }
    int mhz = 0;
    if (FILE* f = std::fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r")) {
        if (std::fscanf(f, "%d", &mhz) == 1) mhz /= 1000;
        std::fclose(f);
    }
    std::printf("{\"mode\": \"%s\",\"threads\":%d,\"n\":%zu,\"median_s\":%.6f,\"scalar_per_s\":%.3f,"
                "\"sum\":%.8f,\"ref_sum\":%.8f,\"max_abs_err\":%.3e,\"s0\":%.8f,\"s_last\":%.8f,\"cpu0_mhz\":%d}\n",
                mode, threads, n, med, double(n) / med, sum, ref, max_abs, out[0], out[n - 1], mhz);
    return 0;
}
