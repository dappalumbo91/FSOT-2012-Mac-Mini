// Throughput of the FSOT archive scalar S = K·(T1+T2+T3), pin AEB2AD.
// One binary, thread count on the command line. Compile once per flag set.
#include <algorithm>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

struct In {
    double N, P, D, dp, hits, rho, dth, scale, amp, trend, obs;
};

static double scalar(const In& a) {
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

static void fill(std::vector<In>& v) {
    for (size_t i = 0; i < v.size(); ++i) {
        double luma = 0.15 + 0.85 * double(i % 997) / 996.0;
        v[i] = In{luma * 4.0, luma, 25.0, double(i % 64) / 63.0 * 6.283185307179586,
                  luma * 0.3, 1.0, 1.0, luma, 1.0, 0.0, 0.0};
    }
}

static void work(const In* in, double* out, size_t b, size_t e) {
    for (size_t i = b; i < e; ++i)
        out[i] = scalar(in[i]);
}

int main(int argc, char** argv) {
    int threads = argc > 1 ? std::atoi(argv[1]) : 1;
    size_t n = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : (1u << 20);
    int rounds = argc > 3 ? std::atoi(argv[3]) : 5;
    if (threads < 1) threads = 1;
    if (threads > 16) threads = 16;
    if (n < 1024) n = 1024;
    std::vector<In> in(n);
    std::vector<double> out(n);
    fill(in);
    // Workers stay up for the whole run. The clock covers the scalar only.
    std::barrier gate(static_cast<std::ptrdiff_t>(threads + 1));
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            size_t chunk = (n + size_t(threads) - 1) / size_t(threads);
            size_t b = size_t(t) * chunk;
            size_t e = std::min(n, b + chunk);
            if (b < n) work(in.data(), out.data(), b, e);
            gate.arrive_and_wait();
            for (int r = 0; r < rounds; ++r) {
                gate.arrive_and_wait();
                if (b < n) work(in.data(), out.data(), b, e);
                gate.arrive_and_wait();
            }
        });
    }
    gate.arrive_and_wait();
    double sum0 = 0;
    for (double x : out) sum0 += x;
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
    double sum = 0;
    for (double x : out) sum += x;
    int mhz = 0;
    if (FILE* f = std::fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r")) {
        if (std::fscanf(f, "%d", &mhz) == 1) mhz /= 1000;
        std::fclose(f);
    }
    std::printf("{\"threads\":%d,\"n\":%zu,\"median_s\":%.6f,\"scalar_per_s\":%.3f,"
                "\"sum\":%.8f,\"s0\":%.8f,\"s_last\":%.8f,\"cpu0_mhz\":%d}\n",
                threads, n, med, double(n) / med, sum, out[0], out[n - 1], mhz);
    if (sum == 0 || sum != sum0) return 0;
    return 0;
}
