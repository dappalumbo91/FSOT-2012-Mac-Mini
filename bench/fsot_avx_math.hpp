// FSOT's own AVX double exp, log, sin, and cos.
// Ivy Bridge: AVX and SSE4.2, no AVX2, no FMA. Integer steps stay in SSE.
// Series and argument splits are evaluated here. Nothing calls libmvec.
#pragma once
#include <immintrin.h>
#include <limits>

static inline __m256d fsot_round(__m256d x) {
    return _mm256_round_pd(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

static inline __m128i fsot_pd_to_i32(__m256d n) {
    return _mm256_cvtpd_epi32(fsot_round(n));
}

// 2^n for each lane. n must be a whole number in [-1022, 1023].
static inline __m256d fsot_exp2i(__m256d n) {
    __m128i ni = fsot_pd_to_i32(n);
    __m128i lo = _mm_cvtepi32_epi64(ni);
    __m128i hi = _mm_cvtepi32_epi64(_mm_srli_si128(ni, 8));
    const __m128i bias = _mm_set1_epi64x(1023);
    lo = _mm_slli_epi64(_mm_add_epi64(lo, bias), 52);
    hi = _mm_slli_epi64(_mm_add_epi64(hi, bias), 52);
    __m256d s = _mm256_castpd128_pd256(_mm_castsi128_pd(lo));
    return _mm256_insertf128_pd(s, _mm_castsi128_pd(hi), 1);
}

// exp(r) = 1 + r*(1/1! + r*(1/2! + ... + r/8!)) for |r| <= 0.02.
// The next term at the edge is far below one ulp. Wider arguments use fsot_exp.
static inline __m256d fsot_exp_tiny(__m256d r) {
    const double c[] = {
        0x1.0000000000000p+0,  0x1.0000000000000p-1,  0x1.5555555555555p-3,
        0x1.5555555555555p-5,  0x1.1111111111111p-7,  0x1.6c16c16c16c17p-10,
        0x1.a01a01a01a01ap-13, 0x1.a01a01a01a01ap-16
    };
    __m256d p = _mm256_set1_pd(c[7]);
    for (int k = 6; k >= 0; --k)
        p = _mm256_add_pd(_mm256_set1_pd(c[k]), _mm256_mul_pd(r, p));
    return _mm256_add_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(r, p));
}

// exp(r) = 1 + r*(1/1! + r*(1/2! + ... + r/14!)) for |r| <= ln2/2.
static inline __m256d fsot_exp_small(__m256d r) {
    const double c[] = {
        0x1.0000000000000p+0,  0x1.0000000000000p-1,  0x1.5555555555555p-3,
        0x1.5555555555555p-5,  0x1.1111111111111p-7,  0x1.6c16c16c16c17p-10,
        0x1.a01a01a01a01ap-13, 0x1.a01a01a01a01ap-16, 0x1.71de3a556c734p-19,
        0x1.27e4fb7789f5cp-22, 0x1.ae64567f544e4p-26, 0x1.1eed8eff8d898p-29,
        0x1.6124613a86d09p-33, 0x1.93974a8c07c9dp-37
    };
    __m256d p = _mm256_set1_pd(c[13]);
    for (int k = 12; k >= 0; --k)
        p = _mm256_add_pd(_mm256_set1_pd(c[k]), _mm256_mul_pd(r, p));
    return _mm256_add_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(r, p));
}

static inline __m256d fsot_exp(__m256d x) {
    // ln2 split so n*ln2_hi is exact for the exponents this scalar produces.
    const __m256d log2e = _mm256_set1_pd(0x1.71547652b82fep+0);
    const __m256d ln2_hi = _mm256_set1_pd(0x1.62e42fef00000p-1);
    const __m256d ln2_lo = _mm256_set1_pd(0x1.473de6af278edp-34);
    __m256d n = fsot_round(_mm256_mul_pd(x, log2e));
    __m256d r = _mm256_sub_pd(x, _mm256_mul_pd(n, ln2_hi));
    r = _mm256_sub_pd(r, _mm256_mul_pd(n, ln2_lo));
    return _mm256_mul_pd(fsot_exp_small(r), fsot_exp2i(n));
}

// Short series when every lane is inside [-0.02, 0.02]. Otherwise the full exp.
static inline __m256d fsot_exp_small_arg(__m256d x) {
    __m256d ax = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
    __m256d ok = _mm256_cmp_pd(ax, _mm256_set1_pd(0.02), _CMP_LE_OQ);
    if (_mm256_movemask_pd(ok) == 0xF)
        return fsot_exp_tiny(x);
    return fsot_exp(x);
}

static inline __m128d fsot_log_sse(__m128d x) {
    const __m128i exp_mask = _mm_set1_epi64x(0x7ff0000000000000LL);
    const __m128i mant_mask = _mm_set1_epi64x(0x000fffffffffffffLL);
    const __m128i one_bits = _mm_set1_epi64x(0x3ff0000000000000LL);
    __m128i bits = _mm_castpd_si128(x);
    __m128i expi = _mm_sub_epi64(_mm_srli_epi64(_mm_and_si128(bits, exp_mask), 52),
                                 _mm_set1_epi64x(1023));
    __m128d m = _mm_castsi128_pd(_mm_or_si128(_mm_and_si128(bits, mant_mask), one_bits));
    const __m128d sqrt2 = _mm_set1_pd(0x1.6a09e667f3bcdp+0);
    __m128d big = _mm_cmpgt_pd(m, sqrt2);
    m = _mm_blendv_pd(m, _mm_mul_pd(m, _mm_set1_pd(0.5)), big);
    expi = _mm_add_epi64(expi, _mm_and_si128(_mm_castpd_si128(big), _mm_set1_epi64x(1)));
    // Lower two int32 lanes are the two exponents. cvtepi32_pd handles negatives.
    __m128i e32 = _mm_shuffle_epi32(expi, _MM_SHUFFLE(3, 1, 2, 0));
    __m128d e = _mm_cvtepi32_pd(e32);
    __m128d one = _mm_set1_pd(1.0);
    __m128d f = _mm_div_pd(_mm_sub_pd(m, one), _mm_add_pd(m, one));
    __m128d f2 = _mm_mul_pd(f, f);
    // 2*atanh(f) = 2*(f + f^3/3 + f^5/5 + ...). |f| <= (sqrt(2)-1)/(sqrt(2)+1).
    const double c[] = {
        1.0, 1.0 / 3, 1.0 / 5, 1.0 / 7, 1.0 / 9, 1.0 / 11, 1.0 / 13, 1.0 / 15,
        1.0 / 17, 1.0 / 19, 1.0 / 21, 1.0 / 23, 1.0 / 25, 1.0 / 27, 1.0 / 29
    };
    __m128d p = _mm_set1_pd(c[14]);
    for (int k = 13; k >= 0; --k)
        p = _mm_add_pd(_mm_set1_pd(c[k]), _mm_mul_pd(f2, p));
    __m128d atanh2 = _mm_mul_pd(_mm_add_pd(f, f), p);
    const __m128d ln2_hi = _mm_set1_pd(0x1.62e42fef00000p-1);
    const __m128d ln2_lo = _mm_set1_pd(0x1.473de6af278edp-34);
    return _mm_add_pd(atanh2, _mm_add_pd(_mm_mul_pd(e, ln2_hi), _mm_mul_pd(e, ln2_lo)));
}

static inline __m256d fsot_log(__m256d x) {
    __m128d lo = fsot_log_sse(_mm256_castpd256_pd128(x));
    __m128d hi = fsot_log_sse(_mm256_extractf128_pd(x, 1));
    __m256d y = _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), hi, 1);
    __m256d pos = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GT_OQ);
    __m256d nan = _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN());
    return _mm256_blendv_pd(nan, y, pos);
}

// sin(r) for |r| <= pi/2. One polynomial: the pi/2 quadrant swap is gone.
static inline __m256d fsot_sin_poly(__m256d r) {
    __m256d r2 = _mm256_mul_pd(r, r);
    const double c[] = {
        0x1.5555555555555p-3,   // 1/3!
        0x1.1111111111111p-7,   // 1/5!
        0x1.a01a01a01a01ap-13,  // 1/7!
        0x1.71de3a556c734p-19,  // 1/9!
        0x1.ae64567f544e4p-26,  // 1/11!
        0x1.6124613a86d09p-33,  // 1/13!
        0x1.ae7f3e733b81fp-41,  // 1/15!
        0x1.952c77030ad4ap-49,  // 1/17!
        0x1.2f49b46814157p-57,  // 1/19!
        0x1.71b8ef6dcf572p-66   // 1/21!
    };
    __m256d p = _mm256_set1_pd(c[9]);
    for (int k = 8; k >= 0; --k)
        p = _mm256_sub_pd(_mm256_set1_pd(c[k]), _mm256_mul_pd(r2, p));
    return _mm256_mul_pd(r, _mm256_sub_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(r2, p)));
}

// cos(r) for |r| <= pi/2, through 1/22!.
static inline __m256d fsot_cos_poly(__m256d r) {
    __m256d r2 = _mm256_mul_pd(r, r);
    const double c[] = {
        0x1.0000000000000p-1,   // 1/2!
        0x1.5555555555555p-5,   // 1/4!
        0x1.6c16c16c16c17p-10,  // 1/6!
        0x1.a01a01a01a01ap-16,  // 1/8!
        0x1.27e4fb7789f5cp-22,  // 1/10!
        0x1.1eed8eff8d898p-29,  // 1/12!
        0x1.93974a8c07c9dp-37,  // 1/14!
        0x1.ae7f3e733b81fp-45,  // 1/16!
        0x1.6827863b97d97p-53,  // 1/18!
        0x1.e542ba4020225p-62,  // 1/20!
        0x1.0ce396db7f853p-70   // 1/22!
    };
    __m256d p = _mm256_set1_pd(c[10]);
    for (int k = 9; k >= 0; --k)
        p = _mm256_sub_pd(_mm256_set1_pd(c[k]), _mm256_mul_pd(r2, p));
    return _mm256_sub_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(r2, p));
}

// r = x - n*pi in [-pi/2, pi/2]. pi_hi has 20 trailing zero bits, so n*pi_hi
// is exact for |n| < 2^20. sin and cos then only flip the sign of one series.
static inline void fsot_reduce_pi(__m256d x, __m256d* r_out, __m256d* n_out) {
    const __m256d inv_hi = _mm256_set1_pd(0x1.45f306dc00000p-2);
    const __m256d inv_lo = _mm256_set1_pd(0x1.3910600000000p-35);
    const __m256d pi_hi = _mm256_set1_pd(0x1.921fb54400000p+1);
    const __m256d pi_lo = _mm256_set1_pd(0x1.0b46000000000p-33);
    __m256d n = fsot_round(_mm256_add_pd(_mm256_mul_pd(x, inv_hi), _mm256_mul_pd(x, inv_lo)));
    __m256d r = _mm256_sub_pd(x, _mm256_mul_pd(n, pi_hi));
    r = _mm256_sub_pd(r, _mm256_mul_pd(n, pi_lo));
    const __m256d halfpi = _mm256_set1_pd(0x1.921fb54442d18p+0);
    const __m256d pi = _mm256_set1_pd(0x1.921fb54442d18p+1);
    __m256d too_hi = _mm256_cmp_pd(r, halfpi, _CMP_GT_OQ);
    __m256d too_lo = _mm256_cmp_pd(r, _mm256_sub_pd(_mm256_setzero_pd(), halfpi), _CMP_LT_OQ);
    n = _mm256_add_pd(n, _mm256_and_pd(too_hi, _mm256_set1_pd(1.0)));
    n = _mm256_sub_pd(n, _mm256_and_pd(too_lo, _mm256_set1_pd(1.0)));
    r = _mm256_sub_pd(r, _mm256_and_pd(too_hi, pi));
    r = _mm256_add_pd(r, _mm256_and_pd(too_lo, pi));
    *r_out = r;
    *n_out = n;
}

static inline __m256d fsot_odd_sign(__m256d y, __m256d n) {
    __m128i ni = fsot_pd_to_i32(n);
    __m128i odd = _mm_cmpeq_epi32(_mm_and_si128(ni, _mm_set1_epi32(1)), _mm_set1_epi32(1));
    __m128i lo = _mm_cvtepi32_epi64(odd);
    __m128i hi = _mm_cvtepi32_epi64(_mm_srli_si128(odd, 8));
    __m256d mask = _mm256_castpd128_pd256(_mm_castsi128_pd(lo));
    mask = _mm256_insertf128_pd(mask, _mm_castsi128_pd(hi), 1);
    return _mm256_blendv_pd(y, _mm256_sub_pd(_mm256_setzero_pd(), y), mask);
}

static inline __m256d fsot_sin(__m256d x) {
    __m256d r, n;
    fsot_reduce_pi(x, &r, &n);
    return fsot_odd_sign(fsot_sin_poly(r), n);
}

static inline __m256d fsot_cos(__m256d x) {
    __m256d r, n;
    fsot_reduce_pi(x, &r, &n);
    return fsot_odd_sign(fsot_cos_poly(r), n);
}

static inline double fsot_exp_d(double x) {
    return _mm256_cvtsd_f64(fsot_exp(_mm256_set1_pd(x)));
}
static inline double fsot_exp_small_arg_d(double x) {
    return _mm256_cvtsd_f64(fsot_exp_small_arg(_mm256_set1_pd(x)));
}
static inline double fsot_log_d(double x) {
    return _mm256_cvtsd_f64(fsot_log(_mm256_set1_pd(x)));
}
static inline double fsot_sin_d(double x) {
    return _mm256_cvtsd_f64(fsot_sin(_mm256_set1_pd(x)));
}
static inline double fsot_cos_d(double x) {
    return _mm256_cvtsd_f64(fsot_cos(_mm256_set1_pd(x)));
}
