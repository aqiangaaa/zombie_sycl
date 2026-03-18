#pragma once

// ============================================================
// 设备侧 Harmonic Green 函数（球域）
//
// 对齐 include/zombie/core/distributions.h 中的
// HarmonicGreensFnBall<2> 和 HarmonicGreensFnBall<3>。
//
// 仅保留 WoS 随机游走内核所需的最小函数集：
// - norm(R):  Green 函数在球域上的积分，用于 source contribution
// - directionSampledPoissonKernel(): Poisson 核 / 方向采样 PDF
// - evaluate(r, R): Green 函数值 G(r, R)
//
// 所有函数为纯算术 inline，可在 SYCL kernel 中安全使用。
// ============================================================

#include <cmath>

#ifndef ZOMBIE_SYCL_DIM
#define ZOMBIE_SYCL_DIM 2
#endif

// ============================================================
// Green 函数积分 (norm)
//
// 用于 source contribution:
//   totalSourceContribution += throughput * greens_fn_norm(R) * sourceValue
//
// 2D: R²/4
// 3D: R²/6
// ============================================================
inline float greens_fn_norm(float R) {
#if ZOMBIE_SYCL_DIM == 2
    return R * R / 4.0f;
#else
    return R * R / 6.0f;
#endif
}

// ============================================================
// 方向采样 Poisson 核
//
// 用于 throughput 更新:
//   throughput *= direction_sampled_poisson_kernel()
//
// Harmonic WoS 下，均匀方向采样时 Poisson 核与采样 PDF 抵消，
// 2D 和 3D 均返回 1.0。
// ============================================================
inline float direction_sampled_poisson_kernel() {
    return 1.0f;
}

// ============================================================
// Green 函数值 G(r, R)
//
// 2D: G(r, R) = ln(R/r) / (2π)
// 3D: G(r, R) = (1/r - 1/R) / (4π)
// ============================================================
inline float greens_fn_evaluate(float r, float R) {
    const float r_safe = (r < 1e-6f) ? 1e-6f : r;
#if ZOMBIE_SYCL_DIM == 2
    return std::log(R / r_safe) / (2.0f * 3.14159265358979323846f);
#else
    return (1.0f / r_safe - 1.0f / R) / (4.0f * 3.14159265358979323846f);
#endif
}

// ============================================================
// Poisson 核（常数形式，球面均匀分布时）
//
// 2D: 1 / (2π)
// 3D: 1 / (4π)
// ============================================================
inline float poisson_kernel_constant() {
#if ZOMBIE_SYCL_DIM == 2
    return 1.0f / (2.0f * 3.14159265358979323846f);
#else
    return 1.0f / (4.0f * 3.14159265358979323846f);
#endif
}
