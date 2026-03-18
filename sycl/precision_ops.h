#pragma once

// ============================================================
// 精度算子（论文第5章）
//
// 为 WalkTask::execute 内部的各阶段提供 FP32/FP64 双版本算子。
//
// 精度映射 s(p):
//   P0: FP64（初始化，随执行路径统一）
//   P1: FP64（几何距离 + 近边界判定），滞回时可降为 FP32
//   P2: FP32（采样 + 步进更新）
//   P3: FP64（近边界精细处理）
//   P4: FP64（终止判定 + 贡献累积）
// ============================================================

#include <cmath>
#include <cstdint>
#include "walkstate_bridge.h"
#include "pcg32_device.h"
#include "greens_function_device.h"

// ============================================================
// P1: 几何距离计算
// ============================================================

// FP64 路径：事件敏感，保护近边界判定
inline double p1_geometry_distance_fp64(double distToAbsorbing) {
    return distToAbsorbing;
}

// FP32 路径：远离边界时使用
inline float p1_geometry_distance_fp32(float distToAbsorbing) {
    return distToAbsorbing;
}

// P1 近边界判定（始终 FP64）
inline bool p1_near_boundary_check(double dist, double epsilon) {
    return dist <= epsilon;
}

// ============================================================
// P2: 采样与步进更新（FP32）
// ============================================================

// 方向采样（FP32）
inline void p2_sample_direction_2d(PCG32State& rng, float& dx, float& dy) {
    float u = pcg32_next_float(rng);
    float theta = 2.0f * 3.14159265358979323846f * u;
    dx = std::cos(theta);
    dy = std::sin(theta);
}

inline void p2_sample_direction_3d(PCG32State& rng,
                                    float& dx, float& dy, float& dz) {
    float u1 = pcg32_next_float(rng);
    float u2 = pcg32_next_float(rng);
    float z = 2.0f * u1 - 1.0f;
    float r = std::sqrt(1.0f - z * z);
    float phi = 2.0f * 3.14159265358979323846f * u2;
    dx = r * std::cos(phi);
    dy = r * std::sin(phi);
    dz = z;
}

inline void p2_sample_direction(PCG32State& rng, float* dir) {
#if ZOMBIE_SYCL_DIM == 2
    p2_sample_direction_2d(rng, dir[0], dir[1]);
    dir[2] = 0.0f;
#else
    p2_sample_direction_3d(rng, dir[0], dir[1], dir[2]);
#endif
}

// 位置更新（FP32）
inline void p2_step_position(float* pt, float radius, const float* dir) {
    for (int d = 0; d < DIM; ++d) {
        pt[d] += radius * dir[d];
    }
}

// ============================================================
// P4: 贡献计算
// ============================================================

// Source contribution（FP64 路径）
inline double p4_source_contribution_fp64(double throughput,
                                           double ballRadius,
                                           double sourceValue) {
    // greensFn->norm() for Harmonic Ball
#if ZOMBIE_SYCL_DIM == 2
    double gNorm = ballRadius * ballRadius / 4.0;
#else
    double gNorm = ballRadius * ballRadius / 6.0;
#endif
    return throughput * gNorm * sourceValue;
}

// Source contribution（FP32 路径，用于对照实验）
inline float p4_source_contribution_fp32(float throughput,
                                          float ballRadius,
                                          float sourceValue) {
    return throughput * greens_fn_norm(ballRadius) * sourceValue;
}

// Terminal contribution（FP64 路径）
inline double p4_terminal_contribution_fp64(double throughput,
                                             double dirichletValue) {
    return throughput * dirichletValue;
}

// Terminal contribution（FP32 路径）
inline float p4_terminal_contribution_fp32(float throughput,
                                            float dirichletValue) {
    return throughput * dirichletValue;
}

// Throughput 更新（FP64 路径）
// Harmonic WoS: directionSampledPoissonKernel = 1.0
inline double p4_throughput_update_fp64() {
    return 1.0;
}

// ============================================================
// 阶段化混合精度执行：单状态一步
//
// 对齐论文算法5-1：
// - 阶段判定由确定性规则给出
// - 精度模式 s 由 s(p) 策略函数给出
// - 切换发生在阶段入口处
//
// enableMixedPrecision = true:  P2 用 FP32，P1/P3/P4 用 FP64
// enableMixedPrecision = false: 全部 FP64（baseline 对照）
// ============================================================
inline void execute_one_walkstate_mixed_precision(
    WalkStateLite& s,
    const zombie::WalkSettings& settings,
    const PhasePolicyConfig& policy,
    bool enableMixedPrecision
) {
    if (s.term) return;

    // ---- P0: 初始化 ----
    s.phase = P0_Init;
    s.visitedP0 = 1;

    // ---- P1: 几何阶段 ----
    const float ballRadius_f = s.distToAbsorbingBoundary;
    double ballRadius_d = static_cast<double>(ballRadius_f);

    // P1 滞回精度策略
    bool p1UseLow = enableMixedPrecision &&
        use_low_precision_for_p1(ballRadius_f, s.p1LowPrecisionState, policy);

    if (p1UseLow) {
        s.geometryDistance = p1_geometry_distance_fp32(ballRadius_f);
        s.usedLowPrecisionP1 = 1;
        s.p1LowPrecisionState = 1;
    } else {
        s.geometryDistance = static_cast<float>(p1_geometry_distance_fp64(ballRadius_d));
        s.usedLowPrecisionP1 = 0;
        s.p1LowPrecisionState = 0;
    }
    s.phase = P1_Geometry;
    s.visitedP1 = 1;

    // ---- P4-source: 源项贡献（位置推进之前）----
    if (enableMixedPrecision) {
        // P4 source 始终 FP64
        double src = p4_source_contribution_fp64(
            static_cast<double>(s.throughput),
            ballRadius_d,
            static_cast<double>(s.sourceValue));
        s.totalSourceContribution += static_cast<float>(src);
    } else {
        double src = p4_source_contribution_fp64(
            static_cast<double>(s.throughput),
            ballRadius_d,
            static_cast<double>(s.sourceValue));
        s.totalSourceContribution += static_cast<float>(src);
    }

    // ---- P2: 步进阶段（始终 FP32）----
    s.usedLowPrecisionP2 = enableMixedPrecision ? 1 : 0;

    float dir[MAX_DIM] = {0.0f};
    p2_sample_direction(s.rng, dir);
    p2_step_position(s.currentPt, ballRadius_f, dir);

    for (int d = 0; d < DIM; ++d) {
        s.prevDirection[d] = dir[d];
    }
    s.prevDistance = ballRadius_f;
    s.phase = P2_Step;
    s.visitedP2 = 1;

    // ---- P4-throughput: 权重更新（位置推进之后）----
    if (enableMixedPrecision) {
        s.throughput *= static_cast<float>(p4_throughput_update_fp64());
    } else {
        s.throughput *= static_cast<float>(p4_throughput_update_fp64());
    }

    // ---- walkLength++ ----
    s.walkLength += 1;

    // ---- P3: 近边界阶段（WoS 模式占位）----
    s.onReflectingBoundary = 0;
    s.phase = P3_NearBoundary;
    s.visitedP3 = 1;

    // ---- P4: 终止判定与贡献累积 ----
    s.phase = P4_Contribution;
    s.visitedP4 = 1;

    // 终止判定（FP64 保护）
    bool terminated = false;
    if (enableMixedPrecision) {
        // 近边界判定用 FP64
        double distA_d = static_cast<double>(s.distToAbsorbingBoundary);
        double eps_d = static_cast<double>(settings.epsilonShellForAbsorbingBoundary);
        if (distA_d <= eps_d && s.walkLength > 0) {
            s.completionCode = WALK_TERMINATED_BY_POSITION_RULE;
            terminated = true;
        }
    } else {
        if (s.distToAbsorbingBoundary <= settings.epsilonShellForAbsorbingBoundary &&
            s.walkLength > 0) {
            s.completionCode = WALK_TERMINATED_BY_POSITION_RULE;
            terminated = true;
        }
    }

    if (!terminated && s.walkLength > settings.maxWalkLength) {
        s.completionCode = WALK_TERMINATED_BY_MAX_LENGTH;
        terminated = true;
    }

    if (terminated) {
        // Terminal contribution（FP64）
        if (s.completionCode == WALK_TERMINATED_BY_POSITION_RULE ||
            s.completionCode == WALK_TERMINATED_BY_MAX_LENGTH) {
            double tc = p4_terminal_contribution_fp64(
                static_cast<double>(s.throughput),
                static_cast<double>(s.dirichletValue));
            s.totalTerminalContribution += static_cast<float>(tc);
        }
        s.term = 1;
    }
}
