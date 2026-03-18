#pragma once

#include <sycl/sycl.hpp>
#include "walkstate_bridge.h"
#include "phase_policy.h"
#include "greens_function_device.h"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cmath>

// ============================================================
// 一次任务块执行后的统计信息
// ============================================================
struct TaskExecutionStats {
    int finishedCount = 0;
    int unfinishedCount = 0;

    int oldGranularity = 0;
    int newGranularity = 0;
    int repackedSize = 0;

    bool tailMode = false;
    bool recycleDecision = false;

    int p0Count = 0;
    int p1Count = 0;
    int p2Count = 0;
    int p3Count = 0;
    int p4Count = 0;

    int p2LowPrecisionCount = 0;
    int p2HighPrecisionCount = 0;

    int p1LowPrecisionCount = 0;
    int p1HighPrecisionCount = 0;

    int terminatedByMaxLengthCount = 0;
    int terminatedByPositionRuleCount = 0;

    float totalSourceContribution = 0.0f;
    float totalReflectingBoundaryContribution = 0.0f;

    float totalTerminalContribution = 0.0f;

    float totalThroughput = 0.0f;
    int totalWalkLength = 0;

    float avgThroughput = 0.0f;
    float avgWalkLength = 0.0f;

    float tailFraction = 0.0f;

};

// ============================================================
// 最小完成码枚举
//
// 作用：
// 1. 让 runtime 不再只有 term=0/1 这种过于粗糙的状态
// 2. 为后面接真实 WoSt 的 completion code 做过渡
// ============================================================
enum WalkCompletionCodeLite : int {
    WALK_ONGOING = 0,
    WALK_TERMINATED_BY_MAX_LENGTH = 1,
    WALK_TERMINATED_BY_POSITION_RULE = 2
};

// ------------------------------------------------------------
// 计算源项贡献
//
// 对齐 walk_on_spheres.h:
//   totalSourceContribution += throughput * greensFn->norm() * source(x)
//
// 其中 greensFn->norm() 对 Harmonic Ball:
//   2D: R²/4,  3D: R²/6
// R = distToAbsorbingBoundary（当前步球半径）
// ------------------------------------------------------------
inline float compute_source_contribution(float ballRadius,
                                          const WalkStateLite& s) {
    return s.throughput * greens_fn_norm(ballRadius) * s.sourceValue;
}

// ------------------------------------------------------------
// 计算反射边界贡献
//
// WoS 模式下无反射边界贡献，返回 0。
// WoSt 模式下后续替换为真实 Robin 边界积分。
// ------------------------------------------------------------
inline float compute_reflecting_boundary_contribution(
    float ballRadius,
    const WalkStateLite& s
) {
    (void)ballRadius;
    (void)s;
    return 0.0f;
}



// ------------------------------------------------------------
// 计算 throughput 更新
//
// 对齐 walk_on_spheres.h:
//   throughput *= greensFn->directionSampledPoissonKernel(currentPt)
//
// Harmonic WoS 下均匀方向采样时恒为 1.0。
// ------------------------------------------------------------
inline float compute_throughput_update(float ballRadius,
                                        const WalkStateLite& s) {
    (void)ballRadius;
    (void)s;
    return direction_sampled_poisson_kernel();
}

// ------------------------------------------------------------
// 终止判定（当前 skeleton，保持与此前原型行为一致）
//
// 这一版的目标：
// 1. 保留 completionCode 接口
// 2. 不改变你前面已经验证过的运行行为
//
// 当前规则：
// - 只有同时满足：
//     walkLength >= settings.maxWalkLength
//     且 currentPt[0] < 4.0f
//   才终止
//
// completionCode 仍然会被写出来，但先不再引入新的“提前终止”分支。
// ------------------------------------------------------------
inline WalkCompletionCodeLite should_terminate_skeleton(
    const WalkStateLite& s,
    const zombie::WalkSettings& settings
) {
    // 先按本地真实 WoS 规则收敛：
    // 1. 到达 absorbing epsilon shell 则结束
    if (s.distToAbsorbingBoundary <= settings.epsilonShellForAbsorbingBoundary &&
        s.walkLength > 0) {
        return WALK_TERMINATED_BY_POSITION_RULE;
    }

    // 2. walkLength 严格大于 maxWalkLength 才按最大步长终止
    if (s.walkLength > settings.maxWalkLength) {
        return WALK_TERMINATED_BY_MAX_LENGTH;
    }

    return WALK_ONGOING;
}


inline void sample_unit_direction_2d(PCG32State& rng, float& dx, float& dy) {
    const float u = pcg32_next_float(rng);
    const float theta = 2.0f * 3.14159265358979323846f * u;
    dx = std::cos(theta);
    dy = std::sin(theta);
}

inline void sample_unit_direction_3d(PCG32State& rng,
                                      float& dx, float& dy, float& dz) {
    const float u1 = pcg32_next_float(rng);
    const float u2 = pcg32_next_float(rng);
    const float z = 2.0f * u1 - 1.0f;
    const float r = std::sqrt(1.0f - z * z);
    const float phi = 2.0f * 3.14159265358979323846f * u2;
    dx = r * std::cos(phi);
    dy = r * std::sin(phi);
    dz = z;
}

// 统一接口：根据 DIM 采样方向，写入 dir[MAX_DIM]
inline void sample_unit_direction(PCG32State& rng, float* dir) {
#if ZOMBIE_SYCL_DIM == 2
    sample_unit_direction_2d(rng, dir[0], dir[1]);
    dir[2] = 0.0f;
#else
    sample_unit_direction_3d(rng, dir[0], dir[1], dir[2]);
#endif
}



// ------------------------------------------------------------
// 计算终止贡献
//
// 对齐 walk_on_spheres.h:
//   terminal contribution = throughput * dirichlet(projectedPt)
//
// dirichletValue 由 host 侧 PDE 回调预计算。
// ------------------------------------------------------------
inline float compute_terminal_contribution(const WalkStateLite& s) {
    return s.throughput * s.dirichletValue;
}



// ============================================================
// 单状态执行函数
//
// 新增参数：policy
// 作用：把阶段精度策略从“硬编码常量”升级成“显式配置”
// ============================================================
inline void execute_one_walkstate(WalkStateLite& s,
                                  const zombie::WalkSettings& settings,
                                  const PhasePolicyConfig& policy) {
    if (s.term) return;

    const bool hasReflecting =
        (s.hasReflectingBoundary != 0);
    (void)hasReflecting;

    // ---------------- P0: 初始化阶段 ----------------
    // 这一阶段目前只做阶段标记；
    // 后面如果你要接更真实的执行上下文，可以从这里扩展。
    s.phase = P0_Init;
    s.visitedP0 = 1;

    // ---------------- P1: 几何阶段 ----------------
    // 读取 host 预计算的 distToAbsorbingBoundary 作为球半径
    s.geometryDistance = s.distToAbsorbingBoundary;
    const float ballRadius = s.distToAbsorbingBoundary;

    // P1 滞回精度策略
    if (use_low_precision_for_p1(s.geometryDistance,
                                 s.p1LowPrecisionState,
                                 policy)) {
        s.usedLowPrecisionP1 = 1;
        s.p1LowPrecisionState = 1;
    } else {
        s.usedLowPrecisionP1 = 0;
        s.p1LowPrecisionState = 0;
    }
    s.phase = P1_Geometry;
    s.visitedP1 = 1;

    // ---------------- P4-source: 源项贡献 ----------------
    // 对齐 walk_on_spheres.h: source contribution 在位置推进之前计算
    // totalSourceContribution += throughput * greensFn->norm() * source(x)
    s.totalSourceContribution +=
        compute_source_contribution(ballRadius, s);

    // ---------------- P2: 步进阶段 ----------------
    // P2 精度标记
    s.usedLowPrecisionP2 = use_low_precision_for_phase(P2_Step) ? 1 : 0;

    // 采样单位方向（2D 或 3D，由编译期 DIM 决定）
    float dir[MAX_DIM] = {0.0f};
    sample_unit_direction(s.rng, dir);

    // 对齐 walk_on_spheres.h: currentPt += distToAbsorbingBoundary * direction
    for (int d = 0; d < DIM; ++d) {
        s.currentPt[d] += ballRadius * dir[d];
        s.prevDirection[d] = dir[d];
    }
    s.prevDistance = ballRadius;
    s.phase = P2_Step;
    s.visitedP2 = 1;

    // ---------------- P4-throughput: 权重更新 ----------------
    // 对齐 walk_on_spheres.h: 位置推进之后更新 throughput
    // throughput *= greensFn->directionSampledPoissonKernel(currentPt)
    s.throughput *= compute_throughput_update(ballRadius, s);

    // ---------------- P4-walkLength: 步数递增 + 终止检查 ----------------
    // 对齐 walk_on_spheres.h: walkLength++ 在 throughput 更新之后
    s.walkLength += 1;

    // ---------------- P3: 近边界阶段 ----------------
    // WoS 模式下 P3 为占位，不处理反射边界
    s.onReflectingBoundary = 0;
    s.phase = P3_NearBoundary;
    s.visitedP3 = 1;

    // ---------------- P4: 终止判定与贡献累积 ----------------
    s.phase = P4_Contribution;
    s.visitedP4 = 1;

    // 终止检查：distToAbsorbingBoundary 将在下一轮 host 侧更新后生效
    // 这里用当前已知的距离做检查（host-device 循环中每步都会更新距离）
    s.completionCode = should_terminate_skeleton(s, settings);
    if (s.completionCode != WALK_ONGOING) {
        // 对齐 walk_on_spheres.h: getTerminalContribution
        // terminal contribution = throughput * dirichlet(projectedPt)
        // dirichletValue 由 host 侧在终止后投影并计算
        if (s.completionCode == WALK_TERMINATED_BY_POSITION_RULE ||
            s.completionCode == WALK_TERMINATED_BY_MAX_LENGTH) {
            s.totalTerminalContribution += compute_terminal_contribution(s);
        }
        s.term = 1;
    }
}

// ============================================================
// 执行一个任务块一次
//
// 新增参数：policy
// 这样任务运行时同时接收：
// - Zombie 原生运行参数 WalkSettings
// - 论文侧阶段精度策略 PhasePolicyConfig
// ============================================================
inline void run_task_once(sycl::queue& q,
                          std::vector<WalkStateLite>& walkTaskStates,
                          int nStepsPerLaunch,
                          const zombie::WalkSettings& settings,
                          const PhasePolicyConfig& policy) {
    if (walkTaskStates.empty()) return;

    sycl::buffer<WalkStateLite, 1> buf(
        walkTaskStates.data(),
        sycl::range<1>(walkTaskStates.size())
    );

    q.submit([&](sycl::handler& h) {
        auto acc = buf.get_access<sycl::access::mode::read_write>(h);

        h.parallel_for(sycl::range<1>(walkTaskStates.size()), [=](sycl::id<1> idx) {
            for (int step = 0; step < nStepsPerLaunch; ++step) {
                execute_one_walkstate(acc[idx[0]], settings, policy);
            }
        });
    });

    q.wait();
}

// ============================================================
// 是否允许回收 / 重封装
// ============================================================
inline bool should_recycle(bool tailMode,
                           int unfinishedCount,
                           int recycleThreshold) {
    if (!tailMode) {
        return false;
    }

    if (unfinishedCount == 0) {
        return false;
    }

    if (unfinishedCount > recycleThreshold) {
        return false;
    }

    return true;
}

// ============================================================
// 纯提取函数：提取未完成状态
// ============================================================
inline std::vector<WalkStateLite> extract_unfinished(
    const std::vector<WalkStateLite>& walkTaskStates
) {
    std::vector<WalkStateLite> repackedWalkTaskStates;

    for (const auto& s : walkTaskStates) {
        if (!s.term) {
            repackedWalkTaskStates.push_back(s);
        }
    }

    return repackedWalkTaskStates;
}

// ============================================================
// 动态粒度更新函数（当前最小版）
// ============================================================
inline int update_granularity(int currentTaskGranularity,
                              int repackedSize,
                              bool tailMode,
                              int B_min,
                              int B_max) {
    (void)B_max;

    if (!tailMode) {
        return currentTaskGranularity;
    }

    if (repackedSize == 0) {
        return 0;
    }

    if (repackedSize < B_min) {
        return B_min;
    }

    return repackedSize;
}

// ============================================================
// 收集一次任务块执行后的统计信息
// ============================================================
inline TaskExecutionStats collect_task_stats(
    const std::vector<WalkStateLite>& walkTaskStates,
    int currentTaskGranularity,
    int recycleThreshold,
    int B_min,
    int B_max
) {
    TaskExecutionStats stats{};
    stats.oldGranularity = currentTaskGranularity;

    for (const auto& s : walkTaskStates) {
        if (s.term) {
            stats.finishedCount++;
        } else {
            stats.unfinishedCount++;
        }

        if (s.completionCode == WALK_TERMINATED_BY_MAX_LENGTH) {
            stats.terminatedByMaxLengthCount++;
        } else if (s.completionCode == WALK_TERMINATED_BY_POSITION_RULE) {
            stats.terminatedByPositionRuleCount++;
        }

        if (s.visitedP0) stats.p0Count++;
        if (s.visitedP1) stats.p1Count++;
        if (s.visitedP2) stats.p2Count++;
        if (s.visitedP3) stats.p3Count++;
        if (s.visitedP4) stats.p4Count++;

        if (s.usedLowPrecisionP2) {
            stats.p2LowPrecisionCount++;
        } else {
            stats.p2HighPrecisionCount++;
        }

        if (s.usedLowPrecisionP1) {
            stats.p1LowPrecisionCount++;
        } else {
            stats.p1HighPrecisionCount++;
        }

        stats.totalSourceContribution += s.totalSourceContribution;
        stats.totalReflectingBoundaryContribution += s.totalReflectingBoundaryContribution;
        stats.totalTerminalContribution += s.totalTerminalContribution;
        stats.totalThroughput += s.throughput;
        stats.totalWalkLength += s.walkLength;
    }

    stats.tailMode = (stats.unfinishedCount > 0);

    stats.recycleDecision = should_recycle(
        stats.tailMode,
        stats.unfinishedCount,
        currentTaskGranularity == 0 ? 0 : recycleThreshold
    );

    stats.repackedSize = stats.recycleDecision ? stats.unfinishedCount : 0;

    stats.newGranularity = update_granularity(
        currentTaskGranularity,
        stats.repackedSize,
        stats.tailMode,
        B_min,
        B_max
    );

    if (!walkTaskStates.empty()) {
        const float taskSize = static_cast<float>(walkTaskStates.size());

        stats.avgThroughput = stats.totalThroughput / taskSize;
        stats.avgWalkLength = static_cast<float>(stats.totalWalkLength) / taskSize;
        stats.tailFraction = static_cast<float>(stats.unfinishedCount) / taskSize;
    }

    return stats;
}


inline float total_contribution_of_state(const WalkStateLite& s) {
    return s.totalSourceContribution +
           s.totalTerminalContribution +
           s.totalReflectingBoundaryContribution;
}

inline void run_task_until_completion(
    sycl::queue& q,
    std::vector<WalkStateLite>& walkTaskStates,
    int nStepsPerLaunch,
    const zombie::WalkSettings& settings,
    const PhasePolicyConfig& policy,
    int recycleThreshold,
    int B_min,
    int B_max
) {
    if (walkTaskStates.empty()) return;

    int currentTaskGranularity = static_cast<int>(walkTaskStates.size());

    for (int launch = 0; launch < 1024; ++launch) {
        run_task_once(
            q,
            walkTaskStates,
            nStepsPerLaunch,
            settings,
            policy
        );

        TaskExecutionStats stats = collect_task_stats(
            walkTaskStates,
            currentTaskGranularity,
            recycleThreshold,
            B_min,
            B_max
        );

        if (stats.unfinishedCount == 0) {
            return;
        }

        if (stats.recycleDecision) {
            walkTaskStates = extract_unfinished(walkTaskStates);

            if (walkTaskStates.empty()) {
                return;
            }

            currentTaskGranularity =
                (stats.newGranularity > 0)
                    ? stats.newGranularity
                    : static_cast<int>(walkTaskStates.size());
        }
    }
}


// ============================================================
// 打印任务块
// ============================================================
inline void print_task(const std::vector<WalkStateLite>& walkTaskStates,
                       const std::string& title) {
    std::cout << "---- " << title << " ----" << std::endl;

    for (size_t i = 0; i < walkTaskStates.size(); ++i) {
        std::cout << i << ": pt=(";
        for (int d = 0; d < DIM; ++d) {
            if (d > 0) std::cout << ", ";
            std::cout << walkTaskStates[i].currentPt[d];
        }
        std::cout << ")"
                  << ", throughput=" << walkTaskStates[i].throughput
                  << ", walkLength=" << walkTaskStates[i].walkLength
                  << ", distA=" << walkTaskStates[i].distToAbsorbingBoundary
                  << ", source=" << walkTaskStates[i].totalSourceContribution
                  << ", terminal=" << walkTaskStates[i].totalTerminalContribution
                  << ", reflectContrib="
                  << walkTaskStates[i].totalReflectingBoundaryContribution
                  << ", phase=" << walkTaskStates[i].phase
                  << ", reflect=" << walkTaskStates[i].onReflectingBoundary
                  << ", term=" << walkTaskStates[i].term
                  << ", completionCode=" << walkTaskStates[i].completionCode
                  << std::endl;
    }
}

// ============================================================
// 混合精度版本的任务块执行
//
// 使用 precision_ops.h 中的 execute_one_walkstate_mixed_precision
// enableMixedPrecision=true:  P2 FP32, P1/P3/P4 FP64
// enableMixedPrecision=false: 全 FP64（baseline 对照）
// ============================================================
#include "precision_ops.h"

inline void run_task_once_mp(sycl::queue& q,
                              std::vector<WalkStateLite>& walkTaskStates,
                              int nStepsPerLaunch,
                              const zombie::WalkSettings& settings,
                              const PhasePolicyConfig& policy,
                              bool enableMixedPrecision) {
    if (walkTaskStates.empty()) return;

    sycl::buffer<WalkStateLite, 1> buf(
        walkTaskStates.data(),
        sycl::range<1>(walkTaskStates.size())
    );

    q.submit([&](sycl::handler& h) {
        auto acc = buf.get_access<sycl::access::mode::read_write>(h);

        h.parallel_for(sycl::range<1>(walkTaskStates.size()), [=](sycl::id<1> idx) {
            for (int step = 0; step < nStepsPerLaunch; ++step) {
                execute_one_walkstate_mixed_precision(
                    acc[idx[0]], settings, policy, enableMixedPrecision);
            }
        });
    });

    q.wait();
}