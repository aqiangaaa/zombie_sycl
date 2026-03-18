#pragma once

// ============================================================
// 批量求解器（Phase 5）
//
// 将同一采样点的 N 条 walk 打包成一个 buffer，
// 每步：host 批量更新距离 → device 一次 kernel 推进 N 条 walk → 循环
//
// 也支持多采样点混合：M 个点 × N 条 walk = M*N 个 work-item
// ============================================================

#include <sycl/sycl.hpp>
#include <zombie/zombie.h>
#include "walkstate_bridge.h"
#include "phase_policy.h"
#include "task_runtime.h"
#include "task_init.h"
#include "geometry_host.h"
#include "statistics.h"
#include "pcg32_device.h"

#include <vector>
#include <chrono>

// ============================================================
// 批量初始化：为一个采样点创建 N 条 walk 的初始状态
// ============================================================
inline std::vector<WalkStateLite> batch_init_walks(
    const zombie::Vector<DIM>& queryPt,
    float initDistA,
    int nWalks,
    std::uint64_t seedBase = 0
) {
    std::vector<WalkStateLite> states(nWalks);
    for (int w = 0; w < nWalks; ++w) {
        WalkStateLite& s = states[w];
        s = WalkStateLite{};
        for (int d = 0; d < DIM; ++d) s.currentPt[d] = queryPt[d];
        for (int d = DIM; d < MAX_DIM; ++d) s.currentPt[d] = 0.0f;
        for (int d = 0; d < MAX_DIM; ++d) {
            s.currentNormal[d] = 0.0f;
            s.prevDirection[d] = 0.0f;
            s.normal[d] = 0.0f;
        }
        s.throughput = 1.0f;
        s.distToAbsorbingBoundary = initDistA;
        s.geometryDistance = initDistA;
        s.rng = pcg32_seed(
            seedBase + static_cast<std::uint64_t>(w) + 1,
            static_cast<std::uint64_t>(w * 2 + 1));
    }
    return states;
}

// ============================================================
// 批量初始化：多采样点 × 每点 N 条 walk
// ============================================================
struct BatchPointInfo {
    zombie::Vector<DIM> pt;
    float initDistA;
    float analyticSolution;
    int startIdx;  // 在全局 states 数组中的起始索引
    int nWalks;
};

inline std::vector<WalkStateLite> batch_init_multipoint(
    const std::vector<BatchPointInfo>& points,
    std::vector<BatchPointInfo>& pointsOut
) {
    int totalWalks = 0;
    for (const auto& p : points) totalWalks += p.nWalks;

    std::vector<WalkStateLite> states(totalWalks);
    pointsOut = points;

    int idx = 0;
    for (size_t pi = 0; pi < points.size(); ++pi) {
        pointsOut[pi].startIdx = idx;
        for (int w = 0; w < points[pi].nWalks; ++w) {
            WalkStateLite& s = states[idx];
            s = WalkStateLite{};
            for (int d = 0; d < DIM; ++d) s.currentPt[d] = points[pi].pt[d];
            for (int d = DIM; d < MAX_DIM; ++d) s.currentPt[d] = 0.0f;
            for (int d = 0; d < MAX_DIM; ++d) {
                s.currentNormal[d] = 0.0f;
                s.prevDirection[d] = 0.0f;
                s.normal[d] = 0.0f;
            }
            s.throughput = 1.0f;
            s.distToAbsorbingBoundary = points[pi].initDistA;
            s.geometryDistance = points[pi].initDistA;
            s.rng = pcg32_seed(
                static_cast<std::uint64_t>(pi * 10000 + w) + 1,
                static_cast<std::uint64_t>(pi * 10000 + w) * 2 + 1);
            idx++;
        }
    }
    return states;
}

// ============================================================
// 泛型距离更新：支持任意几何类型
// 要求 GeomType 提供 distToAbsorbing(Vector<DIM>) 方法
// ============================================================
template <int D, typename GeomType>
inline void host_update_distances_generic(
    std::vector<WalkStateLite>& states,
    const GeomType& geom
) {
    for (auto& s : states) {
        if (s.term) continue;
        zombie::Vector<D> pt = zombie::Vector<D>::Zero();
        for (int d = 0; d < D; ++d) pt[d] = s.currentPt[d];
        float dist = geom.distToAbsorbing(pt);
        s.distToAbsorbingBoundary = std::max(0.0f, dist);
        s.geometryDistance = s.distToAbsorbingBoundary;
    }
}

// ============================================================
// 泛型终止贡献更新：支持任意几何类型
// 要求 GeomType 提供 projectToAbsorbing(pt, normal, dist) 方法
// ============================================================
template <int D, typename GeomType>
inline void host_update_terminal_generic(
    std::vector<WalkStateLite>& states,
    const GeomType& geom,
    const zombie::PDE<float, D>& pde
) {
    for (auto& s : states) {
        if (!s.term) continue;
        if (s.completionCode == WALK_TERMINATED_BY_POSITION_RULE) {
            zombie::Vector<D> pt = zombie::Vector<D>::Zero();
            zombie::Vector<D> normal = zombie::Vector<D>::Zero();
            float dist = 0.0f;
            for (int d = 0; d < D; ++d) pt[d] = s.currentPt[d];
            if (geom.projectToAbsorbing(pt, normal, dist)) {
                if (pde.dirichlet) {
                    s.dirichletValue = pde.dirichlet(pt, false);
                }
            }
        }
    }
}

// ============================================================
// 批量求解：host-device 迭代循环
// ============================================================
struct BatchSolveResult {
    std::vector<PointEstimate> estimates;
    double totalTime;
    double hostTime;   // host 距离计算时间
    double deviceTime; // device kernel 时间
    int totalSteps;    // 总迭代步数
};

template <typename GeomType>
inline BatchSolveResult batch_solve(
    sycl::queue& q,
    std::vector<WalkStateLite>& states,
    const std::vector<BatchPointInfo>& points,
    const GeomType& geom,
    const zombie::PDE<float, DIM>& pde,
    const zombie::WalkSettings& settings,
    const PhasePolicyConfig& policy,
    bool enableMixedPrecision
) {
    BatchSolveResult result;
    result.hostTime = 0.0;
    result.deviceTime = 0.0;
    result.totalSteps = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    // Host-Device 迭代循环
    for (int step = 0; step < settings.maxWalkLength; ++step) {
        // 检查是否所有 walk 都已终止
        bool allDone = true;
        for (const auto& s : states) {
            if (!s.term) { allDone = false; break; }
        }
        if (allDone) break;

        // Host: 批量更新边界距离
        auto th0 = std::chrono::high_resolution_clock::now();
        host_update_distances_generic<DIM>(states, geom);
        apply_host_pde_fields(states, pde);
        auto th1 = std::chrono::high_resolution_clock::now();
        result.hostTime += std::chrono::duration<double>(th1 - th0).count();

        // Device: 一次 kernel 推进所有 walk 一步
        auto td0 = std::chrono::high_resolution_clock::now();
        run_task_once_mp(q, states, 1, settings, policy, enableMixedPrecision);
        auto td1 = std::chrono::high_resolution_clock::now();
        result.deviceTime += std::chrono::duration<double>(td1 - td0).count();

        result.totalSteps++;
    }

    // 终止后投影到边界更新 dirichlet 值
    host_update_terminal_generic<DIM>(states, geom, pde);

    // 收集每个采样点的统计
    result.estimates.resize(points.size());
    for (size_t pi = 0; pi < points.size(); ++pi) {
        PointEstimate& est = result.estimates[pi];
        for (int d = 0; d < DIM; ++d) est.pt[d] = points[pi].pt[d];
        est.analyticSolution = points[pi].analyticSolution;

        for (int w = 0; w < points[pi].nWalks; ++w) {
            int idx = points[pi].startIdx + w;
            const WalkStateLite& s = states[idx];

            float contribution;
            if (s.term && s.completionCode == WALK_TERMINATED_BY_POSITION_RULE) {
                contribution = s.totalSourceContribution +
                               s.throughput * s.dirichletValue;
            } else {
                contribution = s.totalSourceContribution +
                               s.totalTerminalContribution;
            }
            est.solution.add(contribution);
            est.walkLen.add(static_cast<float>(s.walkLength));
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.totalTime = std::chrono::duration<double>(t1 - t0).count();

    return result;
}

// ============================================================
// 便捷接口：单采样点批量求解
// ============================================================
template <typename GeomType>
inline PointEstimate batch_solve_single_point(
    sycl::queue& q,
    const zombie::Vector<DIM>& queryPt,
    float analyticSolution,
    const GeomType& geom,
    const zombie::PDE<float, DIM>& pde,
    const zombie::WalkSettings& settings,
    const PhasePolicyConfig& policy,
    bool enableMixedPrecision,
    int nWalks,
    double* outTime = nullptr
) {
    float initDistA = geom.distToAbsorbing(queryPt);
    auto states = batch_init_walks(queryPt, initDistA, nWalks);

    BatchPointInfo info;
    info.pt = queryPt;
    info.initDistA = initDistA;
    info.analyticSolution = analyticSolution;
    info.startIdx = 0;
    info.nWalks = nWalks;

    auto result = batch_solve(q, states, {info}, geom, pde, settings,
                               policy, enableMixedPrecision);

    if (outTime) *outTime = result.totalTime;
    return result.estimates[0];
}
