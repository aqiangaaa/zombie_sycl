#include <sycl/sycl.hpp>
#include <zombie/zombie.h>
#include "walkstate_bridge.h"
#include "phase_policy.h"
#include "task_runtime.h"
#include "task_init.h"
#include "geometry_host.h"
#include "statistics.h"
#include "metrics.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// ============================================================
// wos_sycl_mp.cpp
//
// Phase 3 集成验证：混合精度版 WoS
//
// 对比两种模式：
// - baseline (全 FP64): enableMixedPrecision = false
// - MP (混合精度):       enableMixedPrecision = true
//
// 问题：2D Laplace Δu = 0，单位圆域，u = x on ∂Ω
// ============================================================

struct RunResult {
    float estimate;
    float stdErr;
    float relErr;
    double elapsed;
    MixedPrecisionMetrics mpMetrics;
};

RunResult run_experiment(
    sycl::queue& q,
    const zombie::Vector<DIM>& queryPt,
    float analyticSolution,
    const zombie::PDE<float, DIM>& pde,
    const zombie::WalkSettings& settings,
    const PhasePolicyConfig& policy,
    const AnalyticUnitSphereGeometry<DIM>& geom,
    int nWalks,
    bool enableMixedPrecision
) {
    float initDistA = geom.distToAbsorbing(queryPt);
    WelfordAccumulator acc;
    MixedPrecisionMetrics mpMetrics;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int w = 0; w < nWalks; ++w) {
        std::vector<WalkStateLite> states(1);
        WalkStateLite& s = states[0];
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
            static_cast<std::uint64_t>(w) + 1,
            static_cast<std::uint64_t>(w * 2 + 1));

        // 累积精度统计（每步都累积）
        for (int step = 0; step < settings.maxWalkLength; ++step) {
            if (s.term) break;
            host_update_boundary_distances_analytic<DIM>(states, geom);
            apply_host_pde_fields(states, pde);

            // 记录本步 P1 精度选择前的距离
            float distBefore = s.distToAbsorbingBoundary;
            bool p1WouldUseLow = enableMixedPrecision &&
                use_low_precision_for_p1(distBefore, s.p1LowPrecisionState, policy);

            run_task_once_mp(q, states, 1, settings, policy, enableMixedPrecision);

            // 累积本步统计
            mpMetrics.p1Count++;
            mpMetrics.p2Count++;
            mpMetrics.p3Count++;
            mpMetrics.p4Count++;
            if (p1WouldUseLow) {
                mpMetrics.p1FP32Count++;
            } else {
                mpMetrics.p1FP64Count++;
            }
            if (enableMixedPrecision) {
                mpMetrics.p2FP32Count++;
            } else {
                mpMetrics.p2FP64Count++;
            }
        }

        // 收集贡献
        float contribution;
        if (s.term && s.completionCode == WALK_TERMINATED_BY_POSITION_RULE) {
            std::vector<WalkStateLite> tmp = {s};
            host_update_terminal_contributions_analytic<DIM>(tmp, geom, pde);
            s = tmp[0];
            contribution = s.totalSourceContribution + s.throughput * s.dirichletValue;
        } else {
            contribution = s.totalSourceContribution + s.totalTerminalContribution;
        }
        acc.add(contribution);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    RunResult result;
    result.estimate = acc.mean;
    result.stdErr = acc.standardError();
    result.relErr = (std::fabs(analyticSolution) > 1e-6f)
        ? std::fabs(acc.mean - analyticSolution) / std::fabs(analyticSolution)
        : std::fabs(acc.mean);
    result.elapsed = elapsed;
    result.mpMetrics = mpMetrics;
    return result;
}

int main() {
    try {
        sycl::queue q;
        std::cout << "Device: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << std::endl;

        // ---- PDE ----
        zombie::PDE<float, DIM> pde;
        pde.source = [](const zombie::Vector<DIM>&) -> float { return 0.0f; };
        pde.dirichlet = [](const zombie::Vector<DIM>& x, bool) -> float { return x[0]; };
        pde.hasReflectingBoundaryConditions = [](const zombie::Vector<DIM>&) -> bool { return false; };
        pde.robin = [](const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool) -> float { return 0.0f; };
        pde.robinCoeff = [](const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool) -> float { return 0.0f; };

        // ---- Walk 设置 ----
        const int maxWalkLength = 128;
        const int nWalks = 1024;
        zombie::WalkSettings settings(1e-3f, 1e-3f, maxWalkLength, false);

        AnalyticUnitSphereGeometry<DIM> geom;

        // ---- 测试点 ----
        struct TestPoint { float x, y, analytic; };
        std::vector<TestPoint> testPoints = {
            { 0.5f,  0.0f,  0.5f},
            {-0.5f,  0.0f, -0.5f},
            { 0.3f,  0.3f,  0.3f},
            { 0.7f,  0.0f,  0.7f},
        };

        // ---- 滞回参数扫描 ----
        struct PolicyConfig {
            float tau_in, tau_out;
            std::string label;
        };
        std::vector<PolicyConfig> policies = {
            {0.7f, 0.4f, "MP(0.7/0.4)"},
            {0.5f, 0.2f, "MP(0.5/0.2)"},
            {0.9f, 0.6f, "MP(0.9/0.6)"},
        };

        std::cout << "\n============ WoS SYCL Mixed Precision ============\n";
        std::cout << "Walks per point: " << nWalks << "\n";
        std::cout << "Max walk length: " << maxWalkLength << "\n\n";

        // ---- 对每个测试点运行 baseline + 各 MP 配置 ----
        for (const auto& tp : testPoints) {
            zombie::Vector<DIM> pt;
            pt[0] = tp.x;
            pt[1] = tp.y;

            std::cout << "--- pt=(" << tp.x << ", " << tp.y
                      << ") analytic=" << tp.analytic << " ---\n";

            // Baseline (全 FP64)
            PhasePolicyConfig baselinePolicy;
            auto blResult = run_experiment(
                q, pt, tp.analytic, pde, settings, baselinePolicy,
                geom, nWalks, false);

            std::cout << "  [Baseline FP64] est=" << blResult.estimate
                      << " stdErr=" << blResult.stdErr
                      << " relErr=" << (blResult.relErr * 100.0f) << "%"
                      << " time=" << blResult.elapsed << "s"
                      << " fp64ratio=" << (blResult.mpMetrics.fp64CoverageRatio() * 100.0f) << "%"
                      << "\n";

            // 各 MP 配置
            for (const auto& pc : policies) {
                PhasePolicyConfig mpPolicy;
                mpPolicy.tau_in = pc.tau_in;
                mpPolicy.tau_out = pc.tau_out;

                auto mpResult = run_experiment(
                    q, pt, tp.analytic, pde, settings, mpPolicy,
                    geom, nWalks, true);

                // 与 baseline 的误差差异
                float errDiff = mpResult.relErr - blResult.relErr;

                std::cout << "  [" << pc.label << "] est=" << mpResult.estimate
                          << " stdErr=" << mpResult.stdErr
                          << " relErr=" << (mpResult.relErr * 100.0f) << "%"
                          << " errDiff=" << (errDiff * 100.0f) << "pp"
                          << " time=" << mpResult.elapsed << "s"
                          << " fp64ratio=" << (mpResult.mpMetrics.fp64CoverageRatio() * 100.0f) << "%"
                          << " P1low=" << mpResult.mpMetrics.p1FP32Count
                          << " P1high=" << mpResult.mpMetrics.p1FP64Count
                          << "\n";
            }
            std::cout << "\n";
        }

        std::cout << "============ Done ============\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
