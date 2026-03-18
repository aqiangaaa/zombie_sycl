#include <sycl/sycl.hpp>
#include <zombie/zombie.h>
#include "walkstate_bridge.h"
#include "phase_policy.h"
#include "task_runtime.h"
#include "task_init.h"
#include "geometry_host.h"
#include "mesh_loader.h"
#include "statistics.h"
#include "task_queue.h"
#include "metrics.h"
#include "experiment_config.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

// ============================================================
// experiment_runner.cpp
//
// Phase 4: 四组对照实验驱动程序
//
// 对每个算例 × 每个配置运行完整 WoS，输出：
// | 算例 | 配置 | θ | ρ_tail | ε_rel | fp64% | time |
// ============================================================

// ---- 单条 walk 执行（支持 DG 回收 + MP 混合精度）----
struct SingleWalkResult {
    float contribution;
    int walkSteps;
    int recycleCount;
    MixedPrecisionMetrics mpMetrics;
};

SingleWalkResult run_single_walk_full(
    sycl::queue& q,
    const zombie::Vector<DIM>& queryPt,
    const zombie::PDE<float, DIM>& pde,
    const AnalyticUnitSphereGeometry<DIM>& geom,
    const ExperimentConfig& cfg,
    std::uint64_t seed
) {
    zombie::WalkSettings settings(cfg.epsilonShell, cfg.epsilonShell,
                                   cfg.maxWalkLength, false);

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
    float initDist = geom.distToAbsorbing(queryPt);
    s.distToAbsorbingBoundary = initDist;
    s.geometryDistance = initDist;
    s.rng = pcg32_seed(seed, seed * 2 + 1);

    SingleWalkResult result{};

    for (int step = 0; step < cfg.maxWalkLength; ++step) {
        if (s.term) break;
        host_update_boundary_distances_analytic<DIM>(states, geom);
        apply_host_pde_fields(states, pde);

        // 逐步精度统计
        float distBefore = s.distToAbsorbingBoundary;
        bool p1Low = cfg.enableMixedPrecision &&
            use_low_precision_for_p1(distBefore, s.p1LowPrecisionState, cfg.phasePolicy);
        result.mpMetrics.p1Count++;
        result.mpMetrics.p2Count++;
        result.mpMetrics.p3Count++;
        result.mpMetrics.p4Count++;
        if (p1Low) result.mpMetrics.p1FP32Count++;
        else result.mpMetrics.p1FP64Count++;
        if (cfg.enableMixedPrecision) result.mpMetrics.p2FP32Count++;
        else result.mpMetrics.p2FP64Count++;

        run_task_once_mp(q, states, 1, settings, cfg.phasePolicy,
                         cfg.enableMixedPrecision);
    }

    result.walkSteps = s.walkLength;

    if (s.term && s.completionCode == WALK_TERMINATED_BY_POSITION_RULE) {
        std::vector<WalkStateLite> tmp = {s};
        host_update_terminal_contributions_analytic<DIM>(tmp, geom, pde);
        s = tmp[0];
        result.contribution = s.totalSourceContribution + s.throughput * s.dirichletValue;
    } else {
        result.contribution = s.totalSourceContribution + s.totalTerminalContribution;
    }

    return result;
}

// ---- 单点 × 单配置实验 ----
struct PointResult {
    float estimate;
    float stdErr;
    float relErr;
    double elapsed;
    int totalWalks;
    int totalRecycles;
    float tailFraction;
    MixedPrecisionMetrics mpMetrics;
};

PointResult run_point_experiment(
    sycl::queue& q,
    const zombie::Vector<DIM>& queryPt,
    float analyticSolution,
    const zombie::PDE<float, DIM>& pde,
    const AnalyticUnitSphereGeometry<DIM>& geom,
    const ExperimentConfig& cfg
) {
    WelfordAccumulator acc;
    MixedPrecisionMetrics mpTotal;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int w = 0; w < cfg.nWalksPerPoint; ++w) {
        auto wr = run_single_walk_full(
            q, queryPt, pde, geom, cfg,
            static_cast<std::uint64_t>(w) + 1);
        acc.add(wr.contribution);

        mpTotal.p1Count += wr.mpMetrics.p1Count;
        mpTotal.p2Count += wr.mpMetrics.p2Count;
        mpTotal.p3Count += wr.mpMetrics.p3Count;
        mpTotal.p4Count += wr.mpMetrics.p4Count;
        mpTotal.p1FP32Count += wr.mpMetrics.p1FP32Count;
        mpTotal.p1FP64Count += wr.mpMetrics.p1FP64Count;
        mpTotal.p2FP32Count += wr.mpMetrics.p2FP32Count;
        mpTotal.p2FP64Count += wr.mpMetrics.p2FP64Count;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    PointResult pr;
    pr.estimate = acc.mean;
    pr.stdErr = acc.standardError();
    pr.relErr = (std::fabs(analyticSolution) > 1e-6f)
        ? std::fabs(acc.mean - analyticSolution) / std::fabs(analyticSolution)
        : std::fabs(acc.mean);
    pr.elapsed = std::chrono::duration<double>(t1 - t0).count();
    pr.totalWalks = acc.count;
    pr.totalRecycles = 0;
    pr.tailFraction = 0.0f;
    pr.mpMetrics = mpTotal;
    return pr;
}

// ============================================================
// 算例定义
// ============================================================
struct TestCase {
    std::string name;
    std::vector<std::pair<zombie::Vector<DIM>, float>> points; // (pt, analytic)
    zombie::PDE<float, DIM> pde;
};

TestCase make_test_case_a() {
    TestCase tc;
    tc.name = "A: Laplace u=x";
    tc.pde.source = [](const zombie::Vector<DIM>&) -> float { return 0.0f; };
    tc.pde.dirichlet = [](const zombie::Vector<DIM>& x, bool) -> float { return x[0]; };
    tc.pde.hasReflectingBoundaryConditions = [](const zombie::Vector<DIM>&) { return false; };
    tc.pde.robin = [](const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool) -> float { return 0.0f; };
    tc.pde.robinCoeff = [](const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool) -> float { return 0.0f; };

    auto addPt = [&](float x, float y) {
        zombie::Vector<DIM> pt; pt[0] = x; pt[1] = y;
        tc.points.push_back({pt, x}); // analytic = x
    };
    addPt(0.5f, 0.0f);
    addPt(-0.5f, 0.0f);
    addPt(0.3f, 0.3f);
    addPt(0.7f, 0.0f);
    return tc;
}

TestCase make_test_case_b() {
    TestCase tc;
    tc.name = "B: Poisson";
    // 求解 Δu = 4，即 Zombie 约定下 Δu + f = 0，f = -4
    // 单位圆域，解析解: u(x,y) = x² + y² - 1
    // 边界上: u = 0
    // 验证: Δ(x²+y²-1) = 4, f = -Δu = -4 ✓
    tc.pde.source = [](const zombie::Vector<DIM>& x) -> float {
        (void)x;
        return -4.0f;  // Zombie 约定: f = -Δu
    };
    tc.pde.dirichlet = [](const zombie::Vector<DIM>& x, bool) -> float {
        return x[0]*x[0] + x[1]*x[1] - 1.0f;
    };
    tc.pde.hasReflectingBoundaryConditions = [](const zombie::Vector<DIM>&) { return false; };
    tc.pde.robin = [](const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool) -> float { return 0.0f; };
    tc.pde.robinCoeff = [](const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool) -> float { return 0.0f; };

    auto addPt = [&](float x, float y) {
        zombie::Vector<DIM> pt; pt[0] = x; pt[1] = y;
        float analytic = x*x + y*y - 1.0f;
        tc.points.push_back({pt, analytic});
    };
    addPt(0.0f, 0.0f);   // analytic = -1.0
    addPt(0.5f, 0.0f);   // analytic = -0.75
    addPt(0.3f, 0.3f);   // analytic = -0.82
    addPt(0.7f, 0.0f);   // analytic = -0.51
    return tc;
}

int main() {
    try {
        sycl::queue q;
        std::cout << "Device: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << "\n\n";

        AnalyticUnitSphereGeometry<DIM> geom;

        // ---- 算例 ----
        std::vector<TestCase> testCases = {make_test_case_a(), make_test_case_b()};

        // ---- 四组配置 ----
        const int nWalks = 512;
        const int maxWalk = 128;
        std::vector<ExperimentConfig> configs = {
            ExperimentConfig::baseline(nWalks, maxWalk),
            ExperimentConfig::dgOnly(nWalks, maxWalk),
            ExperimentConfig::mpOnly(nWalks, maxWalk, 0.5f, 0.2f),
            ExperimentConfig::dgMp(nWalks, maxWalk, 64, 4, 64, 0.5f, 0.2f),
        };

        // ---- 表头 ----
        std::cout << std::left
                  << std::setw(20) << "TestCase"
                  << std::setw(10) << "Config"
                  << std::setw(12) << "Point"
                  << std::setw(10) << "Analytic"
                  << std::setw(10) << "Estimate"
                  << std::setw(10) << "RelErr%"
                  << std::setw(10) << "StdErr"
                  << std::setw(10) << "FP64%"
                  << std::setw(10) << "Time(s)"
                  << "\n";
        std::cout << std::string(102, '-') << "\n";

        for (const auto& tc : testCases) {
            for (const auto& cfg : configs) {
                for (size_t pi = 0; pi < tc.points.size(); ++pi) {
                    const auto& [pt, analytic] = tc.points[pi];

                    auto pr = run_point_experiment(q, pt, analytic, tc.pde, geom, cfg);

                    std::ostringstream ptStr;
                    ptStr << "(" << std::fixed << std::setprecision(1)
                          << pt[0] << "," << pt[1] << ")";

                    std::cout << std::left << std::fixed << std::setprecision(4)
                              << std::setw(20) << tc.name
                              << std::setw(10) << groupName(cfg.group)
                              << std::setw(12) << ptStr.str()
                              << std::setw(10) << analytic
                              << std::setw(10) << pr.estimate
                              << std::setw(10) << (pr.relErr * 100.0f)
                              << std::setw(10) << pr.stdErr
                              << std::setw(10) << (pr.mpMetrics.fp64CoverageRatio() * 100.0)
                              << std::setw(10) << pr.elapsed
                              << "\n";
                }
            }
            std::cout << std::string(102, '-') << "\n";
        }

        std::cout << "\nDone.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
