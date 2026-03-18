#include <sycl/sycl.hpp>
#include <zombie/zombie.h>
#include "walkstate_bridge.h"
#include "phase_policy.h"
#include "task_runtime.h"
#include "task_init.h"
#include "geometry_host.h"
#include "mesh_loader.h"
#include "statistics.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// ============================================================
// wos_sycl_baseline.cpp
//
// Phase 1 集成验证：完整 WoS SYCL 基线程序
//
// 功能：
// 1. 解析单位圆几何 + host-device 迭代循环
// 2. 多采样点 × 多 walk
// 3. Welford 在线统计（均值、方差、标准误差）
// 4. 与解析解对比
//
// 问题：2D Laplace Δu = 0，单位圆域，u = x on ∂Ω
// 解析解：u(x,y) = x
// ============================================================

// 执行单条 walk 并返回贡献值
float run_single_walk(
    sycl::queue& q,
    const zombie::Vector<DIM>& queryPt,
    float initDistA,
    const zombie::PDE<float, DIM>& pde,
    const zombie::WalkSettings& settings,
    const PhasePolicyConfig& policy,
    const AnalyticUnitSphereGeometry<DIM>& geom,
    std::uint64_t walkSeed
) {
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
    s.rng = pcg32_seed(walkSeed, walkSeed * 2 + 1);

    for (int step = 0; step < settings.maxWalkLength; ++step) {
        if (s.term) break;
        host_update_boundary_distances_analytic<DIM>(states, geom);
        apply_host_pde_fields(states, pde);
        run_task_once(q, states, 1, settings, policy);
    }

    // 终止后投影到边界并更新 dirichlet 值
    if (s.term && s.completionCode == WALK_TERMINATED_BY_POSITION_RULE) {
        host_update_terminal_contributions_analytic<DIM>(states, geom, pde);
        return s.totalSourceContribution + s.throughput * s.dirichletValue;
    }
    // 超过最大步数
    return s.totalSourceContribution + s.totalTerminalContribution;
}

int main() {
    try {
        sycl::queue q;
        std::cout << "Device: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << std::endl;

        // ---- PDE: Laplace, u = x on boundary ----
        zombie::PDE<float, DIM> pde;
        pde.source = [](const zombie::Vector<DIM>&) -> float { return 0.0f; };
        pde.dirichlet = [](const zombie::Vector<DIM>& x, bool) -> float { return x[0]; };
        pde.hasReflectingBoundaryConditions = [](const zombie::Vector<DIM>&) -> bool { return false; };
        pde.robin = [](const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool) -> float { return 0.0f; };
        pde.robinCoeff = [](const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool) -> float { return 0.0f; };

        // ---- Walk 设置 ----
        const float epsilonShell = 1e-3f;
        const int maxWalkLength = 128;
        const int nWalksPerPoint = 1024;
        zombie::WalkSettings settings(epsilonShell, epsilonShell, maxWalkLength, false);
        PhasePolicyConfig policy;

        // ---- 解析几何 ----
        AnalyticUnitSphereGeometry<DIM> geom;

        // ---- 采样点：选取几个有代表性的点 ----
        struct TestPoint {
            float x, y;
            float analytic; // u(x,y) = x
        };
        std::vector<TestPoint> testPoints = {
            { 0.5f,  0.0f,  0.5f},
            {-0.5f,  0.0f, -0.5f},
            { 0.0f,  0.5f,  0.0f},
            { 0.3f,  0.3f,  0.3f},
            {-0.3f, -0.3f, -0.3f},
            { 0.7f,  0.0f,  0.7f},
            { 0.0f,  0.0f,  0.0f},
        };

        std::cout << "\n============ WoS SYCL Baseline ============\n";
        std::cout << "Problem: 2D Laplace, unit disk, u = x on boundary\n";
        std::cout << "Walks per point: " << nWalksPerPoint << "\n";
        std::cout << "Max walk length: " << maxWalkLength << "\n";
        std::cout << "Epsilon shell: " << epsilonShell << "\n\n";

        auto t0 = std::chrono::high_resolution_clock::now();

        float maxRelErr = 0.0f;
        float sumRelErr = 0.0f;
        int nTested = 0;

        for (const auto& tp : testPoints) {
            zombie::Vector<DIM> pt;
            pt[0] = tp.x;
            pt[1] = tp.y;
            float initDistA = geom.distToAbsorbing(pt);

            PointEstimate est;
            est.pt[0] = tp.x;
            est.pt[1] = tp.y;
            est.analyticSolution = tp.analytic;

            for (int w = 0; w < nWalksPerPoint; ++w) {
                float contribution = run_single_walk(
                    q, pt, initDistA, pde, settings, policy, geom,
                    static_cast<std::uint64_t>(nTested * nWalksPerPoint + w) + 1);
                est.solution.add(contribution);
            }

            float relErr = est.relativeError();
            if (std::fabs(tp.analytic) > 0.05f) {
                // 只对解析解不太接近 0 的点统计
                if (relErr > maxRelErr) maxRelErr = relErr;
                sumRelErr += relErr;
            }
            nTested++;

            std::cout << "  pt=(" << tp.x << ", " << tp.y << ")"
                      << "  analytic=" << tp.analytic
                      << "  estimate=" << est.solution.mean
                      << "  stdErr=" << est.solution.standardError()
                      << "  relErr=" << (relErr * 100.0f) << "%"
                      << "  nWalks=" << est.solution.count
                      << "\n";
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        float avgRelErr = (nTested > 0) ? sumRelErr / nTested : 0.0f;
        int totalWalks = nTested * nWalksPerPoint;

        std::cout << "\n============ Summary ============\n";
        std::cout << "Points tested: " << nTested << "\n";
        std::cout << "Total walks: " << totalWalks << "\n";
        std::cout << "Avg relative error (|analytic|>0.05): " << (avgRelErr * 100.0f) << "%\n";
        std::cout << "Max relative error (|analytic|>0.05): " << (maxRelErr * 100.0f) << "%\n";
        std::cout << "Elapsed time: " << elapsed << " s\n";
        std::cout << "Throughput: " << totalWalks / elapsed << " walks/s\n";
        std::cout << "=================================\n";

        if (maxRelErr < 0.05f) {
            std::cout << "PASS: max relative error < 5% (for |analytic|>0.05)\n";
        } else {
            std::cout << "WARN: max relative error >= 5%\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
