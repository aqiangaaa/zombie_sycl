#include <sycl/sycl.hpp>
#include <zombie/zombie.h>
#include "walkstate_bridge.h"
#include "phase_policy.h"
#include "task_runtime.h"
#include "task_init.h"
#include "geometry_host.h"
#include "statistics.h"
#include "batch_solver.h"
#include "experiment_config.h"
#include "metrics.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

// ============================================================
// batch_test.cpp
//
// Phase 5 验证：批量并行 vs 逐 walk 串行
//
// 对比：
// 1. 正确性：两种方式的估计值应一致
// 2. 性能：批量版本应显著更快
// ============================================================

int main() {
    try {
        sycl::queue q;
        std::cout << "Device: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << "\n\n";

        // ---- PDE: Laplace u=x ----
        zombie::PDE<float, DIM> pde;
        pde.source = [](const zombie::Vector<DIM>&) -> float { return 0.0f; };
        pde.dirichlet = [](const zombie::Vector<DIM>& x, bool) -> float { return x[0]; };
        pde.hasReflectingBoundaryConditions = [](const zombie::Vector<DIM>&) { return false; };
        pde.robin = [](const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool) -> float { return 0.0f; };
        pde.robinCoeff = [](const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool) -> float { return 0.0f; };

        zombie::WalkSettings settings(1e-3f, 1e-3f, 128, false);
        PhasePolicyConfig policy;
        AnalyticUnitSphereGeometry<DIM> geom;

        // ---- 测试点 ----
        struct TP { float x, y, analytic; };
        std::vector<TP> testPoints = {
            {0.5f, 0.0f, 0.5f},
            {-0.5f, 0.0f, -0.5f},
            {0.3f, 0.3f, 0.3f},
            {0.7f, 0.0f, 0.7f},
        };

        const int nWalks = 1024;

        std::cout << "============ Batch vs Sequential ============\n";
        std::cout << "Walks per point: " << nWalks << "\n\n";

        // ---- 方式1: 批量并行（多采样点混合）----
        {
            std::vector<BatchPointInfo> batchPoints;
            for (const auto& tp : testPoints) {
                BatchPointInfo bp;
                bp.pt = zombie::Vector<DIM>::Zero();
                bp.pt[0] = tp.x; bp.pt[1] = tp.y;
                bp.initDistA = geom.distToAbsorbing(bp.pt);
                bp.analyticSolution = tp.analytic;
                bp.nWalks = nWalks;
                batchPoints.push_back(bp);
            }

            std::vector<BatchPointInfo> pointsOut;
            auto states = batch_init_multipoint(batchPoints, pointsOut);

            auto result = batch_solve(q, states, pointsOut, geom, pde,
                                       settings, policy, false);

            std::cout << "[Batch Parallel] total=" << result.totalTime << "s"
                      << "  host=" << result.hostTime << "s"
                      << "  device=" << result.deviceTime << "s"
                      << "  steps=" << result.totalSteps << "\n";

            for (size_t i = 0; i < result.estimates.size(); ++i) {
                const auto& est = result.estimates[i];
                std::cout << "  pt=(" << testPoints[i].x << "," << testPoints[i].y << ")"
                          << "  analytic=" << testPoints[i].analytic
                          << "  estimate=" << std::fixed << std::setprecision(4) << est.solution.mean
                          << "  relErr=" << std::setprecision(2) << (est.relativeError() * 100.0f) << "%"
                          << "  avgWalkLen=" << std::setprecision(1) << est.walkLen.mean
                          << "\n";
            }
            std::cout << "\n";
        }

        // ---- 方式2: 逐 walk 串行（原 baseline 方式）----
        {
            auto t0 = std::chrono::high_resolution_clock::now();

            for (const auto& tp : testPoints) {
                zombie::Vector<DIM> pt; pt[0] = tp.x; pt[1] = tp.y;
                double ptTime;
                auto est = batch_solve_single_point(
                    q, pt, tp.analytic, geom, pde, settings, policy,
                    false, nWalks, &ptTime);

                // 第一个点时输出标题
                static bool first = true;
                if (first) {
                    auto t0_inner = std::chrono::high_resolution_clock::now();
                    // 逐点也用 batch_solve_single_point，但每个点独立调用
                    first = false;
                }

                std::cout << "[Per-Point Batch] pt=(" << tp.x << "," << tp.y << ")"
                          << "  estimate=" << std::fixed << std::setprecision(4) << est.solution.mean
                          << "  relErr=" << std::setprecision(2) << (est.relativeError() * 100.0f) << "%"
                          << "  time=" << std::setprecision(3) << ptTime << "s"
                          << "\n";
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            double seqTotal = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "[Per-Point Batch] total=" << std::setprecision(3) << seqTotal << "s\n\n";
        }

        std::cout << "============ Done ============\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
