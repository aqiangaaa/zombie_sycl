#include <sycl/sycl.hpp>
#include <zombie/zombie.h>
#include "walkstate_bridge.h"
#include "phase_policy.h"
#include "task_runtime.h"
#include "task_init.h"
#include "geometry_host.h"
#include "mesh_loader.h"

#include <iostream>
#include <vector>
#include <cmath>

// ============================================================
// Step 1.5 验证程序：
// 用 mesh_loader 生成多个域内采样点，批量执行 WoS
//
// 问题：2D Laplace Δu = 0，单位圆域，u = x on ∂Ω
// 解析解：u(x,y) = x
// ============================================================

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
        const float epsilonShell = 1e-3f;
        const int maxWalkLength = 128;
        const int nWalksPerPoint = 512;
        zombie::WalkSettings settings(epsilonShell, epsilonShell, maxWalkLength, false);
        PhasePolicyConfig policy;

        // ---- 解析几何 ----
        AnalyticUnitSphereGeometry<DIM> geom;

        // ---- 生成网格采样点 ----
        auto samplePoints = generate_grid_sample_points_unit_sphere<DIM>(6, 0.7f);
        const int nPoints = static_cast<int>(samplePoints.size());
        std::cout << "Generated " << nPoints << " sample points in unit disk\n";

        // ---- 对每个采样点执行多条 walk ----
        float maxRelError = 0.0f;
        float sumRelError = 0.0f;
        int nTested = 0;

        for (int pi = 0; pi < nPoints; ++pi) {
            const auto& sp = samplePoints[pi];
            float analyticSolution = sp.pt[0]; // u(x,y) = x

            float sumContribution = 0.0f;
            int completedWalks = 0;

            for (int w = 0; w < nWalksPerPoint; ++w) {
                std::vector<WalkStateLite> states(1);
                WalkStateLite& s = states[0];
                s = WalkStateLite{};
                for (int d = 0; d < DIM; ++d) s.currentPt[d] = sp.pt[d];
                for (int d = DIM; d < MAX_DIM; ++d) s.currentPt[d] = 0.0f;
                for (int d = 0; d < MAX_DIM; ++d) {
                    s.currentNormal[d] = 0.0f;
                    s.prevDirection[d] = 0.0f;
                    s.normal[d] = 0.0f;
                }
                s.throughput = 1.0f;
                s.walkLength = 0;
                s.term = 0;
                s.completionCode = 0;
                s.rng = pcg32_seed(
                    static_cast<std::uint64_t>(pi * nWalksPerPoint + w) + 1,
                    static_cast<std::uint64_t>(pi * nWalksPerPoint + w) * 2 + 1);

                for (int step = 0; step < maxWalkLength; ++step) {
                    if (s.term) break;
                    host_update_boundary_distances_analytic<DIM>(states, geom);
                    apply_host_pde_fields(states, pde);
                    run_task_once(q, states, 1, settings, policy);
                }

                if (s.term && s.completionCode == WALK_TERMINATED_BY_POSITION_RULE) {
                    host_update_terminal_contributions_analytic<DIM>(states, geom, pde);
                    sumContribution += s.totalSourceContribution + s.throughput * s.dirichletValue;
                    completedWalks++;
                } else if (s.term) {
                    sumContribution += s.totalSourceContribution + s.totalTerminalContribution;
                    completedWalks++;
                }
            }

            if (completedWalks > 0) {
                float estimate = sumContribution / completedWalks;
                float absAnalytic = std::fabs(analyticSolution);
                float relError = (absAnalytic > 1e-6f)
                    ? std::fabs(estimate - analyticSolution) / absAnalytic
                    : std::fabs(estimate - analyticSolution);

                if (relError > maxRelError) maxRelError = relError;
                sumRelError += relError;
                nTested++;

                std::cout << "  pt=(" << sp.pt[0] << ", " << sp.pt[1]
                          << ") analytic=" << analyticSolution
                          << " estimate=" << estimate
                          << " relErr=" << (relError * 100.0f) << "%\n";
            }
        }

        float avgRelError = (nTested > 0) ? sumRelError / nTested : 0.0f;
        std::cout << "\n========== Multi-Point Summary ==========\n";
        std::cout << "Points tested: " << nTested << "\n";
        std::cout << "Walks per point: " << nWalksPerPoint << "\n";
        std::cout << "Avg relative error: " << (avgRelError * 100.0f) << "%\n";
        std::cout << "Max relative error: " << (maxRelError * 100.0f) << "%\n";
        std::cout << "=========================================\n";

        if (maxRelError < 0.15f) {
            std::cout << "PASS: max relative error < 15%\n";
        } else {
            std::cout << "WARN: max relative error >= 15%\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
