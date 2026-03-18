#include <sycl/sycl.hpp>
#include <zombie/zombie.h>
#include "walkstate_bridge.h"
#include "phase_policy.h"
#include "task_runtime.h"
#include "task_init.h"
#include "geometry_host.h"

#include <iostream>
#include <vector>
#include <cmath>

// ============================================================
// Step 1.4 验证程序：
// 用解析单位圆几何 + host-device 迭代循环执行 WoS
//
// 问题：2D Laplace 方程 Δu = 0
// 域：单位圆 {(x,y) : x² + y² < 1}
// 边界条件：u(x,y) = x  (在 ∂Ω 上)
// 解析解：u(x,y) = x
//
// 在 (0.5, 0) 处估计，期望值 = 0.5
// ============================================================

int main() {
    try {
        sycl::queue q;
        std::cout << "Device: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << std::endl;

        // ---- PDE 定义 ----
        zombie::PDE<float, DIM> pde;
        pde.source = [](const zombie::Vector<DIM>& x) -> float {
            (void)x;
            return 0.0f; // Laplace: 无源项
        };
        pde.dirichlet = [](const zombie::Vector<DIM>& x, bool) -> float {
            return x[0]; // u = x on boundary
        };
        pde.hasReflectingBoundaryConditions = [](const zombie::Vector<DIM>&) -> bool {
            return false;
        };
        pde.robin = [](const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool) -> float {
            return 0.0f;
        };
        pde.robinCoeff = [](const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool) -> float {
            return 0.0f;
        };

        // ---- Walk 设置 ----
        const float epsilonShell = 1e-3f;
        const int maxWalkLength = 128;
        zombie::WalkSettings settings(epsilonShell, epsilonShell, maxWalkLength, false);

        PhasePolicyConfig policy;
        policy.tau_in = 0.7f;
        policy.tau_out = 0.4f;

        // ---- 解析几何：单位圆 ----
        AnalyticUnitSphereGeometry<DIM> geom;

        // ---- 采样点：(0.5, 0)，解析解 = 0.5 ----
        const int nWalks = 256;
        const float queryX = 0.5f;
        const float queryY = 0.0f;

        float sumContribution = 0.0f;
        int completedWalks = 0;

        for (int w = 0; w < nWalks; ++w) {
            // 创建单条轨迹
            std::vector<WalkStateLite> states(1);
            WalkStateLite& s = states[0];
            s = WalkStateLite{};
            s.currentPt[0] = queryX;
            s.currentPt[1] = queryY;
            s.currentPt[2] = 0.0f;
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
                static_cast<std::uint64_t>(w) + 1,
                static_cast<std::uint64_t>(w * 2 + 1));

            // Host-Device 迭代循环
            for (int step = 0; step < maxWalkLength; ++step) {
                if (s.term) break;

                // 1. Host: 更新边界距离
                host_update_boundary_distances_analytic<DIM>(states, geom);

                // 2. Host: 预处理 PDE 字段（source, dirichlet 等）
                apply_host_pde_fields(states, pde);

                // 3. Device: 执行一步
                run_task_once(q, states, 1, settings, policy);
            }

            // 如果轨迹到达边界（epsilon shell），投影并计算终止贡献
            if (s.term && s.completionCode == WALK_TERMINATED_BY_POSITION_RULE) {
                // 投影到边界并更新 dirichlet 值
                host_update_terminal_contributions_analytic<DIM>(states, geom, pde);
                // 重新计算 terminal contribution
                float contribution = s.totalSourceContribution +
                                     s.throughput * s.dirichletValue;
                sumContribution += contribution;
                completedWalks++;
            } else if (s.term && s.completionCode == WALK_TERMINATED_BY_MAX_LENGTH) {
                // 超过最大步数，仍然累积贡献
                float contribution = s.totalSourceContribution +
                                     s.totalTerminalContribution;
                sumContribution += contribution;
                completedWalks++;
            }
        }

        float estimate = (completedWalks > 0) ? sumContribution / completedWalks : 0.0f;
        float analyticSolution = queryX; // u(0.5, 0) = 0.5
        float relError = std::fabs(estimate - analyticSolution) /
                         std::fabs(analyticSolution);

        std::cout << "\n========== WoS with Analytic Geometry ==========\n";
        std::cout << "Query point: (" << queryX << ", " << queryY << ")\n";
        std::cout << "Analytic solution: " << analyticSolution << "\n";
        std::cout << "Estimated solution: " << estimate << "\n";
        std::cout << "Completed walks: " << completedWalks << " / " << nWalks << "\n";
        std::cout << "Relative error: " << (relError * 100.0f) << "%\n";
        std::cout << "================================================\n";

        if (relError < 0.05f) {
            std::cout << "PASS: relative error < 5%\n";
        } else {
            std::cout << "WARN: relative error >= 5%, may need more walks\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
