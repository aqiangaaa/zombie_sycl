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

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// ============================================================
// wos_sycl_dg.cpp
//
// Phase 2 集成验证：动态粒度调度版 WoS
//
// 与 baseline 对比：
// - baseline: 逐条 walk 串行执行
// - DG: 任务块批量执行 + 尾部回收重封装
//
// 问题：2D Laplace Δu = 0，单位圆域，u = x on ∂Ω
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
        zombie::WalkSettings settings(epsilonShell, epsilonShell, maxWalkLength, false);
        PhasePolicyConfig policy;

        // ---- 解析几何 ----
        AnalyticUnitSphereGeometry<DIM> geom;

        // ---- 测试点 ----
        struct TestPoint { float x, y, analytic; };
        std::vector<TestPoint> testPoints = {
            { 0.5f,  0.0f,  0.5f},
            {-0.5f,  0.0f, -0.5f},
            { 0.3f,  0.3f,  0.3f},
            {-0.3f, -0.3f, -0.3f},
            { 0.7f,  0.0f,  0.7f},
        };

        // ---- 调度参数 ----
        const int nWalksPerPoint = 1024;
        const int B_init = 64;
        const int B_min = 4;
        const int B_max = 64;

        GranularityPolicy granPolicy(B_init, B_min, B_max);
        TailDetector tailDetector(1, 2.0f); // 单设备，阈值 2.0

        std::cout << "\n============ WoS SYCL Dynamic Granularity ============\n";
        std::cout << "B_init=" << B_init << " B_min=" << B_min
                  << " B_max=" << B_max << "\n";
        std::cout << "Walks per point: " << nWalksPerPoint << "\n";
        std::cout << "Max walk length: " << maxWalkLength << "\n\n";

        auto t0_total = std::chrono::high_resolution_clock::now();

        for (const auto& tp : testPoints) {
            zombie::Vector<DIM> queryPt;
            queryPt[0] = tp.x;
            queryPt[1] = tp.y;
            float initDistA = geom.distToAbsorbing(queryPt);

            // ---- 创建所有 walk 的初始状态 ----
            std::vector<WalkStateLite> allStates(nWalksPerPoint);
            for (int w = 0; w < nWalksPerPoint; ++w) {
                WalkStateLite& s = allStates[w];
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
            }

            // ---- 按 B_init 分块入队 ----
            TaskQueue taskQueue;
            SchedulingStats schedStats;
            auto initialTasks = create_tasks_from_states(allStates, B_init, policy);
            for (auto& t : initialTasks) {
                taskQueue.submit(std::move(t));
            }
            schedStats.granularityHistory.push_back(B_init);

            // ---- 收集已完成轨迹的贡献 ----
            WelfordAccumulator solutionAcc;

            auto t0 = std::chrono::high_resolution_clock::now();
            bool inTailPhase = false;
            auto tailStart = t0;

            // ---- 主调度循环 ----
            while (!taskQueue.empty()) {
                WalkTask task = taskQueue.fetch();

                // Host-Device 迭代：每个任务块执行一步
                for (int step = 0; step < maxWalkLength; ++step) {
                    if (task.allFinished()) break;

                    // Host: 更新边界距离
                    host_update_boundary_distances_analytic<DIM>(task.states, geom);
                    // Host: 预处理 PDE 字段
                    apply_host_pde_fields(task.states, pde);
                    // Device: 执行一步
                    run_task_once(q, task.states, 1, settings, policy);
                }

                schedStats.totalTasksExecuted++;

                // 收集已完成轨迹的贡献
                for (auto& s : task.states) {
                    if (s.term) {
                        float contribution;
                        if (s.completionCode == WALK_TERMINATED_BY_POSITION_RULE) {
                            // 投影到边界更新 dirichlet
                            std::vector<WalkStateLite> tmp = {s};
                            host_update_terminal_contributions_analytic<DIM>(tmp, geom, pde);
                            s = tmp[0];
                            contribution = s.totalSourceContribution +
                                           s.throughput * s.dirichletValue;
                        } else {
                            contribution = s.totalSourceContribution +
                                           s.totalTerminalContribution;
                        }
                        solutionAcc.add(contribution);
                        schedStats.totalWalksCompleted++;
                    }
                }

                // 尾部检测
                bool nowTail = tailDetector.isTailPhase(taskQueue.size());
                if (nowTail && !inTailPhase) {
                    inTailPhase = true;
                    tailStart = std::chrono::high_resolution_clock::now();
                }

                // 回收未完成轨迹
                int unfinished = task.unfinishedCount();
                if (unfinished > 0) {
                    auto recycled = recycle_and_repack(task, granPolicy, nowTail);
                    for (auto& rt : recycled) {
                        taskQueue.submit(std::move(rt));
                    }
                    if (!recycled.empty()) {
                        schedStats.totalRecycles++;
                        schedStats.granularityHistory.push_back(
                            recycled[0].granularity);
                    }
                } else {
                    taskQueue.markCompleted();
                }
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            schedStats.totalExecutionTime = elapsed;
            if (inTailPhase) {
                schedStats.tailPhaseTime =
                    std::chrono::duration<double>(t1 - tailStart).count();
            }

            // ---- 输出结果 ----
            float relErr = 0.0f;
            if (std::fabs(tp.analytic) > 1e-6f) {
                relErr = std::fabs(solutionAcc.mean - tp.analytic) /
                         std::fabs(tp.analytic);
            }

            std::cout << "pt=(" << tp.x << ", " << tp.y << ")"
                      << "  analytic=" << tp.analytic
                      << "  estimate=" << solutionAcc.mean
                      << "  stdErr=" << solutionAcc.standardError()
                      << "  relErr=" << (relErr * 100.0f) << "%"
                      << "  walks=" << solutionAcc.count
                      << "  tasks=" << schedStats.totalTasksExecuted
                      << "  recycles=" << schedStats.totalRecycles
                      << "  tailFrac=" << (schedStats.tailFraction() * 100.0f) << "%"
                      << "  time=" << elapsed << "s"
                      << "\n";

            // 粒度演化
            std::cout << "  granularity: ";
            for (size_t i = 0; i < schedStats.granularityHistory.size() && i < 10; ++i) {
                if (i > 0) std::cout << " -> ";
                std::cout << schedStats.granularityHistory[i];
            }
            if (schedStats.granularityHistory.size() > 10) std::cout << " ...";
            std::cout << "\n";
        }

        auto t1_total = std::chrono::high_resolution_clock::now();
        double totalElapsed = std::chrono::duration<double>(t1_total - t0_total).count();

        std::cout << "\n============ Summary ============\n";
        std::cout << "Total time: " << totalElapsed << " s\n";
        std::cout << "Total walks: " << testPoints.size() * nWalksPerPoint << "\n";
        std::cout << "=================================\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
