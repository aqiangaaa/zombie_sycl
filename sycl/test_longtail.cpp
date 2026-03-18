#include <sycl/sycl.hpp>
#include <zombie/zombie.h>
#include "walkstate_bridge.h"
#include "phase_policy.h"
#include "task_runtime.h"
#include "task_init.h"
#include "geometry_host.h"
#include "batch_solver.h"
#include "task_queue.h"
#include "statistics.h"
#include "metrics.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <numeric>

// ============================================================
// test_longtail.cpp
//
// Phase 7: 长尾场景验证
//
// 几何：窄矩形域 [-1,1] × [-0.05, 0.05]
// 域内点到边界的距离最大 0.05（窄方向），
// 但靠近中心的点需要大量步数才能到达左右边界。
// 靠近左右端的点很快终止。
// → walk 步数分布呈现明显长尾。
//
// 对比 Baseline vs DG 的尾部比例。
// ============================================================

// 混合距离几何：不同采样点到边界的距离差异极大
// 模拟"部分点靠近边界（快速终止）+ 部分点远离边界（长尾）"
struct MixedDistanceGeometry {
    // 单位圆域，但不同点的初始距离差异大
    float distToAbsorbing(const zombie::Vector<DIM>& pt) const {
        return std::max(0.0f, 1.0f - static_cast<float>(pt.norm()));
    }

    bool projectToAbsorbing(zombie::Vector<DIM>& pt,
                            zombie::Vector<DIM>& normal,
                            float& dist) const {
        float norm = static_cast<float>(pt.norm());
        if (norm < 1e-8f) {
            normal = zombie::Vector<DIM>::Zero();
            normal[0] = 1.0f;
            pt = normal;
            dist = 1.0f;
            return true;
        }
        normal = pt.normalized();
        pt = normal;
        dist = 1.0f - norm;
        return true;
    }
};

// 单条 walk 执行，返回步数
int run_single_walk_steps(
    sycl::queue& q,
    const zombie::Vector<DIM>& queryPt,
    const MixedDistanceGeometry& geom,
    const zombie::PDE<float, DIM>& pde,
    const zombie::WalkSettings& settings,
    const PhasePolicyConfig& policy,
    std::uint64_t seed
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
    float initDist = geom.distToAbsorbing(queryPt);
    s.distToAbsorbingBoundary = initDist;
    s.geometryDistance = initDist;
    s.rng = pcg32_seed(seed, seed * 2 + 1);

    for (int step = 0; step < settings.maxWalkLength; ++step) {
        if (s.term) break;
        host_update_distances_generic<DIM>(states, geom);
        apply_host_pde_fields(states, pde);
        run_task_once(q, states, 1, settings, policy);
    }
    return s.walkLength;
}

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

        const int maxWalkLength = 64;
        zombie::WalkSettings settings(1e-5f, 1e-5f, maxWalkLength, false);
        PhasePolicyConfig policy;
        MixedDistanceGeometry geom;

        // ---- 采样点：混合距离，制造长尾 ----
        // 靠近边界的点（dist ≈ 0.01-0.05）快速终止
        // 远离边界的点（dist ≈ 0.5-0.99）需要大量步数
        const int nWalksPerPoint = 128;
        std::vector<float> xPositions = {0.99f, 0.95f, 0.9f, 0.7f,
                                          0.5f,  0.3f,  0.1f, 0.01f};

        std::cout << "============ Long-Tail Walk Length Analysis ============\n";
        std::cout << "Geometry: unit disk, mixed-distance sample points\n";
        std::cout << "Epsilon shell: " << settings.epsilonShellForAbsorbingBoundary << "\n";
        std::cout << "Max walk length: " << maxWalkLength << "\n";
        std::cout << "Walks per point: " << nWalksPerPoint << "\n\n";

        // ---- 收集所有 walk 的步数 ----
        std::vector<int> allWalkLengths;

        for (int pi = 0; pi < static_cast<int>(xPositions.size()); ++pi) {
            zombie::Vector<DIM> pt;
            pt[0] = xPositions[pi];
            pt[1] = 0.0f;
            float distA = geom.distToAbsorbing(pt);

            std::vector<int> lengths;
            for (int w = 0; w < nWalksPerPoint; ++w) {
                int len = run_single_walk_steps(
                    q, pt, geom, pde, settings, policy,
                    static_cast<std::uint64_t>(pi * nWalksPerPoint + w) + 1);
                lengths.push_back(len);
                allWalkLengths.push_back(len);
            }

            std::sort(lengths.begin(), lengths.end());
            float mean = std::accumulate(lengths.begin(), lengths.end(), 0.0f) / lengths.size();
            int p50 = lengths[lengths.size() / 2];
            int p90 = lengths[static_cast<int>(lengths.size() * 0.9)];
            int p99 = lengths[static_cast<int>(lengths.size() * 0.99)];
            int maxLen = lengths.back();

            std::cout << "  x=" << std::setw(5) << xPositions[pi]
                      << "  distA=" << std::setw(6) << std::fixed << std::setprecision(4) << distA
                      << "  mean=" << std::setw(7) << std::setprecision(1) << mean
                      << "  P50=" << std::setw(5) << p50
                      << "  P90=" << std::setw(5) << p90
                      << "  P99=" << std::setw(5) << p99
                      << "  max=" << std::setw(5) << maxLen
                      << "\n";
        }

        // ---- 全局统计 ----
        std::sort(allWalkLengths.begin(), allWalkLengths.end());
        int totalWalks = static_cast<int>(allWalkLengths.size());
        float globalMean = std::accumulate(allWalkLengths.begin(), allWalkLengths.end(), 0.0f) / totalWalks;
        int gP50 = allWalkLengths[totalWalks / 2];
        int gP90 = allWalkLengths[static_cast<int>(totalWalks * 0.9)];
        int gP99 = allWalkLengths[static_cast<int>(totalWalks * 0.99)];
        int gMax = allWalkLengths.back();

        std::cout << "\n  GLOBAL: mean=" << std::setprecision(1) << globalMean
                  << "  P50=" << gP50
                  << "  P90=" << gP90
                  << "  P99=" << gP99
                  << "  max=" << gMax
                  << "  P99/P50=" << std::setprecision(1) << (gP50 > 0 ? (float)gP99/gP50 : 0.0f)
                  << "\n";

        if ((float)gP99 / std::max(1, gP50) > 3.0f) {
            std::cout << "\nPASS: Long-tail distribution detected (P99/P50 > 3)\n";
        } else {
            std::cout << "\nINFO: Moderate tail (P99/P50 <= 3)\n";
        }

        // ============================================================
        // Step 7.2-7.3: DG vs Baseline 对比
        // 混合不同距离的采样点在同一批 walk 中，制造任务块内长尾
        // ============================================================
        std::cout << "\n============ DG vs Baseline Comparison ============\n";

        // 混合采样点：一半靠近边界（快终止），一半在中心（慢终止）
        const int nWalksDG = 512;
        std::vector<WalkStateLite> mixedStates;
        for (int w = 0; w < nWalksDG; ++w) {
            WalkStateLite s = WalkStateLite{};
            zombie::Vector<DIM> pt = zombie::Vector<DIM>::Zero();
            if (w < nWalksDG / 2) {
                // 靠近边界：x=0.99, distA≈0.01
                pt[0] = 0.99f;
            } else {
                // 远离边界：x=0.01, distA≈0.99
                pt[0] = 0.01f;
            }
            pt[1] = 0.0f;
            float distA = geom.distToAbsorbing(pt);
            for (int d = 0; d < DIM; ++d) s.currentPt[d] = pt[d];
            for (int d = DIM; d < MAX_DIM; ++d) s.currentPt[d] = 0.0f;
            for (int d = 0; d < MAX_DIM; ++d) {
                s.currentNormal[d] = 0.0f;
                s.prevDirection[d] = 0.0f;
                s.normal[d] = 0.0f;
            }
            s.throughput = 1.0f;
            s.distToAbsorbingBoundary = distA;
            s.geometryDistance = distA;
            s.rng = pcg32_seed(
                static_cast<std::uint64_t>(w) + 1,
                static_cast<std::uint64_t>(w * 2 + 1));
            mixedStates.push_back(s);
        }

        // --- Baseline: 批量求解，无回收 ---
        {
            auto states = mixedStates; // copy
            auto t0 = std::chrono::high_resolution_clock::now();

            int stepsDone = 0;
            for (int step = 0; step < maxWalkLength; ++step) {
                bool allDone = true;
                for (const auto& s : states) { if (!s.term) { allDone = false; break; } }
                if (allDone) break;
                host_update_distances_generic<DIM>(states, geom);
                apply_host_pde_fields(states, pde);
                run_task_once(q, states, 1, settings, policy);
                stepsDone++;
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();

            int finished = 0;
            std::vector<int> lens;
            for (const auto& s : states) {
                if (s.term) finished++;
                lens.push_back(s.walkLength);
            }
            std::sort(lens.begin(), lens.end());

            std::cout << "[Baseline] time=" << std::setprecision(3) << elapsed << "s"
                      << "  finished=" << finished << "/" << nWalksDG
                      << "  steps=" << stepsDone
                      << "  P50=" << lens[lens.size()/2]
                      << "  P99=" << lens[static_cast<int>(lens.size()*0.99)]
                      << "  max=" << lens.back()
                      << "\n";
        }

        // --- DG: 任务块调度 + 回收 ---
        // 关键：每个任务块只执行 stepsPerLaunch 步，然后检查回收
        {
            GranularityPolicy granPolicy(64, 4, 64);
            TailDetector tailDetector(1, 2.0f);
            const int stepsPerLaunch = 8; // 每次只推进 8 步

            auto allStates = mixedStates; // copy

            TaskQueue taskQueue;
            auto initialTasks = create_tasks_from_states(allStates, granPolicy.B_init, policy);
            for (auto& t : initialTasks) taskQueue.submit(std::move(t));

            int totalRecycles = 0;
            int totalTasksExec = 0;
            WelfordAccumulator solAcc;
            std::vector<int> granHistory;
            granHistory.push_back(granPolicy.B_init);

            auto t0 = std::chrono::high_resolution_clock::now();

            while (!taskQueue.empty()) {
                WalkTask task = taskQueue.fetch();

                // 执行任务块：只推进 stepsPerLaunch 步，而非跑到终止
                for (int step = 0; step < stepsPerLaunch; ++step) {
                    if (task.allFinished()) break;
                    host_update_distances_generic<DIM>(task.states, geom);
                    apply_host_pde_fields(task.states, pde);
                    run_task_once(q, task.states, 1, settings, policy);
                }
                totalTasksExec++;

                // 收集已完成
                for (auto& s : task.states) {
                    if (s.term) {
                        std::vector<WalkStateLite> tmp = {s};
                        host_update_terminal_generic<DIM>(tmp, geom, pde);
                        s = tmp[0];
                        float c = s.totalSourceContribution + s.throughput * s.dirichletValue;
                        solAcc.add(c);
                    }
                }

                // 尾部检测 + 回收
                bool nowTail = tailDetector.isTailPhase(taskQueue.size());
                int unfinished = task.unfinishedCount();
                if (unfinished > 0) {
                    auto recycled = recycle_and_repack(task, granPolicy, nowTail);
                    for (auto& rt : recycled) {
                        taskQueue.submit(std::move(rt));
                    }
                    if (!recycled.empty()) {
                        totalRecycles++;
                        granHistory.push_back(recycled[0].granularity);
                    }
                }
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();

            std::cout << "[DG]       time=" << std::setprecision(3) << elapsed << "s"
                      << "  finished=" << solAcc.count << "/" << nWalksDG
                      << "  tasks=" << totalTasksExec
                      << "  recycles=" << totalRecycles
                      << "  estimate=" << std::setprecision(4) << solAcc.mean
                      << "\n";

            // 粒度演化
            std::cout << "  granularity: ";
            for (size_t i = 0; i < granHistory.size() && i < 10; ++i) {
                if (i > 0) std::cout << " -> ";
                std::cout << granHistory[i];
            }
            if (granHistory.size() > 10) std::cout << " ...";
            std::cout << "\n";
        }

        std::cout << "\n============ Done ============\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
