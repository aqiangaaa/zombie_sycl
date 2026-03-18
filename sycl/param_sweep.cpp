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
#include "experiment_config.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>

// ============================================================
// param_sweep.cpp
//
// Phase 9: 参数扫描 + 开销分解 + CSV 输出
//
// 9.1: B 敏感性扫描
// 9.2: τ_in/τ_out 敏感性扫描
// 9.3: 开销分解（host vs device）
// 9.4: 全部结果输出为 CSV
//
// 问题：2D Laplace Δu=0，单位圆域，u=x
// ============================================================

struct SweepResult {
    std::string label;
    int B;
    float tau_in, tau_out;
    bool enableRecycle;
    bool enableMP;
    float estimate;
    float stdErr;
    float relErr;
    double totalTime;
    double hostTime;
    double deviceTime;
    int totalSteps;
    float fp64Ratio;
    int recycles;
    float avgWalkLen;
};

// 单配置运行：使用长尾混合距离场景
SweepResult run_sweep_config(
    sycl::queue& q,
    const std::string& label,
    int B, float tau_in, float tau_out,
    bool enableRecycle, bool enableMP,
    int nWalks, int maxWalkLength, int stepsPerLaunch
) {
    // PDE
    zombie::PDE<float, DIM> pde;
    pde.source = [](const zombie::Vector<DIM>&) -> float { return 0.0f; };
    pde.dirichlet = [](const zombie::Vector<DIM>& x, bool) -> float { return x[0]; };
    pde.hasReflectingBoundaryConditions = [](const zombie::Vector<DIM>&) { return false; };
    pde.robin = [](const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool) -> float { return 0.0f; };
    pde.robinCoeff = [](const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool) -> float { return 0.0f; };

    zombie::WalkSettings settings(1e-5f, 1e-5f, maxWalkLength, false);
    PhasePolicyConfig policy;
    policy.tau_in = tau_in;
    policy.tau_out = tau_out;
    AnalyticUnitSphereGeometry<DIM> geom;

    // 混合距离采样点：一半近边界，一半远离
    zombie::Vector<DIM> queryPt = zombie::Vector<DIM>::Zero();
    queryPt[0] = 0.3f; // analytic = 0.3
    float initDistA = geom.distToAbsorbing(queryPt);

    SweepResult res;
    res.label = label;
    res.B = B;
    res.tau_in = tau_in;
    res.tau_out = tau_out;
    res.enableRecycle = enableRecycle;
    res.enableMP = enableMP;
    res.recycles = 0;

    // 初始化所有 walk
    auto allStates = batch_init_walks(queryPt, initDistA, nWalks);

    // P1 精度统计
    int64_t p1Low = 0, p1High = 0, p2Total = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    double hostTime = 0.0, deviceTime = 0.0;

    if (!enableRecycle) {
        // 无回收：批量执行
        int steps = 0;
        for (int step = 0; step < maxWalkLength; ++step) {
            bool allDone = true;
            for (const auto& s : allStates) { if (!s.term) { allDone = false; break; } }
            if (allDone) break;

            auto th0 = std::chrono::high_resolution_clock::now();
            host_update_distances_generic<DIM>(allStates, geom);
            apply_host_pde_fields(allStates, pde);
            auto th1 = std::chrono::high_resolution_clock::now();
            hostTime += std::chrono::duration<double>(th1 - th0).count();

            // 精度统计
            for (const auto& s : allStates) {
                if (s.term) continue;
                bool low = enableMP &&
                    use_low_precision_for_p1(s.distToAbsorbingBoundary,
                                              s.p1LowPrecisionState, policy);
                if (low) p1Low++; else p1High++;
                p2Total++;
            }

            auto td0 = std::chrono::high_resolution_clock::now();
            run_task_once_mp(q, allStates, 1, settings, policy, enableMP);
            auto td1 = std::chrono::high_resolution_clock::now();
            deviceTime += std::chrono::duration<double>(td1 - td0).count();
            steps++;
        }
        res.totalSteps = steps;

        // 终止投影
        host_update_terminal_generic<DIM>(allStates, geom, pde);

    } else {
        // 有回收：任务块调度
        GranularityPolicy granPolicy(B, 4, B);
        TailDetector tailDetector(1, 2.0f);

        TaskQueue taskQueue;
        auto initialTasks = create_tasks_from_states(allStates, B, policy);
        for (auto& t : initialTasks) taskQueue.submit(std::move(t));

        int totalSteps = 0;

        while (!taskQueue.empty()) {
            WalkTask task = taskQueue.fetch();

            for (int step = 0; step < stepsPerLaunch; ++step) {
                if (task.allFinished()) break;

                auto th0 = std::chrono::high_resolution_clock::now();
                host_update_distances_generic<DIM>(task.states, geom);
                apply_host_pde_fields(task.states, pde);
                auto th1 = std::chrono::high_resolution_clock::now();
                hostTime += std::chrono::duration<double>(th1 - th0).count();

                for (const auto& s : task.states) {
                    if (s.term) continue;
                    bool low = enableMP &&
                        use_low_precision_for_p1(s.distToAbsorbingBoundary,
                                                  s.p1LowPrecisionState, policy);
                    if (low) p1Low++; else p1High++;
                    p2Total++;
                }

                auto td0 = std::chrono::high_resolution_clock::now();
                run_task_once_mp(q, task.states, 1, settings, policy, enableMP);
                auto td1 = std::chrono::high_resolution_clock::now();
                deviceTime += std::chrono::duration<double>(td1 - td0).count();
                totalSteps++;
            }

            // 终止投影
            for (auto& s : task.states) {
                if (s.term) {
                    std::vector<WalkStateLite> tmp = {s};
                    host_update_terminal_generic<DIM>(tmp, geom, pde);
                    s = tmp[0];
                }
            }

            // 回收
            bool nowTail = tailDetector.isTailPhase(taskQueue.size());
            int unfinished = task.unfinishedCount();
            if (unfinished > 0) {
                auto recycled = recycle_and_repack(task, granPolicy, nowTail);
                for (auto& rt : recycled) taskQueue.submit(std::move(rt));
                if (!recycled.empty()) res.recycles++;
            }

            // 把已完成的写回 allStates（用于最终统计）
            for (const auto& s : task.states) {
                if (s.term) allStates.push_back(s);
            }
        }
        res.totalSteps = totalSteps;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.totalTime = std::chrono::duration<double>(t1 - t0).count();
    res.hostTime = hostTime;
    res.deviceTime = deviceTime;

    // 统计
    WelfordAccumulator acc;
    WelfordAccumulator lenAcc;
    for (const auto& s : allStates) {
        if (!s.term) continue;
        float c = s.totalSourceContribution + s.throughput * s.dirichletValue;
        acc.add(c);
        lenAcc.add(static_cast<float>(s.walkLength));
    }

    res.estimate = acc.mean;
    res.stdErr = acc.standardError();
    res.relErr = std::fabs(acc.mean - 0.3f) / 0.3f; // analytic = 0.3
    res.avgWalkLen = lenAcc.mean;

    // FP64 覆盖
    int64_t totalOps = p1Low + p1High + p2Total;
    int64_t fp64Ops = p1High + (enableMP ? 0 : p2Total);
    res.fp64Ratio = (totalOps > 0) ? static_cast<float>(fp64Ops) / totalOps : 1.0f;

    return res;
}

int main() {
    try {
        sycl::queue q;
        std::cout << "Device: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << "\n\n";

        const int nWalks = 512;
        const int maxWalkLength = 64;
        const int stepsPerLaunch = 8;

        std::vector<SweepResult> allResults;

        // ============================================================
        // 9.1: B 敏感性扫描（全 FP64，有回收）
        // ============================================================
        std::cout << "=== 9.1: B Sensitivity Sweep ===\n";
        std::vector<int> bValues = {4, 8, 16, 32, 64, 128, 256};
        for (int b : bValues) {
            auto r = run_sweep_config(q, "DG_B" + std::to_string(b),
                                       b, 0.5f, 0.2f, true, false,
                                       nWalks, maxWalkLength, stepsPerLaunch);
            allResults.push_back(r);
            std::cout << "  B=" << std::setw(4) << b
                      << "  est=" << std::fixed << std::setprecision(4) << r.estimate
                      << "  relErr=" << std::setprecision(1) << (r.relErr*100) << "%"
                      << "  time=" << std::setprecision(3) << r.totalTime << "s"
                      << "  recycles=" << r.recycles
                      << "\n";
        }

        // ============================================================
        // 9.2: τ_in/τ_out 敏感性扫描（固定 B=64，有 MP）
        // ============================================================
        std::cout << "\n=== 9.2: tau_in/tau_out Sensitivity Sweep ===\n";
        struct TauPair { float in, out; };
        std::vector<TauPair> tauValues = {
            {0.3f, 0.1f}, {0.5f, 0.2f}, {0.7f, 0.4f},
            {0.9f, 0.6f}, {0.95f, 0.8f}
        };
        for (const auto& tp : tauValues) {
            std::ostringstream lbl;
            lbl << "MP_" << tp.in << "/" << tp.out;
            auto r = run_sweep_config(q, lbl.str(),
                                       64, tp.in, tp.out, false, true,
                                       nWalks, maxWalkLength, stepsPerLaunch);
            allResults.push_back(r);
            std::cout << "  tau=(" << std::setprecision(2) << tp.in << "," << tp.out << ")"
                      << "  est=" << std::setprecision(4) << r.estimate
                      << "  relErr=" << std::setprecision(1) << (r.relErr*100) << "%"
                      << "  fp64=" << std::setprecision(1) << (r.fp64Ratio*100) << "%"
                      << "  time=" << std::setprecision(3) << r.totalTime << "s"
                      << "\n";
        }

        // ============================================================
        // 9.3: 四组对照 + 开销分解
        // ============================================================
        std::cout << "\n=== 9.3: Four-Group Comparison + Cost Breakdown ===\n";
        struct GroupCfg { std::string name; bool recycle; bool mp; };
        std::vector<GroupCfg> groups = {
            {"Baseline", false, false},
            {"DG",       true,  false},
            {"MP",       false, true},
            {"DG+MP",    true,  true},
        };
        for (const auto& g : groups) {
            auto r = run_sweep_config(q, g.name,
                                       64, 0.5f, 0.2f, g.recycle, g.mp,
                                       nWalks, maxWalkLength, stepsPerLaunch);
            allResults.push_back(r);
            std::cout << "  " << std::setw(8) << g.name
                      << "  est=" << std::setprecision(4) << r.estimate
                      << "  relErr=" << std::setprecision(1) << (r.relErr*100) << "%"
                      << "  fp64=" << std::setprecision(1) << (r.fp64Ratio*100) << "%"
                      << "  total=" << std::setprecision(3) << r.totalTime << "s"
                      << "  host=" << r.hostTime << "s"
                      << "  device=" << r.deviceTime << "s"
                      << "  recycles=" << r.recycles
                      << "\n";
        }

        // ============================================================
        // 9.4: CSV 输出
        // ============================================================
        std::string csvFile = "../sycl/experiment_results.csv";
        std::ofstream csv(csvFile);
        csv << "label,B,tau_in,tau_out,recycle,mp,estimate,stdErr,relErr,"
            << "totalTime,hostTime,deviceTime,steps,fp64Ratio,recycles,avgWalkLen\n";
        for (const auto& r : allResults) {
            csv << r.label << ","
                << r.B << ","
                << r.tau_in << "," << r.tau_out << ","
                << r.enableRecycle << "," << r.enableMP << ","
                << r.estimate << "," << r.stdErr << "," << r.relErr << ","
                << r.totalTime << "," << r.hostTime << "," << r.deviceTime << ","
                << r.totalSteps << "," << r.fp64Ratio << ","
                << r.recycles << "," << r.avgWalkLen << "\n";
        }
        csv.close();
        std::cout << "\nCSV written to: " << csvFile << "\n";
        std::cout << "Total configs: " << allResults.size() << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
