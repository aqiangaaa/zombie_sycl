#include <sycl/sycl.hpp>
#include <zombie/zombie.h>
#include "walkstate_bridge.h"
#include "phase_policy.h"
#include "task_runtime.h"
#include "task_init.h"

#include <iostream>
#include <sstream>
#include <vector>

// ============================================================
// walktask_phased_sycl.cpp
//
// 这个文件是“论文核心运行时语义”的本地 SYCL 原型驱动程序。
// 它的目标不是替代真实 Zombie / WoSt 实现，而是先把下面两条主线
// 以最小可运行方式落在本地服务器上：
//
// [A] 动态粒度调度雏形
//   - WalkTask(B) 以状态数组形式存在
//   - 一次任务块执行：run_task_once(...)
//   - 执行后统计：collect_task_stats(...)
//   - tailMode / recycleDecision
//   - 提取未完成状态：extract_unfinished(...)
//   - 粒度更新：update_granularity(...)
//   - B_min / B_max / recycleThreshold
//
// [B] 阶段感知混合精度雏形
//   - 显式阶段：P0 / P1 / P2 / P3 / P4
//   - P2 固定低精度路径
//   - P1 采用 tau_in / tau_out 的最小滞回骨架
//   - 通过 prevState 保持中间区间的状态记忆
//   - 统计 P1 / P2 的低高精度路径计数
//
// [C] 已接入的真实 Zombie 对象
//   - zombie::WalkState<Real, DIM>
//   - zombie::SamplePoint<Real, DIM>
//   - zombie::WalkSettings
//
// [D] 当前仍然是 skeleton / 占位逻辑的部分
//   1. execute_one_walkstate(...) 还不是 walk_on_stars.h 的真实内核，
//      只是一个更接近 WoSt 骨架的最小版本。
//   2. P1 的 geometryDistance 目前仍然来自简化状态传播，
//      还没有真正接入真实几何查询。
//   3. P2 的“低精度路径”目前只是策略标记和统计，
//      还没有真正切换到不同数值类型。
//   4. 动态粒度当前还是单机原型级逻辑，
//      还没有接入真实队列/真实 solver 批处理框架。
// ============================================================

// ============================================================
// 根据 tailFraction 给出一个简短标签
// ============================================================
static std::string classify_tail_state(float tailFraction) {
    if (tailFraction <= 0.0f) {
        return "no-tail";
    }
    if (tailFraction >= 1.0f) {
        return "pure-tail";
    }
    return "mixed-tail";
}

// ============================================================
// 一组扫描 case 的紧凑摘要
// ============================================================
struct SweepSummary {
    int maxWalkLength = 0;
    int recycleThreshold = 0;

    int finishedCount = 0;
    int unfinishedCount = 0;
    float tailFraction = 0.0f;
    int suggestedNextGranularity = 0;

    bool hasRepacked = false;
    float repackedTailFraction = 0.0f;
    int repackedSuggestedNextGranularity = 0;

    std::string granularityHistory;
};

int main() {
    try {
        sycl::queue q;
        std::cout << "Device: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << std::endl;

        // ---------------- 运行时实验配置 ----------------
        RuntimeExperimentConfig expCfg;

        // ============================================================
        // 真实 PDE（当前先给一个最小可运行版本）
        //
        // 注意：
        // 这些回调留在 host 侧，不进入 kernel。
        // kernel 只读 apply_host_pde_fields(...) 写进去的数值字段。
        // ============================================================
        zombie::PDE<float, DIM> pde;

        // 最小 source：先返回常数 0.1
        pde.source = [](const zombie::Vector<DIM>& x) -> float {
            (void)x;
            return 0.1f;
        };

        pde.dirichlet = [](const zombie::Vector<DIM>& x, bool normalAligned) -> float {
            (void)normalAligned;
            // 先给一个最小可运行边界值
            return 1.0f + 0.01f * x[0];
        };

        // 最小“反射边界判定”：
        // 先用 x[0] < 4.0f 作为占位规则。
        // 后面可以替换成更贴近真实几何/边界定义的版本。
        pde.hasReflectingBoundaryConditions = [](const zombie::Vector<DIM>& x) -> bool {
            return x[0] < 4.0f;
        };

        pde.robin = [](const zombie::Vector<DIM>& x,
               const zombie::Vector<DIM>& n,
               bool isReflecting) -> float {
            (void)x;
            (void)n;
            return isReflecting ? 0.05f : 0.0f;
        };

        pde.robinCoeff = [](const zombie::Vector<DIM>& x,
                            const zombie::Vector<DIM>& n,
                            bool isReflecting) -> float {
            (void)x;
            (void)n;
            return isReflecting ? 1.0f : 0.0f;
        };


        // false: 只打印最后的紧凑摘要表
        // true : 每个 case 打印完整详细日志
        bool verbosePerCase = true;
        std::vector<SweepSummary> sweepSummaries;

        // 当前默认配置
        expCfg.B = 8;
        expCfg.B_min = 2;
        expCfg.B_max = 16;

        expCfg.firstLaunchSteps = 3;
        expCfg.secondLaunchSteps = 1;

        expCfg.recycleThreshold = 8;
        expCfg.useSamplePointInit = true;

        expCfg.walkSettings = zombie::WalkSettings(1e-3f, 1e-3f, 2, false);

        expCfg.phasePolicy.tau_in = 0.7f;
        expCfg.phasePolicy.tau_out = 0.4f;

        std::vector<int> maxWalkLengthSweep = {2, 3, 4};
        std::vector<int> recycleThresholdSweep = {2, 4, 8};

        auto run_one_experiment = [&](int maxWalkLength, int recycleThreshold) {
            expCfg.walkSettings.maxWalkLength = maxWalkLength;
            expCfg.recycleThreshold = recycleThreshold;

            if (verbosePerCase) {
                std::cout << "\n==============================\n";
                std::cout << "sweep case: "
                          << "maxWalkLength=" << maxWalkLength
                          << ", recycleThreshold=" << recycleThreshold
                          << std::endl;
                std::cout << "==============================\n";
            }

            int currentTaskGranularity = expCfg.B;
            int nextTaskGranularity = 0;

            std::vector<int> granularityHistory;
            granularityHistory.push_back(currentTaskGranularity);

            SweepSummary summary;
            summary.maxWalkLength = maxWalkLength;
            summary.recycleThreshold = recycleThreshold;

            // ---------------- 初始化主机侧真实状态 ----------------
            std::vector<zombie::WalkState<Real, DIM>> hostStates;
            std::vector<zombie::SamplePoint<Real, DIM>> samplePoints;

            if (!expCfg.useSamplePointInit) {
                TaskInitConfig initCfg;
                hostStates = make_initial_host_walkstates(expCfg.B, initCfg);
            } else {
                samplePoints.reserve(expCfg.B);

                for (int i = 0; i < expCfg.B; ++i) {
                    zombie::Vector<DIM> pt = zombie::Vector<DIM>::Zero();
                    pt[0] = float(i);
                    pt[1] = float(i + 100);

                    zombie::Vector<DIM> normal = zombie::Vector<DIM>::Zero();
                    normal[0] = 0.0f;
                    normal[1] = 1.0f;

                    float distA = 0.5f;
                    float distR = 0.5f;

                    if (i < expCfg.B / 2) {
                        // 前半部分：离吸收边界更远
                        distA = 0.8f;
                        distR = 0.3f;
                    } else {
                        // 后半部分：离吸收边界更近
                        distA = 0.3f;
                        distR = 0.8f;
                    }

                    samplePoints.emplace_back(
                        pt,
                        normal,
                        zombie::SampleType::InDomain,
                        zombie::EstimationQuantity::Solution,
                        1.0f,   // pdf
                        distA,  // distToAbsorbingBoundary
                        distR   // distToReflectingBoundary
                    );
                }

                hostStates = make_initial_host_walkstates_from_sample_points(samplePoints);
            }

            // ---------------- 打包成第一个任务块 ----------------
            std::vector<WalkStateLite> walkTaskStates;
            if (!expCfg.useSamplePointInit) {
                walkTaskStates = make_initial_walktask_states(hostStates);
            } else {
                walkTaskStates = make_initial_walktask_states(hostStates, samplePoints);
            }

            // ============================================================
            // 在 host 上用真实 PDE 预处理 device 需要的字段
            // ============================================================
            apply_host_pde_fields(walkTaskStates, pde);

            // ---------------- 第一次执行 ----------------
            run_task_once(
                q,
                walkTaskStates,
                expCfg.firstLaunchSteps,
                expCfg.walkSettings,
                expCfg.phasePolicy
            );

            for (int i = 0; i < expCfg.B; ++i) {
                unpackState(walkTaskStates[i], hostStates[i]);
            }

            // ---------------- 收集第一轮统计 ----------------
            TaskExecutionStats stats = collect_task_stats(
                walkTaskStates,
                currentTaskGranularity,
                expCfg.recycleThreshold,
                expCfg.B_min,
                expCfg.B_max
            );

            nextTaskGranularity = stats.newGranularity;
            granularityHistory.push_back(nextTaskGranularity);

            summary.finishedCount = stats.finishedCount;
            summary.unfinishedCount = stats.unfinishedCount;
            summary.tailFraction = stats.tailFraction;
            summary.suggestedNextGranularity = stats.newGranularity;

            if (verbosePerCase) {
                std::cout << "currentTaskGranularity(before first launch)="
                          << currentTaskGranularity << std::endl;

                std::cout << "finishedCount=" << stats.finishedCount
                          << ", unfinishedCount=" << stats.unfinishedCount
                          << std::endl;

                std::cout << "tailMode=" << stats.tailMode << std::endl;
                std::cout << "recycleDecision=" << stats.recycleDecision << std::endl;
                std::cout << "recycleThreshold=" << expCfg.recycleThreshold << std::endl;
                std::cout << "B_min=" << expCfg.B_min
                          << ", B_max=" << expCfg.B_max
                          << std::endl;
                std::cout << "useSamplePointInit=" << expCfg.useSamplePointInit << std::endl;
                std::cout << "walkSettings.maxWalkLength="
                          << expCfg.walkSettings.maxWalkLength
                          << std::endl;
                std::cout << "phasePolicy.tau_in=" << expCfg.phasePolicy.tau_in
                          << ", tau_out=" << expCfg.phasePolicy.tau_out
                          << std::endl;

                std::cout << "phaseCounts: "
                          << "P0=" << stats.p0Count << ", "
                          << "P1=" << stats.p1Count << ", "
                          << "P2=" << stats.p2Count << ", "
                          << "P3=" << stats.p3Count << ", "
                          << "P4=" << stats.p4Count
                          << std::endl;

                std::cout << "P2PrecisionCounts: "
                          << "low=" << stats.p2LowPrecisionCount << ", "
                          << "high=" << stats.p2HighPrecisionCount
                          << std::endl;

                std::cout << "P1PrecisionCounts: "
                          << "low=" << stats.p1LowPrecisionCount << ", "
                          << "high=" << stats.p1HighPrecisionCount
                          << std::endl;

                std::cout << "taskTotals: "
                        << "source=" << stats.totalSourceContribution << ", "
                        << "terminal=" << stats.totalTerminalContribution << ", "
                        << "reflectContrib=" << stats.totalReflectingBoundaryContribution
                        << std::endl;
                
                std::cout << "completionCounts: "
                        << "maxLength=" << stats.terminatedByMaxLengthCount << ", "
                        << "positionRule=" << stats.terminatedByPositionRuleCount
                        << std::endl;

                std::cout << "taskAverages: "
                          << "throughput=" << stats.avgThroughput << ", "
                          << "walkLength=" << stats.avgWalkLength << ", "
                          << "tailFraction=" << stats.tailFraction
                          << std::endl;

                std::cout << "summary: "
                          << "originalTaskState=" << classify_tail_state(stats.tailFraction)
                          << ", suggestedNextGranularity=" << stats.newGranularity
                          << std::endl;

                print_task(walkTaskStates, "original task after first execution");
            }
                        // ---------------- 回收 / 重封装 ----------------
            std::vector<WalkStateLite> repackedWalkTaskStates;
            if (stats.recycleDecision) {
                repackedWalkTaskStates = extract_unfinished(walkTaskStates);
            }

            if (!repackedWalkTaskStates.empty()) {
                run_task_once(
                    q,
                    repackedWalkTaskStates,
                    expCfg.secondLaunchSteps,
                    expCfg.walkSettings,
                    expCfg.phasePolicy
                );

                TaskExecutionStats repackedStats = collect_task_stats(
                    repackedWalkTaskStates,
                    nextTaskGranularity,
                    expCfg.recycleThreshold,
                    expCfg.B_min,
                    expCfg.B_max
                );

                summary.hasRepacked = true;
                summary.repackedTailFraction = repackedStats.tailFraction;
                summary.repackedSuggestedNextGranularity = repackedStats.newGranularity;

                granularityHistory.push_back(repackedStats.newGranularity);

                if (verbosePerCase) {
                    std::cout << "repackedWalkTaskStates.size()="
                              << repackedWalkTaskStates.size()
                              << std::endl;

                    std::cout << "currentTaskGranularity(after first launch)="
                              << currentTaskGranularity << std::endl;
                    std::cout << "nextTaskGranularity(after collect_task_stats)="
                              << nextTaskGranularity << std::endl;
                    std::cout << "granularity update: "
                              << currentTaskGranularity
                              << " -> "
                              << nextTaskGranularity
                              << std::endl;

                    std::cout << "[repacked] finishedCount=" << repackedStats.finishedCount
                              << ", unfinishedCount=" << repackedStats.unfinishedCount
                              << std::endl;

                    std::cout << "[repacked] tailMode=" << repackedStats.tailMode << std::endl;
                    std::cout << "[repacked] recycleDecision=" << repackedStats.recycleDecision << std::endl;
                    std::cout << "[repacked] nextTaskGranularity="
                              << repackedStats.newGranularity
                              << std::endl;

                    std::cout << "[repacked] phaseCounts: "
                              << "P0=" << repackedStats.p0Count << ", "
                              << "P1=" << repackedStats.p1Count << ", "
                              << "P2=" << repackedStats.p2Count << ", "
                              << "P3=" << repackedStats.p3Count << ", "
                              << "P4=" << repackedStats.p4Count
                              << std::endl;

                    std::cout << "[repacked] P2PrecisionCounts: "
                              << "low=" << repackedStats.p2LowPrecisionCount << ", "
                              << "high=" << repackedStats.p2HighPrecisionCount
                              << std::endl;

                    std::cout << "[repacked] P1PrecisionCounts: "
                              << "low=" << repackedStats.p1LowPrecisionCount << ", "
                              << "high=" << repackedStats.p1HighPrecisionCount
                              << std::endl;

                    std::cout << "[repacked] taskTotals: "
                            << "source=" << repackedStats.totalSourceContribution << ", "
                            << "terminal=" << repackedStats.totalTerminalContribution << ", "
                            << "reflectContrib=" << repackedStats.totalReflectingBoundaryContribution
                            << std::endl;

                    std::cout << "[repacked] completionCounts: "
                            << "maxLength=" << repackedStats.terminatedByMaxLengthCount << ", "
                            << "positionRule=" << repackedStats.terminatedByPositionRuleCount
                            << std::endl;

                    std::cout << "[repacked] taskAverages: "
                              << "throughput=" << repackedStats.avgThroughput << ", "
                              << "walkLength=" << repackedStats.avgWalkLength << ", "
                              << "tailFraction=" << repackedStats.tailFraction
                              << std::endl;

                    std::cout << "[repacked] summary: "
                              << "taskState=" << classify_tail_state(repackedStats.tailFraction)
                              << ", suggestedNextGranularity=" << repackedStats.newGranularity
                              << std::endl;

                    std::cout << "[repacked] granularity evolution: "
                              << nextTaskGranularity
                              << " -> "
                              << repackedStats.newGranularity;

                    if (nextTaskGranularity == repackedStats.newGranularity) {
                        std::cout << " (stable)";
                    } else {
                        std::cout << " (changing)";
                    }
                    std::cout << std::endl;

                    print_task(repackedWalkTaskStates, "repacked walk task after second execution");
                }
            } else {
                if (verbosePerCase) {
                    std::cout << "repackedWalkTaskStates.size()=0" << std::endl;
                    std::cout << "currentTaskGranularity(after first launch)="
                              << currentTaskGranularity << std::endl;
                    std::cout << "nextTaskGranularity(after collect_task_stats)="
                              << nextTaskGranularity << std::endl;
                    std::cout << "granularity update: "
                              << currentTaskGranularity
                              << " -> 0 (no repack)" << std::endl;
                    std::cout << "repackedWalkTaskStates is empty, no second execution." << std::endl;
                }
            }

            {
                std::ostringstream oss;
                for (size_t i = 0; i < granularityHistory.size(); ++i) {
                    oss << granularityHistory[i];
                    if (i + 1 < granularityHistory.size()) {
                        oss << " -> ";
                    }
                }
                summary.granularityHistory = oss.str();
            }

            sweepSummaries.push_back(summary);

            if (verbosePerCase) {
                std::cout << "granularityHistory: "
                          << summary.granularityHistory
                          << std::endl;
            }
        };

        for (int maxWalkLength : maxWalkLengthSweep) {
            for (int recycleThreshold : recycleThresholdSweep) {
                run_one_experiment(maxWalkLength, recycleThreshold);
            }
        }

        std::cout << "\n========================================\n";
        std::cout << "compact sweep summary\n";
        std::cout << "========================================\n";

        for (const auto& s : sweepSummaries) {
            std::cout
                << "maxWalkLength=" << s.maxWalkLength
                << ", recycleThreshold=" << s.recycleThreshold
                << " | finished=" << s.finishedCount
                << ", unfinished=" << s.unfinishedCount
                << ", tailFraction=" << s.tailFraction
                << ", nextB=" << s.suggestedNextGranularity
                << ", repacked=" << (s.hasRepacked ? 1 : 0);
            

            if (s.hasRepacked) {
                std::cout
                    << ", repackedTailFraction=" << s.repackedTailFraction
                    << ", repackedNextB=" << s.repackedSuggestedNextGranularity;
            }

            std::cout
                << ", history=" << s.granularityHistory
                << std::endl;
            
            if (verbosePerCase) {
                std::cout << "----------------------------------------" << std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "SYCL error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}