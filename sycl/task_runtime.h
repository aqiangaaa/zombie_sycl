#pragma once

#include <sycl/sycl.hpp>
#include "walkstate_bridge.h"
#include "phase_policy.h"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

// ============================================================
// 一次任务块执行后的统计信息
// ============================================================
struct TaskExecutionStats {
    int finishedCount = 0;
    int unfinishedCount = 0;

    int oldGranularity = 0;
    int newGranularity = 0;
    int repackedSize = 0;

    bool tailMode = false;
    bool recycleDecision = false;

    int p0Count = 0;
    int p1Count = 0;
    int p2Count = 0;
    int p3Count = 0;
    int p4Count = 0;

    int p2LowPrecisionCount = 0;
    int p2HighPrecisionCount = 0;

    int p1LowPrecisionCount = 0;
    int p1HighPrecisionCount = 0;

    float totalSourceContribution = 0.0f;
    float totalReflectingBoundaryContribution = 0.0f;

    float totalThroughput = 0.0f;
    int totalWalkLength = 0;

    float avgThroughput = 0.0f;
    float avgWalkLength = 0.0f;

    float tailFraction = 0.0f;

};

// ============================================================
// 最小完成码枚举
//
// 作用：
// 1. 让 runtime 不再只有 term=0/1 这种过于粗糙的状态
// 2. 为后面接真实 WoSt 的 completion code 做过渡
// ============================================================
enum WalkCompletionCodeLite : int {
    WALK_ONGOING = 0,
    WALK_TERMINATED_BY_MAX_LENGTH = 1,
    WALK_TERMINATED_BY_POSITION_RULE = 2
};

// ------------------------------------------------------------
// 计算源项贡献（更贴近真实 WoSt 的最小版本）
//
// 当前仍然不是 WoSt 的真实 source contribution 公式，
// 但已经不再是纯常数，而是显式依赖：
// - geometryDistance
// - 是否处于反射边界状态
//
// 当前规则：
// - 基础贡献：0.1
// - geometryDistance 越大，贡献略增
// - 若处于反射边界，则再增加一个很小的附加量
//
// 设计目的：
// 1. 保持简单可运行
// 2. 让 source contribution 开始受“当前步状态”影响
// 3. 为后面替换成更真实 WoSt 逻辑留接口
// ------------------------------------------------------------
inline float compute_source_contribution_skeleton(float geometryDistance,
                                                  const WalkStateLite& s) {
    float contribution = 0.1f;

    // geometryDistance 越大，贡献略增
    contribution += 0.02f * geometryDistance;

    // 若处于反射边界状态，则增加一个小附加量
    if (s.onReflectingBoundary) {
        contribution += 0.01f;
    }

    return contribution;
}

// ------------------------------------------------------------
// 计算反射边界贡献（当前 skeleton）
//
// 当前仍然不是 WoSt 的真实 reflecting boundary contribution 公式，
// 但现在把它单独拆出来，职责更接近真实实现。
//
// 当前规则：
// - 若当前更接近反射边界（reflect=1），给一个小贡献
// - 若不在反射边界侧，则贡献为 0
//
// 设计目的：
// 1. 让反射边界贡献不再隐含在别处
// 2. 让 totalReflectingBoundaryContribution 开始真正有意义
// 3. 为后面替换成更真实 WoSt 逻辑留接口
// ------------------------------------------------------------
inline float compute_reflecting_boundary_contribution_skeleton(
    float geometryDistance,
    const WalkStateLite& s
) {
    (void)geometryDistance;

    if (!(s.hasReflectingBoundary && s.onReflectingBoundary)) {
        return 0.0f;
    }

    // 第一版：直接用主机侧预处理好的 robin 数据构成一个最小边界贡献
    return s.robinValue + 0.1f * s.robinCoeffValue;
}



// ------------------------------------------------------------
// 计算 throughput 更新（更贴近真实 WoSt 的最小版本）
//
// 当前仍然不是 WoSt 的真实公式，
// 但已经不再是纯常数，而是显式依赖：
// - geometryDistance
// - 是否处于反射边界状态
//
// 设计目的：
// 1. 保持简单、可运行
// 2. 让 throughput 更新开始受到“当前步状态”的影响
// 3. 为后面接更真实的 WoSt 权重更新留接口
//
// 当前规则：
// - 基础衰减因子：0.95
// - 如果 geometryDistance 更大，则衰减略弱
// - 如果处于反射边界，则再额外乘一个小因子
//
// 注意：
// 这里只是“更接近真实形状”的 skeleton，不是最终公式。
// ------------------------------------------------------------
inline float compute_throughput_skeleton(float geometryDistance,
                                         const WalkStateLite& s) {
    float factor = 0.95f;

    // geometryDistance 越大，额外衰减越小
    // 当前做一个很温和的修正，避免一下子把行为改得太猛
    factor += 0.01f * geometryDistance;

    // 若已经处于反射边界状态，则额外施加一个小衰减
    if (s.onReflectingBoundary) {
        factor *= 0.98f;
    }

    return factor;
}

// ------------------------------------------------------------
// 终止判定（当前 skeleton，保持与此前原型行为一致）
//
// 这一版的目标：
// 1. 保留 completionCode 接口
// 2. 不改变你前面已经验证过的运行行为
//
// 当前规则：
// - 只有同时满足：
//     walkLength >= settings.maxWalkLength
//     且 currentPt[0] < 4.0f
//   才终止
//
// completionCode 仍然会被写出来，但先不再引入新的“提前终止”分支。
// ------------------------------------------------------------
inline WalkCompletionCodeLite should_terminate_skeleton(
    const WalkStateLite& s,
    const zombie::WalkSettings& settings
) {
    if (s.walkLength >= settings.maxWalkLength && s.currentPt[0] < 4.0f) {
        return WALK_TERMINATED_BY_MAX_LENGTH;
    }

    return WALK_ONGOING;
}


// ============================================================
// 单状态执行函数
//
// 新增参数：policy
// 作用：把阶段精度策略从“硬编码常量”升级成“显式配置”
// ============================================================
inline void execute_one_walkstate(WalkStateLite& s,
                                  const zombie::WalkSettings& settings,
                                  const PhasePolicyConfig& policy) {
    if (s.term) return;

    // ---------------- P0: 初始化阶段 ----------------
    // 这一阶段目前只做阶段标记；
    // 后面如果你要接更真实的执行上下文，可以从这里扩展。
    s.phase = P0_Init;

    // ---------------- P1: 几何阶段 ----------------
    // 现在开始更贴近真实输入几何信息：
    // geometryDistance 不再直接等于 prevDistance，
    // 而是来自两类边界距离的较小值。
    //
    // 这仍然不是 WoSt 的真实几何查询，
    // 但已经比“单一 prevDistance 传播”更接近真实输入结构。
    s.geometryDistance =
        std::min(s.distToAbsorbingBoundary, s.distToReflectingBoundary);

    // ========================================================
    // P1 滞回策略：
    // - 若几何距离足够大，切入低精度
    // - 若几何距离足够小，切回高精度
    // - 中间区间保持上一次状态
    // ========================================================
    if (use_low_precision_for_p1(s.geometryDistance,
                                 s.p1LowPrecisionState,
                                 policy)) {
        s.usedLowPrecisionP1 = 1;
        s.p1LowPrecisionState = 1;
    } else {
        s.usedLowPrecisionP1 = 0;
        s.p1LowPrecisionState = 0;
    }

    s.phase = P1_Geometry;

    // ---------------- P2: 步进阶段 ----------------
    // 当前仍然保留“P2 默认低精度路径”的策略接口。
    if (use_low_precision_for_phase(P2_Step)) {
        s.usedLowPrecisionP2 = 1;
    } else {
        s.usedLowPrecisionP2 = 0;
    }

    // 用 geometryDistance 来驱动一次最小位置推进
    // 当前只是 skeleton，不是 WoSt 的真实球面采样更新。
    s.currentPt[0] += 0.25f;
    s.currentPt[1] += 0.75f;

    // 把“本步几何结果”记回 prevDistance，供下一步继续使用
    s.prevDistance = s.geometryDistance;

    // 步数在 P2 完成后递增
    s.walkLength += 1;
    s.phase = P2_Step;

    // ---------------- P3: 近边界阶段 ----------------
    // 当前开始更明确地区分：
    // 若反射边界距离不大于吸收边界距离，
    // 则认为当前更接近反射边界。
    if (s.distToReflectingBoundary <= s.distToAbsorbingBoundary) {
        s.onReflectingBoundary = 1;
    } else {
        s.onReflectingBoundary = 0;
    }
    s.phase = P3_NearBoundary;

    // ---------------- P4: 贡献阶段 ----------------
    // 现在把 P4 进一步拆成 3 类职责：
    // 1. source contribution
    // 2. reflecting boundary contribution
    // 3. throughput 更新
    //
    // 这一步之后，runtime 的职责划分会更接近真实 WoSt：
    // - source contribution
    // - reflecting boundary contribution
    // - throughput
    s.totalSourceContribution +=
        compute_source_contribution_skeleton(s.geometryDistance, s);

    s.totalReflectingBoundaryContribution +=
        compute_reflecting_boundary_contribution_skeleton(s.geometryDistance, s);

    s.throughput *= compute_throughput_skeleton(s.geometryDistance, s);
    s.phase = P4_Contribution;

    // ---------------- 终止检查 ----------------
    s.completionCode = should_terminate_skeleton(s, settings);
    if (s.completionCode != WALK_ONGOING) {
        s.term = 1;
    }
}

// ============================================================
// 执行一个任务块一次
//
// 新增参数：policy
// 这样任务运行时同时接收：
// - Zombie 原生运行参数 WalkSettings
// - 论文侧阶段精度策略 PhasePolicyConfig
// ============================================================
inline void run_task_once(sycl::queue& q,
                          std::vector<WalkStateLite>& walkTaskStates,
                          int nStepsPerLaunch,
                          const zombie::WalkSettings& settings,
                          const PhasePolicyConfig& policy) {
    if (walkTaskStates.empty()) return;

    sycl::buffer<WalkStateLite, 1> buf(
        walkTaskStates.data(),
        sycl::range<1>(walkTaskStates.size())
    );

    q.submit([&](sycl::handler& h) {
        auto acc = buf.get_access<sycl::access::mode::read_write>(h);

        h.parallel_for(sycl::range<1>(walkTaskStates.size()), [=](sycl::id<1> idx) {
            for (int step = 0; step < nStepsPerLaunch; ++step) {
                execute_one_walkstate(acc[idx[0]], settings, policy);
            }
        });
    });

    q.wait();
}

// ============================================================
// 是否允许回收 / 重封装
// ============================================================
inline bool should_recycle(bool tailMode,
                           int unfinishedCount,
                           int recycleThreshold) {
    if (!tailMode) {
        return false;
    }

    if (unfinishedCount == 0) {
        return false;
    }

    if (unfinishedCount > recycleThreshold) {
        return false;
    }

    return true;
}

// ============================================================
// 纯提取函数：提取未完成状态
// ============================================================
inline std::vector<WalkStateLite> extract_unfinished(
    const std::vector<WalkStateLite>& walkTaskStates
) {
    std::vector<WalkStateLite> repackedWalkTaskStates;

    for (const auto& s : walkTaskStates) {
        if (!s.term) {
            repackedWalkTaskStates.push_back(s);
        }
    }

    return repackedWalkTaskStates;
}

// ============================================================
// 动态粒度更新函数（当前最小版）
// ============================================================
inline int update_granularity(int currentTaskGranularity,
                              int repackedSize,
                              bool tailMode,
                              int B_min,
                              int B_max) {
    (void)B_max;

    if (!tailMode) {
        return currentTaskGranularity;
    }

    if (repackedSize == 0) {
        return 0;
    }

    if (repackedSize < B_min) {
        return B_min;
    }

    return repackedSize;
}

// ============================================================
// 收集一次任务块执行后的统计信息
// ============================================================
inline TaskExecutionStats collect_task_stats(
    const std::vector<WalkStateLite>& walkTaskStates,
    int currentTaskGranularity,
    int recycleThreshold,
    int B_min,
    int B_max
) {
    TaskExecutionStats stats{};
    stats.oldGranularity = currentTaskGranularity;

    for (const auto& s : walkTaskStates) {
        if (s.term) {
            stats.finishedCount++;
        } else {
            stats.unfinishedCount++;
        }

        switch (s.phase) {
            case P0_Init:         stats.p0Count++; break;
            case P1_Geometry:     stats.p1Count++; break;
            case P2_Step:         stats.p2Count++; break;
            case P3_NearBoundary: stats.p3Count++; break;
            case P4_Contribution: stats.p4Count++; break;
            default: break;
        }

        if (s.usedLowPrecisionP2) {
            stats.p2LowPrecisionCount++;
        } else {
            stats.p2HighPrecisionCount++;
        }

        if (s.usedLowPrecisionP1) {
            stats.p1LowPrecisionCount++;
        } else {
            stats.p1HighPrecisionCount++;
        }
        stats.totalSourceContribution += s.totalSourceContribution;
        stats.totalReflectingBoundaryContribution +=s.totalReflectingBoundaryContribution;
        stats.totalThroughput += s.throughput;
        stats.totalWalkLength += s.walkLength;
        
    }

    stats.tailMode = (stats.unfinishedCount > 0);

    stats.recycleDecision = should_recycle(
        stats.tailMode,
        stats.unfinishedCount,
        currentTaskGranularity == 0 ? 0 : recycleThreshold
    );

    stats.repackedSize = stats.recycleDecision ? stats.unfinishedCount : 0;

    stats.newGranularity = update_granularity(
        currentTaskGranularity,
        stats.repackedSize,
        stats.tailMode,
        B_min,
        B_max
    );

    // ========================================================
    // 平均量
    // ========================================================
    if (!walkTaskStates.empty()) {
        const float taskSize = static_cast<float>(walkTaskStates.size());

        stats.avgThroughput =
            stats.totalThroughput / taskSize;

        stats.avgWalkLength =
            static_cast<float>(stats.totalWalkLength) / taskSize;

        stats.tailFraction =
            static_cast<float>(stats.unfinishedCount) / taskSize;
    }

    return stats;
}

// ============================================================
// 打印任务块
// ============================================================
inline void print_task(const std::vector<WalkStateLite>& walkTaskStates,
                       const std::string& title) {
    std::cout << "---- " << title << " ----" << std::endl;

    for (size_t i = 0; i < walkTaskStates.size(); ++i) {
        std::cout << i
                  << ": pt=(" << walkTaskStates[i].currentPt[0]
                  << ", " << walkTaskStates[i].currentPt[1] << ")"
                  << ", throughput=" << walkTaskStates[i].throughput
                  << ", walkLength=" << walkTaskStates[i].walkLength
                  << ", source=" << walkTaskStates[i].totalSourceContribution
                  << ", reflectContrib="
                  << walkTaskStates[i].totalReflectingBoundaryContribution
                  << ", phase=" << walkTaskStates[i].phase
                  << ", reflect=" << walkTaskStates[i].onReflectingBoundary
                  << ", term=" << walkTaskStates[i].term
                  << ", completionCode=" << walkTaskStates[i].completionCode
                  << std::endl;
    }
}