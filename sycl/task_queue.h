#pragma once

// ============================================================
// 任务队列模型（论文第4章）
//
// 实现 MPE 侧任务队列 Q：
// - WalkTask: 最小可调度单元，包含 B 条轨迹
// - TaskQueue: 队列管理，支持提交/取出/回收
// - TailDetector: 尾部阶段检测
// - GranularityPolicy: 动态粒度 B 的自适应调整
// ============================================================

#include "walkstate_bridge.h"
#include "phase_policy.h"

#include <vector>
#include <deque>
#include <algorithm>
#include <cmath>

// ============================================================
// WalkTask: 最小可调度单元
// ============================================================
struct WalkTask {
    int taskId = 0;
    int granularity = 0;                    // B: 本任务块包含的轨迹数
    std::vector<WalkStateLite> states;      // 轨迹状态数组
    PhasePolicyConfig policy;               // 精度策略（只读）
    bool isRecycled = false;                // 是否为回收重封装的任务块

    int finishedCount() const {
        int n = 0;
        for (const auto& s : states) {
            if (s.term) n++;
        }
        return n;
    }

    int unfinishedCount() const {
        return static_cast<int>(states.size()) - finishedCount();
    }

    bool allFinished() const {
        for (const auto& s : states) {
            if (!s.term) return false;
        }
        return true;
    }
};

// ============================================================
// GranularityPolicy: 动态粒度策略
//
// 论文第4章核心：稳态粗粒度 → 尾部细粒度
// ============================================================
struct GranularityPolicy {
    int B_init;     // 初始粒度（通常等于 B_max）
    int B_min;      // 最小粒度
    int B_max;      // 最大粒度

    GranularityPolicy(int init = 64, int bmin = 4, int bmax = 64)
        : B_init(init), B_min(bmin), B_max(bmax) {}

    // 稳态阶段：使用 B_init
    // 尾部阶段：根据未完成轨迹数量动态调整
    int computeGranularity(bool tailMode, int unfinishedCount) const {
        if (!tailMode) return B_init;
        if (unfinishedCount <= 0) return 0;
        if (unfinishedCount <= B_min) return unfinishedCount;
        // 尾部阶段：尽量分成多块，每块不小于 B_min
        int numChunks = (unfinishedCount + B_min - 1) / B_min;
        int B = (unfinishedCount + numChunks - 1) / numChunks;
        return std::clamp(B, B_min, B_max);
    }
};

// ============================================================
// TailDetector: 尾部阶段检测
//
// 论文第4章：当 |Q|/Nc < threshold 时进入尾部
// ============================================================
struct TailDetector {
    int numComputeUnits = 1;        // Nc: 并行计算单元数
    float tailThreshold = 2.0f;     // 当 |Q|/Nc < threshold 时进入尾部

    TailDetector(int nc = 1, float thresh = 2.0f)
        : numComputeUnits(nc), tailThreshold(thresh) {}

    bool isTailPhase(int queueSize) const {
        float ratio = static_cast<float>(queueSize) /
                      static_cast<float>(std::max(1, numComputeUnits));
        return ratio < tailThreshold;
    }
};

// ============================================================
// TaskQueue: 任务队列
// ============================================================
struct TaskQueue {
    std::deque<WalkTask> queue;
    int nextTaskId = 0;
    int totalSubmitted = 0;
    int totalCompleted = 0;
    int totalRecycled = 0;

    void submit(WalkTask task) {
        task.taskId = nextTaskId++;
        task.granularity = static_cast<int>(task.states.size());
        totalSubmitted++;
        queue.push_back(std::move(task));
    }

    bool empty() const { return queue.empty(); }
    int size() const { return static_cast<int>(queue.size()); }

    WalkTask fetch() {
        WalkTask task = std::move(queue.front());
        queue.pop_front();
        return task;
    }

    void markCompleted() { totalCompleted++; }

    float completionRatio() const {
        return (totalSubmitted > 0)
            ? static_cast<float>(totalCompleted) / static_cast<float>(totalSubmitted)
            : 0.0f;
    }
};

// ============================================================
// 从轨迹集合按粒度 B 分块生成任务
// ============================================================
inline std::vector<WalkTask> create_tasks_from_states(
    const std::vector<WalkStateLite>& states,
    int B,
    const PhasePolicyConfig& policy
) {
    std::vector<WalkTask> tasks;
    const int n = static_cast<int>(states.size());
    for (int i = 0; i < n; i += B) {
        WalkTask task;
        int end = std::min(i + B, n);
        task.states.assign(states.begin() + i, states.begin() + end);
        task.granularity = end - i;
        task.policy = policy;
        task.isRecycled = false;
        tasks.push_back(std::move(task));
    }
    return tasks;
}

// ============================================================
// 回收重封装：提取未完成轨迹，按新粒度分块
//
// 论文第4章：保持轨迹状态完整连续（rng/position/throughput 不变）
// ============================================================
inline std::vector<WalkTask> recycle_and_repack(
    const WalkTask& completedTask,
    const GranularityPolicy& granPolicy,
    bool tailMode
) {
    // 1. 提取未完成轨迹
    std::vector<WalkStateLite> unfinished;
    for (const auto& s : completedTask.states) {
        if (!s.term) {
            unfinished.push_back(s);
        }
    }
    if (unfinished.empty()) return {};

    // 2. 计算新粒度
    int newB = granPolicy.computeGranularity(tailMode, static_cast<int>(unfinished.size()));
    if (newB <= 0) newB = static_cast<int>(unfinished.size());

    // 3. 按新粒度分块
    std::vector<WalkTask> newTasks;
    const int n = static_cast<int>(unfinished.size());
    for (int i = 0; i < n; i += newB) {
        WalkTask task;
        int end = std::min(i + newB, n);
        task.states.assign(unfinished.begin() + i, unfinished.begin() + end);
        task.granularity = end - i;
        task.policy = completedTask.policy;
        task.isRecycled = true;
        newTasks.push_back(std::move(task));
    }
    return newTasks;
}

// ============================================================
// 调度统计
// ============================================================
struct SchedulingStats {
    int totalTasksExecuted = 0;
    int totalRecycles = 0;
    int totalWalksCompleted = 0;
    std::vector<int> granularityHistory;
    double totalExecutionTime = 0.0;
    double tailPhaseTime = 0.0;

    float tailFraction() const {
        return (totalExecutionTime > 0.0)
            ? static_cast<float>(tailPhaseTime / totalExecutionTime)
            : 0.0f;
    }

    float throughput() const {
        return (totalExecutionTime > 0.0)
            ? static_cast<float>(totalWalksCompleted / totalExecutionTime)
            : 0.0f;
    }
};
