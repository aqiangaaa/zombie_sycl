#pragma once

// ============================================================
// 性能指标采集（论文第7章）
//
// 统一定义性能、负载与精度三类指标：
// - 吞吐 θ、加速比 S、效率 E
// - 尾部比例 ρ_tail
// - 相对误差 ε_rel
// - 调度开销统计
// - 混合精度统计
// ============================================================

#include <cmath>
#include <cstdint>
#include <vector>

// ============================================================
// 调度性能指标
// ============================================================
struct SchedulingMetrics {
    // 吞吐 θ: 单位时间完成的轨迹数
    double throughput = 0.0;

    // 加速比 S: θ_parallel / θ_serial
    double speedup = 0.0;

    // 效率 E: S / Nc
    double efficiency = 0.0;

    // 尾部比例 ρ_tail: T_tail / T_total
    double tailFraction = 0.0;

    // 调度开销: 队列操作 + 回收重封装的时间占比
    double schedulingOverhead = 0.0;

    // 总执行时间
    double totalTime = 0.0;

    // 总完成轨迹数
    int totalWalksCompleted = 0;

    // 总任务块执行次数
    int totalTasksExecuted = 0;

    // 总回收次数
    int totalRecycles = 0;

    // 粒度演化历史
    std::vector<int> granularityHistory;

    void computeDerived(double baselineThroughput = 0.0, int Nc = 1) {
        if (totalTime > 0.0) {
            throughput = totalWalksCompleted / totalTime;
        }
        if (baselineThroughput > 0.0) {
            speedup = throughput / baselineThroughput;
        }
        if (Nc > 0 && speedup > 0.0) {
            efficiency = speedup / Nc;
        }
    }
};

// ============================================================
// 混合精度指标
// ============================================================
struct MixedPrecisionMetrics {
    // 各阶段执行次数
    int64_t p0Count = 0;
    int64_t p1Count = 0;
    int64_t p2Count = 0;
    int64_t p3Count = 0;
    int64_t p4Count = 0;

    // P1 精度分布
    int64_t p1FP32Count = 0;
    int64_t p1FP64Count = 0;

    // P2 精度分布
    int64_t p2FP32Count = 0;
    int64_t p2FP64Count = 0;

    // 精度切换次数
    int64_t precisionSwitchCount = 0;

    // FP64 覆盖比例 π64
    double fp64CoverageRatio() const {
        int64_t total = p1Count + p2Count + p3Count + p4Count;
        if (total == 0) return 0.0;
        int64_t fp64 = p1FP64Count + p2FP64Count + p3Count + p4Count;
        return static_cast<double>(fp64) / static_cast<double>(total);
    }

    // 从 TaskExecutionStats 累积
    void accumulate(int p0, int p1, int p2, int p3, int p4,
                    int p1Low, int p1High, int p2Low, int p2High) {
        p0Count += p0;
        p1Count += p1;
        p2Count += p2;
        p3Count += p3;
        p4Count += p4;
        p1FP32Count += p1Low;
        p1FP64Count += p1High;
        p2FP32Count += p2Low;
        p2FP64Count += p2High;
    }
};

// ============================================================
// 精度指标
// ============================================================
struct AccuracyMetrics {
    // 相对误差 ε_rel = |u_test - u_ref| / |u_ref|
    double relativeError = 0.0;

    // 绝对误差
    double absoluteError = 0.0;

    static AccuracyMetrics compute(double estimate, double reference) {
        AccuracyMetrics m;
        m.absoluteError = std::fabs(estimate - reference);
        double absRef = std::fabs(reference);
        m.relativeError = (absRef > 1e-10) ? m.absoluteError / absRef : m.absoluteError;
        return m;
    }
};

// ============================================================
// 综合实验结果
// ============================================================
struct ExperimentResult {
    SchedulingMetrics scheduling;
    MixedPrecisionMetrics precision;
    AccuracyMetrics accuracy;

    float estimatedSolution = 0.0f;
    float standardError = 0.0f;
    float analyticSolution = 0.0f;
};
