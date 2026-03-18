#pragma once

// ============================================================
// Welford 在线统计
//
// 对齐 zombie::SampleStatistics 的核心功能：
// 在线计算均值和方差，无需存储所有样本。
//
// 参考：Welford's online algorithm
// ============================================================

#include <cmath>

struct WelfordAccumulator {
    int count = 0;
    float mean = 0.0f;
    float M2 = 0.0f;

    void add(float x) {
        count++;
        float delta = x - mean;
        mean += delta / static_cast<float>(count);
        float delta2 = x - mean;
        M2 += delta * delta2;
    }

    float variance() const {
        return (count > 1) ? M2 / static_cast<float>(count - 1) : 0.0f;
    }

    float standardError() const {
        return (count > 1) ? std::sqrt(variance() / static_cast<float>(count)) : 0.0f;
    }

    // 合并两个累加器（用于并行归约）
    void merge(const WelfordAccumulator& other) {
        if (other.count == 0) return;
        if (count == 0) {
            *this = other;
            return;
        }
        int newCount = count + other.count;
        float delta = other.mean - mean;
        float newMean = mean + delta * static_cast<float>(other.count) / static_cast<float>(newCount);
        float newM2 = M2 + other.M2 +
                       delta * delta * static_cast<float>(count) * static_cast<float>(other.count) /
                       static_cast<float>(newCount);
        count = newCount;
        mean = newMean;
        M2 = newM2;
    }
};

// ============================================================
// 每个采样点的完整统计
// ============================================================
struct PointEstimate {
    float pt[3] = {0.0f};       // 采样点位置
    WelfordAccumulator solution; // 解的统计
    WelfordAccumulator walkLen;  // walk 长度统计

    float analyticSolution = 0.0f; // 解析解（如果有）

    float relativeError() const {
        float absAnalytic = std::fabs(analyticSolution);
        if (absAnalytic < 1e-6f) {
            return std::fabs(solution.mean);
        }
        return std::fabs(solution.mean - analyticSolution) / absAnalytic;
    }
};
