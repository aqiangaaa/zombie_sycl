#pragma once

// ============================================================
// 实验配置（论文第6-7章）
//
// 四组对照：
// - BASELINE: 全 FP64，固定 B，无回收
// - DG_ONLY:  全 FP64，动态 B，有回收
// - MP_ONLY:  混合精度，固定 B，无回收
// - DG_MP:    混合精度，动态 B，有回收
// ============================================================

#include "phase_policy.h"
#include "task_queue.h"
#include <string>

enum class ExperimentGroup {
    BASELINE,
    DG_ONLY,
    MP_ONLY,
    DG_MP
};

inline std::string groupName(ExperimentGroup g) {
    switch (g) {
        case ExperimentGroup::BASELINE: return "Baseline";
        case ExperimentGroup::DG_ONLY:  return "DG";
        case ExperimentGroup::MP_ONLY:  return "MP";
        case ExperimentGroup::DG_MP:    return "DG+MP";
    }
    return "?";
}

struct ExperimentConfig {
    ExperimentGroup group;

    // 调度参数
    int B;
    int B_min, B_max;
    bool enableRecycle;

    // 混合精度参数
    bool enableMixedPrecision;
    PhasePolicyConfig phasePolicy;

    // Walk 参数
    int maxWalkLength;
    float epsilonShell;
    int nWalksPerPoint;

    static ExperimentConfig baseline(int nWalks = 1024, int maxWalk = 128) {
        ExperimentConfig c;
        c.group = ExperimentGroup::BASELINE;
        c.B = 64; c.B_min = 64; c.B_max = 64;
        c.enableRecycle = false;
        c.enableMixedPrecision = false;
        c.maxWalkLength = maxWalk;
        c.epsilonShell = 1e-3f;
        c.nWalksPerPoint = nWalks;
        return c;
    }

    static ExperimentConfig dgOnly(int nWalks = 1024, int maxWalk = 128,
                                    int bInit = 64, int bMin = 4, int bMax = 64) {
        ExperimentConfig c;
        c.group = ExperimentGroup::DG_ONLY;
        c.B = bInit; c.B_min = bMin; c.B_max = bMax;
        c.enableRecycle = true;
        c.enableMixedPrecision = false;
        c.maxWalkLength = maxWalk;
        c.epsilonShell = 1e-3f;
        c.nWalksPerPoint = nWalks;
        return c;
    }

    static ExperimentConfig mpOnly(int nWalks = 1024, int maxWalk = 128,
                                    float tau_in = 0.5f, float tau_out = 0.2f) {
        ExperimentConfig c;
        c.group = ExperimentGroup::MP_ONLY;
        c.B = 64; c.B_min = 64; c.B_max = 64;
        c.enableRecycle = false;
        c.enableMixedPrecision = true;
        c.phasePolicy.tau_in = tau_in;
        c.phasePolicy.tau_out = tau_out;
        c.maxWalkLength = maxWalk;
        c.epsilonShell = 1e-3f;
        c.nWalksPerPoint = nWalks;
        return c;
    }

    static ExperimentConfig dgMp(int nWalks = 1024, int maxWalk = 128,
                                  int bInit = 64, int bMin = 4, int bMax = 64,
                                  float tau_in = 0.5f, float tau_out = 0.2f) {
        ExperimentConfig c;
        c.group = ExperimentGroup::DG_MP;
        c.B = bInit; c.B_min = bMin; c.B_max = bMax;
        c.enableRecycle = true;
        c.enableMixedPrecision = true;
        c.phasePolicy.tau_in = tau_in;
        c.phasePolicy.tau_out = tau_out;
        c.maxWalkLength = maxWalk;
        c.epsilonShell = 1e-3f;
        c.nWalksPerPoint = nWalks;
        return c;
    }
};
