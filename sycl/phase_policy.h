#pragma once

// ============================================================
// 阶段定义
// ============================================================
enum Phase : int {
    P0_Init = 0,
    P1_Geometry = 1,
    P2_Step = 2,
    P3_NearBoundary = 3,
    P4_Contribution = 4
};

// ============================================================
// 阶段精度策略配置
//
// 现在把 P1 从单阈值升级成最小滞回双阈值：
// - tau_in : 切入低精度的阈值
// - tau_out: 切回高精度的阈值
//
// 约定：tau_in >= tau_out
// ============================================================
struct PhasePolicyConfig {
    float tau_in = 0.7f;
    float tau_out = 0.4f;
};

// ============================================================
// P2 阶段精度策略
// 当前最小策略：P2 固定低精度路径
// ============================================================
inline bool use_low_precision_for_phase(int phase) {
    switch (phase) {
        case P2_Step:
            return true;
        default:
            return false;
    }
}

// ============================================================
// P1 阶段最小滞回策略
//
// 参数：
// - geometryDistance: 当前几何阶段距离
// - prevState: 上一次 P1 的低精度状态（1=低，0=高）
// - policy: 含 tau_in / tau_out 的配置
//
// 规则：
// 1. geometryDistance >= tau_in  -> 低精度
// 2. geometryDistance <= tau_out -> 高精度
// 3. 中间区间                    -> 保持 prevState
//
// 这就是最小可运行版的滞回控制骨架。
// ============================================================
inline bool use_low_precision_for_p1(float geometryDistance,
                                     int prevState,
                                     const PhasePolicyConfig& policy) {
    if (geometryDistance >= policy.tau_in) {
        return true;
    }

    if (geometryDistance <= policy.tau_out) {
        return false;
    }

    // 中间区间：保持上一次状态
    return (prevState != 0);
}