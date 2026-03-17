#pragma once

#include <zombie/zombie.h>
#include <array>

using Real = float;
constexpr int DIM = 2;

// ============================================================
// 设备侧轻量状态
//
// 作用：
// 1. 作为真实 Zombie::WalkState 的设备侧桥接结构
// 2. 让状态更容易放进 SYCL kernel
// ============================================================
struct WalkStateLite {
    float totalReflectingBoundaryContribution;
    float totalSourceContribution;

    std::array<float, 2> currentPt;
    std::array<float, 2> currentNormal;
    std::array<float, 2> prevDirection;

    float prevDistance;
    float throughput;
    float geometryDistance;
    // ========================================================
    // 更贴近真实 Zombie / WoSt 输入的两类边界距离
    // ========================================================
    float distToAbsorbingBoundary;
    float distToReflectingBoundary;

    int walkLength;
    int onReflectingBoundary;

    int phase;
    int term;
    int completionCode;

    int usedLowPrecisionP2;
    int usedLowPrecisionP1;
    // ========================================================
    // P1 滞回状态：
    // 1 -> 当前记忆为“低精度路径”
    // 0 -> 当前记忆为“高精度路径”
    //
    // 当 geometryDistance 落在 (tau_out, tau_in) 中间区间时，
    // 我们不重新切换，而是沿用这个状态。
    // ========================================================
    int p1LowPrecisionState;
    
    float sourceValue;
    int hasReflectingBoundary;

    float robinValue;
    float robinCoeffValue;

    float normal[DIM];

};


// ============================================================
// 把真实 Zombie::WalkState 打包成设备侧轻量状态
// ============================================================
inline WalkStateLite packState(const zombie::WalkState<Real, DIM>& s) {
    WalkStateLite out{};
    out.totalReflectingBoundaryContribution = s.totalReflectingBoundaryContribution;
    out.totalSourceContribution = s.totalSourceContribution;

    out.currentPt = {s.currentPt[0], s.currentPt[1]};
    out.currentNormal = {s.currentNormal[0], s.currentNormal[1]};
    out.prevDirection = {s.prevDirection[0], s.prevDirection[1]};

    out.prevDistance = s.prevDistance;
    out.throughput = s.throughput;
    out.geometryDistance = s.prevDistance;
    // 当前 packState 只能从真实 WalkState 里拿到 prevDistance，
    // 所以这里先用同一个值占位。
    // 真正更真实的两类边界距离，会在 task_init.h 的
    // SamplePoint 初始化入口里补进去。
    out.distToAbsorbingBoundary = s.prevDistance;
    out.distToReflectingBoundary = s.prevDistance;

    out.walkLength = s.walkLength;
    out.onReflectingBoundary = s.onReflectingBoundary ? 1 : 0;

    out.phase = 0;
    out.term = 0;
    out.completionCode = 0;
    out.usedLowPrecisionP2 = 0;
    out.usedLowPrecisionP1 = 0;
    out.p1LowPrecisionState = 1;
    out.sourceValue = 0.1f;
    out.hasReflectingBoundary = 0;

    out.robinValue = 0.0f;
    out.robinCoeffValue = 0.0f;

    for (int d = 0; d < int(DIM); ++d) {
        out.normal[d] = 0.0f;
    }
    if constexpr (DIM >= 2) {
        out.normal[1] = 1.0f;
    }

    return out;
}

// ============================================================
// 把设备侧轻量状态写回真实 Zombie::WalkState
// ============================================================
inline void unpackState(const WalkStateLite& in, zombie::WalkState<Real, DIM>& s) {
    s.totalReflectingBoundaryContribution = in.totalReflectingBoundaryContribution;
    s.totalSourceContribution = in.totalSourceContribution;

    s.currentPt[0] = in.currentPt[0];
    s.currentPt[1] = in.currentPt[1];

    s.currentNormal[0] = in.currentNormal[0];
    s.currentNormal[1] = in.currentNormal[1];

    s.prevDirection[0] = in.prevDirection[0];
    s.prevDirection[1] = in.prevDirection[1];

    s.prevDistance = in.prevDistance;
    s.throughput = in.throughput;

    s.walkLength = in.walkLength;
    s.onReflectingBoundary = (in.onReflectingBoundary != 0);
}