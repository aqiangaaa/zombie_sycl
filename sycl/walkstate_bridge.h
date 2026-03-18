#pragma once

#include <zombie/zombie.h>
#include <array>
#include <cstdint>
#include "pcg32_device.h"

using Real = float;

// ============================================================
// DIM 配置：2 或 3，全局唯一定义点
// 修改此值即可切换 2D / 3D 模式
// ============================================================
#ifndef ZOMBIE_SYCL_DIM
#define ZOMBIE_SYCL_DIM 2
#endif
constexpr int DIM = ZOMBIE_SYCL_DIM;

// ============================================================
// 设备侧轻量状态
//
// 使用固定大小数组 float[3]，同时兼容 2D 和 3D。
// DIM=2 时第三个分量不使用但不影响正确性。
// 这样 WalkStateLite 的内存布局在编译期固定，
// 可以安全地放进 SYCL buffer。
// ============================================================
constexpr int MAX_DIM = 3;

struct WalkStateLite {
    float totalReflectingBoundaryContribution;
    float totalSourceContribution;
    float totalTerminalContribution;

    float currentPt[MAX_DIM];
    float currentNormal[MAX_DIM];
    float prevDirection[MAX_DIM];
    float normal[MAX_DIM];

    float prevDistance;
    float throughput;
    float geometryDistance;

    float distToAbsorbingBoundary;
    float distToReflectingBoundary;

    int walkLength;
    int onReflectingBoundary;

    int phase;
    int term;
    int completionCode;

    int visitedP0;
    int visitedP1;
    int visitedP2;
    int visitedP3;
    int visitedP4;

    int usedLowPrecisionP2;
    int usedLowPrecisionP1;
    // P1 滞回状态：1=低精度记忆，0=高精度记忆
    int p1LowPrecisionState;

    float sourceValue;
    int hasReflectingBoundary;

    float robinValue;
    float robinCoeffValue;
    float dirichletValue;
    PCG32State rng;
};

// ============================================================
// 把真实 Zombie::WalkState 打包成设备侧轻量状态
// ============================================================
inline WalkStateLite packState(const zombie::WalkState<Real, DIM>& s) {
    WalkStateLite out{};
    out.totalReflectingBoundaryContribution = s.totalReflectingBoundaryContribution;
    out.totalSourceContribution = s.totalSourceContribution;
    out.totalTerminalContribution = 0.0f;

    for (int d = 0; d < MAX_DIM; ++d) {
        out.currentPt[d] = 0.0f;
        out.currentNormal[d] = 0.0f;
        out.prevDirection[d] = 0.0f;
        out.normal[d] = 0.0f;
    }
    for (int d = 0; d < DIM; ++d) {
        out.currentPt[d] = s.currentPt[d];
        out.currentNormal[d] = s.currentNormal[d];
        out.prevDirection[d] = s.prevDirection[d];
        out.normal[d] = s.currentNormal[d];
    }

    out.prevDistance = s.prevDistance;
    out.throughput = s.throughput;
    out.geometryDistance = s.prevDistance;
    out.distToAbsorbingBoundary = s.prevDistance;
    out.distToReflectingBoundary = s.prevDistance;

    out.walkLength = s.walkLength;
    out.onReflectingBoundary = s.onReflectingBoundary ? 1 : 0;

    out.phase = 0;
    out.term = 0;
    out.completionCode = 0;

    out.visitedP0 = 0;
    out.visitedP1 = 0;
    out.visitedP2 = 0;
    out.visitedP3 = 0;
    out.visitedP4 = 0;

    out.usedLowPrecisionP2 = 0;
    out.usedLowPrecisionP1 = 0;
    out.p1LowPrecisionState = 1;
    out.sourceValue = 0.0f;
    out.hasReflectingBoundary = 0;

    out.robinValue = 0.0f;
    out.robinCoeffValue = 0.0f;
    out.dirichletValue = 0.0f;
    out.rng = pcg32_seed(0);

    return out;
}

// ============================================================
// 把设备侧轻量状态写回真实 Zombie::WalkState
// ============================================================
inline void unpackState(const WalkStateLite& in, zombie::WalkState<Real, DIM>& s) {
    s.totalReflectingBoundaryContribution = in.totalReflectingBoundaryContribution;
    s.totalSourceContribution = in.totalSourceContribution;

    for (int d = 0; d < DIM; ++d) {
        s.currentPt[d] = in.currentPt[d];
        s.currentNormal[d] = in.currentNormal[d];
        s.prevDirection[d] = in.prevDirection[d];
    }

    s.prevDistance = in.prevDistance;
    s.throughput = in.throughput;
    s.walkLength = in.walkLength;
    s.onReflectingBoundary = (in.onReflectingBoundary != 0);
}
