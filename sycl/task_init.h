#pragma once

#include <zombie/zombie.h>
#include "walkstate_bridge.h"
#include "phase_policy.h"

#include <array>
#include <vector>
#include <algorithm>

// ============================================================
// 初始化配置
//
// 作用：
// 1. 把当前“测试用初始化参数”从代码逻辑里抽出来
// 2. 后面如果你要：
//    - 扫参数
//    - 换不同初始状态
//    - 接真实 SamplePoint / demo 输入
//    都可以从这个结构继续往下扩展
// ============================================================
struct TaskInitConfig {
    // 初始位置：第 i 条状态会在这个 base 上做偏移
    std::array<float, 2> basePt = {0.0f, 100.0f};

    // 每条状态在 x/y 上的增量
    std::array<float, 2> ptStride = {1.0f, 1.0f};

    // 初始法向
    std::array<float, 2> currentNormal = {0.0f, 1.0f};

    // 初始前一步方向
    std::array<float, 2> prevDirection = {1.0f, 0.0f};

    // 初始距离 / 权重 / 步数 / 边界状态
    float prevDistance = 0.5f;
    float throughput = 1.0f;
    int walkLength = 0;
    bool onReflectingBoundary = false;
};

// ============================================================
// 运行时实验配置
//
// 作用：
// 1. 把当前 driver 中分散的关键参数集中起来
// 2. 方便后面做参数扫描 / 实验对照
// ============================================================
struct RuntimeExperimentConfig {
    int B = 8;
    int B_min = 2;
    int B_max = 16;

    int firstLaunchSteps = 3;
    int secondLaunchSteps = 2;

    int recycleThreshold = 4;

    bool useSamplePointInit = true;

    zombie::WalkSettings walkSettings =
        zombie::WalkSettings(1e-3f, 1e-3f, 2, false);

    PhasePolicyConfig phasePolicy{};
};

// ============================================================
// 生成一组主机侧真实 Zombie::WalkState
//
// 当前版本：
// - 仍然是测试初始化
// - 但已经不再把参数写死在函数体里
// ============================================================
inline std::vector<zombie::WalkState<Real, DIM>>
make_initial_host_walkstates(int B, const TaskInitConfig& cfg) {
    std::vector<zombie::WalkState<Real, DIM>> hostStates(B);

    for (int i = 0; i < B; ++i) {
        hostStates[i].currentPt[0] = cfg.basePt[0] + i * cfg.ptStride[0];
        hostStates[i].currentPt[1] = cfg.basePt[1] + i * cfg.ptStride[1];

        hostStates[i].currentNormal[0] = cfg.currentNormal[0];
        hostStates[i].currentNormal[1] = cfg.currentNormal[1];

        hostStates[i].prevDirection[0] = cfg.prevDirection[0];
        hostStates[i].prevDirection[1] = cfg.prevDirection[1];

        hostStates[i].prevDistance = cfg.prevDistance;
        hostStates[i].throughput = cfg.throughput;
        hostStates[i].walkLength = cfg.walkLength;
        hostStates[i].onReflectingBoundary = cfg.onReflectingBoundary;
    }

    return hostStates;
}

// ============================================================
// 把主机侧真实状态数组，打包成一个 WalkTask 的状态数组
// ============================================================
inline std::vector<WalkStateLite>
make_initial_walktask_states(const std::vector<zombie::WalkState<Real, DIM>>& hostStates) {
    std::vector<WalkStateLite> walkTaskStates(hostStates.size());

    for (size_t i = 0; i < hostStates.size(); ++i) {
        walkTaskStates[i] = packState(hostStates[i]);
    }

    return walkTaskStates;
}

// ============================================================
// 从真实 SamplePoint 初始化 WalkTask 状态数组
//
// 这个版本比单纯 packState(hostStates) 更贴近真实输入，
// 因为它把两类边界距离都保留下来了：
// - distToAbsorbingBoundary
// - distToReflectingBoundary
// ============================================================
inline std::vector<WalkStateLite>
make_initial_walktask_states(
    const std::vector<zombie::WalkState<Real, DIM>>& hostStates,
    const std::vector<zombie::SamplePoint<Real, DIM>>& samplePoints
) {
    std::vector<WalkStateLite> walkTaskStates(hostStates.size());

    for (size_t i = 0; i < hostStates.size(); ++i) {
        walkTaskStates[i] = packState(hostStates[i]);

        // 用真实 SamplePoint 里的两类边界距离覆盖占位值
        walkTaskStates[i].distToAbsorbingBoundary = samplePoints[i].distToAbsorbingBoundary;
        walkTaskStates[i].distToReflectingBoundary = samplePoints[i].distToReflectingBoundary;


        for (int d = 0; d < int(DIM); ++d) {
            walkTaskStates[i].normal[d] = samplePoints[i].normal[d];
        }

        // geometryDistance 当前继续取两者较小值，作为最小可运行版
        walkTaskStates[i].geometryDistance =
            std::min(samplePoints[i].distToAbsorbingBoundary,
                     samplePoints[i].distToReflectingBoundary);
        // 主机侧预处理后的“问题定义层”数值
        walkTaskStates[i].sourceValue = 0.1f;

        walkTaskStates[i].hasReflectingBoundary =
            (samplePoints[i].distToReflectingBoundary <=
            samplePoints[i].distToAbsorbingBoundary) ? 1 : 0;
    }

    return walkTaskStates;
}

// ============================================================
// 从真实 Zombie::SamplePoint 数组生成主机侧 WalkState 数组
//
// 这是第一次“更贴近真实 Zombie 输入”的入口。
// 当前规则仍然是最小映射：
// - currentPt      <- samplePoint.pt
// - currentNormal  <- samplePoint.normal
// - prevDirection  <- (1, 0) 先给一个固定值
// - prevDistance   <- 用吸收/反射边界距离中的较小者
// - throughput     <- 1.0
// - walkLength     <- 0
// - onReflectingBoundary <- false
//
// 后面如果你开始真正接 WoS/WoSt 的初始化逻辑，
// 可以继续在这个函数里逐步替换。
// ============================================================
inline std::vector<zombie::WalkState<Real, DIM>>
make_initial_host_walkstates_from_sample_points(
    const std::vector<zombie::SamplePoint<Real, DIM>>& samplePoints
) {
    std::vector<zombie::WalkState<Real, DIM>> hostStates(samplePoints.size());

    for (size_t i = 0; i < samplePoints.size(); ++i) {
        const auto& sp = samplePoints[i];

        // 当前位置来自真实 SamplePoint
        hostStates[i].currentPt = sp.pt;

        // 法向来自真实 SamplePoint
        hostStates[i].currentNormal = sp.normal;

        // 当前先给一个固定前向方向，后面再逐步替换成更真实的初始化策略
        hostStates[i].prevDirection[0] = 1.0f;
        hostStates[i].prevDirection[1] = 0.0f;

        // 用两类边界距离中的较小值，作为一个最小可运行版的初始距离
        hostStates[i].prevDistance =
            std::min(sp.distToAbsorbingBoundary, sp.distToReflectingBoundary);

        hostStates[i].throughput = 1.0f;
        hostStates[i].walkLength = 0;
        hostStates[i].onReflectingBoundary = false;
    }

    return hostStates;
}

// ============================================================
// 用真实 PDE 在主机侧预处理设备需要的“纯数值字段”
//
// 目的：
// 1. 不把 zombie::PDE 直接传进 SYCL kernel
// 2. 在 host 上先调用 PDE 的真实回调
// 3. 把结果写进 WalkStateLite，可供 device 侧直接读取
//
// 当前先预处理两个字段：
// - sourceValue
// - hasReflectingBoundary
//
// 后面还可以继续扩展：
// - robinValue
// - robinCoeffValue
// ============================================================
inline void apply_host_pde_fields(
    std::vector<WalkStateLite>& walkTaskStates,
    const zombie::PDE<float, DIM>& pde
) {
    for (auto& s : walkTaskStates) {
        zombie::Vector<DIM> x = zombie::Vector<DIM>::Zero();
        zombie::Vector<DIM> n = zombie::Vector<DIM>::Zero();

        for (int d = 0; d < int(DIM); ++d) {
            x[d] = s.currentPt[d];
            n[d] = s.normal[d];
        }

        // ---------------- sourceValue ----------------
        if (pde.source) {
            s.sourceValue = pde.source(x);
        } else {
            s.sourceValue = 0.0f;
        }

        // ---------------- hasReflectingBoundary ----------------
        if (pde.hasReflectingBoundaryConditions) {
            s.hasReflectingBoundary =
                pde.hasReflectingBoundaryConditions(x) ? 1 : 0;
        } else {
            s.hasReflectingBoundary = 0;
        }

        // ---------------- robinValue ----------------
        if (pde.robin) {
            s.robinValue = pde.robin(x, n, s.hasReflectingBoundary != 0);
        } else {
            s.robinValue = 0.0f;
        }

        // ---------------- robinCoeffValue ----------------
        if (pde.robinCoeff) {
            s.robinCoeffValue =
                pde.robinCoeff(x, n, s.hasReflectingBoundary != 0);
        } else {
            s.robinCoeffValue = 0.0f;
        }
    }
}