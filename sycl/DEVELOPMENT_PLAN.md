# Zombie SYCL 迁移与创新点实现开发计划

## 项目概述

将 C++ header-only Zombie 框架（Walk-on-Spheres/Walk-on-Stars 随机游走 PDE 求解器）迁移到 SYCL，
并实现论文两个核心创新点：
- 创新点1（第4章）：自适应并行粒度的动态负载均衡调度
- 创新点2（第5章）：阶段感知混合精度策略

本地验证平台：2×RTX 3090 + Intel SYCL/DPC++ 编译器（icpx）

## 环境配置

```bash
source /home/tools/intel/oneapi/setvars.sh
cd /home/powerzhang/zombie/build_icpx
cmake .. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) <target_name>
```

## 当前进度（Phase 0 已完成）

| 文件 | 状态 | 说明 |
|------|------|------|
| `hello_sycl.cpp` | ✅ | SYCL 设备检测与基础 buffer 操作 |
| `zombie_sycl_smoke.cpp` | ✅ | Zombie 类型在 SYCL 中的兼容性验证 |
| `walkstate_lite_sycl.cpp` | ✅ | WalkStateLite 设备侧结构验证 |
| `walkstate_bridge_sycl.cpp` | ✅ | pack/unpack 双向桥接验证 |
| `walktask_lite_sycl.cpp` | ✅ | 基础任务块 SYCL 并行执行 |
| `walktask_phased_sycl.cpp` | ✅ | 阶段化执行骨架 + 参数扫描原型 |
| `walkstate_bridge.h` | ✅ | WalkStateLite 定义 + pack/unpack |
| `phase_policy.h` | ✅ | P0-P4 阶段枚举 + 滞回精度策略骨架 |
| `task_runtime.h` | ✅ | 执行引擎 + 统计 + 回收 + 粒度更新 |
| `task_init.h` | ✅ | 任务初始化 + PDE 字段预处理 |

### 当前骨架的已知局限

1. **仅 2D**：`constexpr int DIM = 2`，WalkStateLite 用 `std::array<float,2>`
2. **几何查询为假**：P2 步进后 `distToAbsorbingBoundary *= 0.25f`，未接入 FCPW
3. **RNG 为简易 LCG**：非 pcg32，质量不足以支撑收敛性验证
4. **Green 函数未实现**：`compute_throughput_skeleton` 返回 1.0f
5. **P3 反射边界为空占位**：`onReflectingBoundary = 0`
6. **混合精度仅标记**：`usedLowPrecisionP1/P2` 只是 flag，未真正切换数值类型
7. **无真实边界网格加载**：SamplePoint 手工构造，无 OBJ 文件输入
8. **无多 walk 统计**：每个采样点只跑 1 条轨迹，无 Welford 方差估计

---

## Phase 1：WoS 核心算法 SYCL 真实化（基础迁移）

> 目标：让 SYCL 内核执行的 WoS 随机游走在数值上与原始 Zombie CPU 版本一致。
> 每个 Step 产出一个可编译运行的验证程序。

### Step 1.1 — 扩展到 2D/3D 模板化

**目标**：将 `DIM=2` 硬编码改为模板参数，支持 2D 和 3D。

**改动文件**：
- `walkstate_bridge.h`：`WalkStateLite` 中 `std::array<float, 2>` → `std::array<float, DIM>`
- `task_runtime.h`：添加 `sample_unit_direction_3d()`
- `phase_policy.h`：无需改动（与维度无关）

**关键实现**：
```cpp
// 3D 球面均匀采样
inline void sample_unit_direction_3d(std::uint64_t& state,
                                      float& dx, float& dy, float& dz) {
    float u1 = next_uniform_01(state);
    float u2 = next_uniform_01(state);
    float z = 2.0f * u1 - 1.0f;
    float r = sycl::sqrt(1.0f - z * z);
    float phi = 2.0f * M_PI * u2;
    dx = r * sycl::cos(phi);
    dy = r * sycl::sin(phi);
    dz = z;
}
```

**验证方式**：
```bash
# 编译运行，对比 2D/3D 两种模式下的轨迹输出
make walktask_phased_sycl && ./walktask_phased_sycl
```

**验证标准**：2D 模式输出与改动前一致；3D 模式正常运行无崩溃。

---

### Step 1.2 — 设备侧 PCG32 随机数生成器

**目标**：替换 LCG 为 pcg32，保证 SYCL kernel 内可用。

**改动文件**：
- 新建 `sycl/pcg32_device.h`：设备侧 pcg32 实现（纯算术，无 std 依赖）
- `walkstate_bridge.h`：RNG 状态从 `uint64_t` 扩展为 `{state, inc}` 对
- `task_runtime.h`：替换 `advance_rng_state` / `next_uniform_01`

**关键实现**：
```cpp
struct PCG32State {
    std::uint64_t state;
    std::uint64_t inc;
};

inline std::uint32_t pcg32_next(PCG32State& rng) {
    std::uint64_t oldstate = rng.state;
    rng.state = oldstate * 6364136223846793005ULL + (rng.inc | 1);
    std::uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    std::uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}
```

**验证方式**：生成 10000 个随机数，检查均匀性（均值 ≈ 0.5，方差 ≈ 1/12）。

**验证标准**：均值误差 < 0.01，方差误差 < 0.01。

---

### Step 1.3 — 真实 Harmonic Green 函数（球域）

**目标**：实现 2D/3D Harmonic Ball Green 函数，替换 skeleton 中的常数返回值。

**改动文件**：
- 新建 `sycl/greens_function_device.h`：设备侧 Green 函数
- `task_runtime.h`：替换 `compute_source_contribution_skeleton` 和 `compute_throughput_skeleton`

**关键实现**（对齐 `include/zombie/core/distributions.h`）：
```cpp
// 2D Harmonic Ball: G(r,R) = ln(R/r) / (2π)
inline float harmonic_greens_fn_ball_2d(float r, float R) {
    return sycl::log(R / sycl::fmax(r, 1e-6f)) / (2.0f * M_PI);
}

// 3D Harmonic Ball: G(r,R) = (1/r - 1/R) / (4π)
inline float harmonic_greens_fn_ball_3d(float r, float R) {
    return (1.0f / sycl::fmax(r, 1e-6f) - 1.0f / R) / (4.0f * M_PI);
}

// Poisson kernel (方向采样版): directionSampledPoissonKernel
// 2D: 返回 1.0（均匀采样时 Poisson kernel 与采样 PDF 抵消）
// 3D: 返回 1.0（同理）
inline float direction_sampled_poisson_kernel() {
    return 1.0f;
}

// Source contribution: G_norm * sourceValue
// G_norm_2d(R) = R²/4
// G_norm_3d(R) = R²/6
inline float greens_fn_norm_2d(float R) { return R * R / 4.0f; }
inline float greens_fn_norm_3d(float R) { return R * R / 6.0f; }
```

**验证方式**：
- 构造 Laplace 方程 Δu=0，圆域边界 u=1，解析解 u(x)=1
- 运行 N=1000 条轨迹，检查估计值收敛到 1.0

**验证标准**：1000 条轨迹均值与解析解的相对误差 < 5%。

---

### Step 1.4 — Host 侧 FCPW 几何查询预计算

**目标**：在 host 侧用 FCPW 计算真实边界距离，写入 WalkStateLite 供 kernel 使用。

**设计思路**：
FCPW 的加速结构（BVH）无法直接在 SYCL kernel 中使用。
采用"host 预计算 + device 步进 + host 回写距离"的迭代模式：
1. Host 用 FCPW 计算每个状态的 distToAbsorbingBoundary
2. 写入 WalkStateLite buffer
3. Device kernel 执行一步 WoS（用预计算的距离作为球半径）
4. Host 读回新位置，重新计算距离
5. 循环直到所有轨迹终止

**改动文件**：
- 新建 `sycl/geometry_host.h`：封装 FCPW 查询接口
- `task_runtime.h`：新增 `run_task_host_device_loop()` 函数
- `CMakeLists.txt`：新增编译目标

**关键实现**：
```cpp
// geometry_host.h
#include <zombie/utils/fcpw_geometric_queries.h>

template <int DIM>
struct HostGeometry {
    zombie::FcpwDirichletBoundaryHandler<DIM> dirichletHandler;
    zombie::GeometricQueries<DIM> queries;

    void build(const std::vector<zombie::Vector<DIM>>& positions,
               const std::vector<zombie::Vectori<DIM>>& indices) {
        dirichletHandler.buildAccelerationStructure(positions, indices, false);
        zombie::populateGeometricQueriesForDirichletBoundary(
            dirichletHandler, queries);
    }

    float distToAbsorbing(const zombie::Vector<DIM>& pt) {
        return queries.computeDistToAbsorbingBoundary(pt, false);
    }
};
```

**Host-Device 迭代循环**：
```cpp
void run_task_host_device_loop(
    sycl::queue& q,
    std::vector<WalkStateLite>& states,
    HostGeometry<DIM>& geom,
    const WalkSettings& settings,
    const PhasePolicyConfig& policy
) {
    for (int iter = 0; iter < settings.maxWalkLength; ++iter) {
        // 1. Host: 更新每个状态的边界距离
        for (auto& s : states) {
            if (s.term) continue;
            Vector<DIM> pt;
            for (int d = 0; d < DIM; ++d) pt[d] = s.currentPt[d];
            s.distToAbsorbingBoundary = geom.distToAbsorbing(pt);
        }
        // 2. Device: 执行一步
        run_task_once(q, states, 1, settings, policy);
    }
}
```

**验证方式**：
- 加载一个简单圆形边界（或正方形），设置 Dirichlet 边界条件
- 对比 host-device 循环结果与纯 CPU Zombie 结果

**验证标准**：与 CPU 版本的相对误差 < 1%（相同 RNG 种子下应完全一致）。

---

### Step 1.5 — 边界网格加载与 SamplePoint 生成

**目标**：从 OBJ 文件加载边界网格，自动生成域内采样点。

**改动文件**：
- 新建 `sycl/mesh_loader.h`：OBJ 加载 + 采样点生成
- 新建 `sycl/wos_sycl_demo.cpp`：完整 WoS SYCL 演示程序

**关键实现**：
```cpp
// mesh_loader.h — 复用 zombie 已有的 OBJ 加载
#include <zombie/utils/fcpw_geometric_queries.h>

template <int DIM>
struct MeshData {
    std::vector<zombie::Vector<DIM>> positions;
    std::vector<zombie::Vectori<DIM>> indices;

    void load(const std::string& objFile) {
        zombie::loadBoundaryMesh<DIM>(objFile, positions, indices);
        zombie::normalize<DIM>(positions);
    }
};
```

**验证方式**：加载 demo/ 目录下的测试网格，打印顶点数和面数。

**验证标准**：加载成功，顶点/面数与预期一致。

---

### Step 1.6 — 完整 WoS 单步执行对齐

**目标**：将 `execute_one_walkstate` 的每个阶段与原始 `walk_on_spheres.h` 对齐。

**改动文件**：
- `task_runtime.h`：重写 `execute_one_walkstate`

**阶段对齐细节**：

| 阶段 | 原始 Zombie (walk_on_spheres.h) | SYCL 实现 |
|------|------|------|
| P0 | 初始化 Green 函数 | 设置阶段标记，选择 Green 函数类型 |
| P1 | `computeDistToAbsorbingBoundary` | 读取 host 预计算的 `distToAbsorbingBoundary` |
| P2 | 采样方向 + `currentPt += dist * dir` | `sample_unit_direction` + 位置更新 |
| P3 | 检查 domain escape + Tikhonov | 域逃逸检测（当前简化为跳过） |
| P4 | source contribution + throughput | `G_norm * sourceValue` + `directionSampledPoissonKernel` |
| 终止 | `dist <= epsilon` 或 `walkLength > max` | 同原始逻辑 |

**Source Contribution 对齐**：
```cpp
// 原始: totalSourceContribution += throughput * greensFn->norm() * source(x)
// SYCL: totalSourceContribution += throughput * greens_fn_norm(R) * sourceValue
```

**Throughput 对齐**：
```cpp
// 原始: throughput *= greensFn->directionSampledPoissonKernel(currentPt)
// SYCL: throughput *= 1.0f  (Harmonic WoS 下恒为 1)
```

**Terminal Contribution 对齐**：
```cpp
// 原始: throughput * dirichlet(projectedPt)
// SYCL: throughput * dirichletValue (host 预计算)
```

**验证方式**：
- 构造 2D Laplace 方程，单位圆域，边界 u(x,y) = x
- 解析解 u(x,y) = x
- 在 (0.5, 0) 处估计，N=10000 条轨迹

**验证标准**：估计值与解析解 0.5 的相对误差 < 2%。

---

### Step 1.7 — 多 Walk 统计与 Welford 方差估计

**目标**：对每个采样点运行多条轨迹，用 Welford 算法在线计算均值和方差。

**改动文件**：
- 新建 `sycl/statistics.h`：Welford 在线统计
- `wos_sycl_demo.cpp`：集成多 walk 循环

**关键实现**：
```cpp
struct WelfordAccumulator {
    int count = 0;
    float mean = 0.0f;
    float M2 = 0.0f;

    void add(float x) {
        count++;
        float delta = x - mean;
        mean += delta / count;
        float delta2 = x - mean;
        M2 += delta * delta2;
    }

    float variance() const {
        return count > 1 ? M2 / (count - 1) : 0.0f;
    }
};
```

**验证方式**：
- 同 Step 1.6 的测试用例，N=10000
- 输出均值、方差、标准误差

**验证标准**：标准误差随 N 增大以 O(1/√N) 下降。

---

### Step 1.8 — Phase 1 集成验证：wos_sycl_baseline

**目标**：产出一个完整的 WoS SYCL 基线程序，作为后续创新点实现的参照。

**新建文件**：`sycl/wos_sycl_baseline.cpp`

**功能**：
1. 加载边界网格（或使用内置解析边界）
2. 生成域内采样点
3. Host-Device 迭代循环执行 WoS
4. 多 walk 统计
5. 输出：每个采样点的估计值、方差、walk 长度统计
6. 与 CPU Zombie 结果对比

**CMakeLists.txt 新增**：
```cmake
add_executable(wos_sycl_baseline sycl/wos_sycl_baseline.cpp)
target_link_libraries(wos_sycl_baseline PRIVATE zombie)
target_compile_options(wos_sycl_baseline PRIVATE -fsycl)
target_link_options(wos_sycl_baseline PRIVATE -fsycl)
```

**验证标准**：与 CPU Zombie 在相同问题上的估计值相对误差 < 1%。

---

## Phase 2：创新点1 — 自适应并行粒度的动态负载均衡调度（第4章）

> 目标：在 Phase 1 的 WoS 基线上实现任务块调度模型，
> 支持稳态粗粒度 → 尾部细粒度的自适应切换与未完成轨迹回收重封装。

### Step 2.1 — 任务队列模型与 WalkTask 封装

**目标**：实现 MPE 侧任务队列 Q，以 WalkTask 为最小调度单元。

**新建文件**：`sycl/task_queue.h`

**关键数据结构**：
```cpp
struct WalkTask {
    int taskId;
    int granularity;                    // B: 本任务块包含的轨迹数
    std::vector<WalkStateLite> states;  // 轨迹状态数组
    PhasePolicyConfig policy;           // 精度策略（只读）
    bool isRecycled;                    // 是否为回收重封装的任务块
};

struct TaskQueue {
    std::deque<WalkTask> queue;
    int totalSubmitted = 0;
    int totalCompleted = 0;

    void submit(WalkTask&& task);
    bool empty() const;
    WalkTask fetch();                   // CPE "空闲即取"
    float completionRatio() const;      // |completed| / |submitted|
};
```

**验证方式**：构造 100 个任务块入队，逐个取出执行，检查队列状态转换正确。

**验证标准**：所有任务块正确执行完毕，队列最终为空。

---

### Step 2.2 — 尾部检测与触发条件

**目标**：实现基于 |Q|/Nc 比值的尾部阶段检测。

**改动文件**：`sycl/task_queue.h`

**关键实现**：
```cpp
// 尾部判定：当队列剩余任务块数量与计算核数的比值低于阈值时进入尾部
struct TailDetector {
    int numComputeUnits;        // Nc: 计算核数（SYCL work-group 数）
    float tailThreshold = 2.0f; // 当 |Q|/Nc < threshold 时进入尾部

    bool isTailPhase(const TaskQueue& q) const {
        float ratio = static_cast<float>(q.queue.size()) / numComputeUnits;
        return ratio < tailThreshold;
    }
};
```

**验证方式**：模拟队列消耗过程，检查尾部触发时机是否符合预期。

**验证标准**：尾部触发时 |Q|/Nc 确实低于阈值。

---

### Step 2.3 — 动态粒度自适应：B 的调整策略

**目标**：实现稳态 B_max → 尾部 B → B_min 的粒度递减策略。

**改动文件**：
- `task_runtime.h`：增强 `update_granularity`
- `sycl/task_queue.h`：在任务提交时应用粒度策略

**关键实现**：
```cpp
struct GranularityPolicy {
    int B_init;     // 初始粒度
    int B_min;      // 最小粒度
    int B_max;      // 最大粒度

    // 稳态阶段：使用 B_init（通常等于 B_max）
    // 尾部阶段：根据未完成轨迹数量动态调整
    int computeGranularity(bool tailMode, int unfinishedCount) const {
        if (!tailMode) return B_init;
        if (unfinishedCount <= B_min) return B_min;
        // 尾部阶段：B = max(B_min, unfinishedCount / 分块数)
        int numChunks = (unfinishedCount + B_min - 1) / B_min;
        int B = unfinishedCount / std::max(1, numChunks);
        return std::clamp(B, B_min, B_max);
    }
};
```

**验证方式**：
- 运行 1000 条轨迹，观察粒度演化历史
- 输出每轮的 B 值、未完成数、尾部标志

**验证标准**：稳态阶段 B 保持不变；尾部阶段 B 逐步减小至 B_min。

---

### Step 2.4 — 未完成轨迹回收与重封装

**目标**：在任务块执行完毕后，提取未完成轨迹，重封装为新任务块入队。

**改动文件**：
- `task_runtime.h`：增强 `extract_unfinished`，保证 RNG 状态连续性
- `sycl/task_queue.h`：添加回收入队逻辑

**关键实现**：
```cpp
// 回收重封装：保持轨迹状态完整连续
std::vector<WalkTask> recycle_and_repack(
    const WalkTask& completedTask,
    const GranularityPolicy& policy,
    bool tailMode
) {
    // 1. 提取未完成轨迹（保持 rng/position/throughput 等状态不变）
    auto unfinished = extract_unfinished(completedTask.states);
    if (unfinished.empty()) return {};

    // 2. 计算新粒度
    int newB = policy.computeGranularity(tailMode, unfinished.size());

    // 3. 按新粒度分块
    std::vector<WalkTask> newTasks;
    for (size_t i = 0; i < unfinished.size(); i += newB) {
        WalkTask task;
        task.isRecycled = true;
        task.granularity = std::min(newB, (int)(unfinished.size() - i));
        task.states.assign(
            unfinished.begin() + i,
            unfinished.begin() + i + task.granularity);
        task.policy = completedTask.policy;
        newTasks.push_back(std::move(task));
    }
    return newTasks;
}
```

**验证方式**：
- 设置 maxWalkLength=10，B=64，观察回收次数和重封装后的任务块大小
- 检查回收前后轨迹的 rngState 和 currentPt 连续性

**验证标准**：回收后轨迹状态与回收前完全一致（rng、位置、throughput）。

---

### Step 2.5 — 完整调度循环集成

**目标**：将队列、尾部检测、粒度自适应、回收重封装集成为完整的调度循环。

**新建文件**：`sycl/wos_sycl_dg.cpp`（DG = Dynamic Granularity）

**执行流程**：
```
1. 初始化：生成所有采样点 → 按 B_init 分块入队
2. 主循环：
   while (!queue.empty()) {
       task = queue.fetch()
       host_update_distances(task, geometry)    // Host: FCPW 距离查询
       device_execute_one_step(q, task)         // Device: SYCL kernel
       stats = collect_stats(task)
       if (all_finished(task)) {
           accumulate_results(task)
       } else if (tail_detector.isTailPhase(queue)) {
           new_tasks = recycle_and_repack(task, policy, true)
           for (auto& t : new_tasks) queue.submit(t)
       } else {
           queue.submit(task)  // 未完成，整块重新入队
       }
   }
3. 输出：每个采样点的估计值、方差、调度统计
```

**验证方式**：
- 与 Phase 1 的 baseline 对比，相同问题相同 walk 数
- 输出调度统计：总任务块数、回收次数、尾部比例 ρ_tail

**验证标准**：
- 数值结果与 baseline 一致（相对误差 < 0.1%）
- 尾部比例 ρ_tail 相比无回收版本显著降低

---

### Step 2.6 — 调度性能指标采集

**目标**：实现论文第7章所需的调度性能指标。

**新建文件**：`sycl/metrics.h`

**指标定义**：
```cpp
struct SchedulingMetrics {
    // 吞吐 θ: 单位时间完成的轨迹数
    double throughput;

    // 加速比 S: θ_parallel / θ_serial
    double speedup;

    // 效率 E: S / Nc
    double efficiency;

    // 尾部比例 ρ_tail: T_tail / T_total
    double tailFraction;

    // 调度开销: 队列操作 + 回收重封装的时间占比
    double schedulingOverhead;

    // 粒度演化历史
    std::vector<int> granularityHistory;
};
```

**验证方式**：运行完整调度循环，输出所有指标。

**验证标准**：指标数值合理（θ > 0, 0 < E ≤ 1, 0 ≤ ρ_tail ≤ 1）。

---

## Phase 3：创新点2 — 阶段感知混合精度策略（第5章）

> 目标：在 WalkTask::execute 内部实现真实的 FP32/FP64 精度切换，
> 按阶段标志 p 驱动精度模式 s 的选择。

### Step 3.1 — FP64 算子实现（P1/P3/P4 高精度路径）

**目标**：为事件敏感阶段实现 FP64 版本的关键算子。

**新建文件**：`sycl/precision_ops.h`

**关键实现**：
```cpp
// P1: 几何距离计算（FP64）
inline double compute_geometry_distance_fp64(
    const double* currentPt, /* host 预计算的距离 */ double distA) {
    return distA;  // 距离本身由 host FCPW 以 FP64 计算
}

// P3: 近边界判定（FP64）
inline bool near_boundary_check_fp64(double dist, double epsilon) {
    return dist <= epsilon;
}

// P4: 终止贡献计算（FP64）
inline double compute_terminal_contribution_fp64(
    double throughput, double dirichletValue) {
    return throughput * dirichletValue;
}

// P4: Source contribution（FP64）
inline double compute_source_contribution_fp64(
    double throughput, double greensFnNorm, double sourceValue) {
    return throughput * greensFnNorm * sourceValue;
}
```

**验证方式**：对比 FP32 和 FP64 版本在近边界点的计算结果差异。

**验证标准**：FP64 版本在近边界（dist < 1e-4）时精度显著优于 FP32。

---

### Step 3.2 — FP32 算子实现（P2 低精度路径）

**目标**：为步进阶段实现 FP32 版本的算子。

**改动文件**：`sycl/precision_ops.h`

**关键实现**：
```cpp
// P2: 方向采样（FP32）
inline void sample_direction_fp32(PCG32State& rng,
                                   float& dx, float& dy) {
    float u = pcg32_next_float(rng);
    float theta = 2.0f * 3.14159265f * u;
    dx = sycl::cos(theta);
    dy = sycl::sin(theta);
}

// P2: 位置更新（FP32）
inline void step_position_fp32(float* pt, float radius,
                                float dx, float dy) {
    pt[0] += radius * dx;
    pt[1] += radius * dy;
}
```

**验证方式**：对比 FP32 步进与 FP64 步进在 1000 步后的位置偏差。

**验证标准**：FP32 步进的累积误差在可接受范围内（相对误差 < 0.1%）。

---

### Step 3.3 — 阶段化精度切换引擎

**目标**：在 `execute_one_walkstate` 中实现真实的阶段精度切换。

**改动文件**：
- `task_runtime.h`：重写 `execute_one_walkstate`
- `phase_policy.h`：增强精度映射

**关键实现**：
```cpp
inline void execute_one_walkstate_mixed_precision(
    WalkStateLite& s,
    const WalkSettings& settings,
    const PhasePolicyConfig& policy,
    bool enableMixedPrecision
) {
    if (s.term) return;

    // P0: 初始化（无精度要求）
    s.phase = P0_Init;

    // P1: 几何阶段
    if (enableMixedPrecision &&
        use_low_precision_for_p1(s.geometryDistance,
                                  s.p1LowPrecisionState, policy)) {
        // FP32 路径
        s.geometryDistance = static_cast<float>(s.distToAbsorbingBoundary);
        s.usedLowPrecisionP1 = 1;
    } else {
        // FP64 路径
        double distA_fp64 = static_cast<double>(s.distToAbsorbingBoundary);
        s.geometryDistance = static_cast<float>(distA_fp64);
        s.usedLowPrecisionP1 = 0;
    }
    s.phase = P1_Geometry;

    // P2: 步进阶段（始终 FP32）
    float stepRadius = s.distToAbsorbingBoundary;
    sample_direction_fp32(s.rng, dirX, dirY);
    step_position_fp32(s.currentPt.data(), stepRadius, dirX, dirY);
    s.walkLength++;
    s.phase = P2_Step;

    // P3: 近边界处理
    if (enableMixedPrecision) {
        // FP64 近边界判定
        // ...
    }
    s.phase = P3_NearBoundary;

    // P4: 贡献累积
    if (enableMixedPrecision) {
        // FP64 贡献计算
        double src_fp64 = compute_source_contribution_fp64(...);
        s.totalSourceContribution += static_cast<float>(src_fp64);
    } else {
        s.totalSourceContribution += compute_source_contribution_fp32(...);
    }
    s.phase = P4_Contribution;

    // 终止检查
    check_termination(s, settings);
}
```

**验证方式**：
- 同一问题分别用全 FP64、全 FP32、混合精度运行
- 对比三者的估计值和方差

**验证标准**：混合精度的误差接近全 FP64（相对误差差异 < 0.5%），性能接近全 FP32。

---

### Step 3.4 — 滞回机制完善：τ_in / τ_out 双阈值

**目标**：完善 P1 阶段的滞回精度切换，避免在边界附近频繁切换。

**改动文件**：`phase_policy.h`

**当前实现已有骨架**，需要完善：
1. 滞回状态在多步之间正确传递
2. 添加滞回切换次数统计
3. 支持从配置文件读取 τ_in / τ_out

**验证方式**：
- 构造一条轨迹在边界附近反复游走的场景
- 输出每步的精度选择和滞回状态

**验证标准**：在 (τ_out, τ_in) 区间内精度不发生切换。

---

### Step 3.5 — 混合精度统计与开销分解

**目标**：实现论文第7章所需的混合精度统计指标。

**改动文件**：`sycl/metrics.h`

**新增指标**：
```cpp
struct MixedPrecisionMetrics {
    // 各阶段执行次数
    int64_t p0Count, p1Count, p2Count, p3Count, p4Count;

    // P1 精度分布
    int64_t p1FP32Count, p1FP64Count;

    // 精度切换次数
    int64_t precisionSwitchCount;

    // 相对误差 ε_rel（与全 FP64 基线对比）
    double relativeError;

    // 各阶段耗时占比
    double p1TimeRatio, p2TimeRatio, p3TimeRatio, p4TimeRatio;
};
```

**验证方式**：运行混合精度版本，输出所有统计指标。

**验证标准**：指标数值合理，P1 FP32/FP64 比例与 τ_in/τ_out 设置一致。

---

### Step 3.6 — Phase 3 集成验证：wos_sycl_mp

**新建文件**：`sycl/wos_sycl_mp.cpp`（MP = Mixed Precision）

**功能**：在固定粒度 B 下运行混合精度 WoS，输出精度和性能指标。

**验证标准**：
- 相对误差 ε_rel < 1%（与全 FP64 对比）
- P2 阶段全部使用 FP32
- P1 在远离边界时使用 FP32，近边界时使用 FP64

---

## Phase 4：联合机制与实验框架（第6-7章）

> 目标：将动态粒度调度与混合精度联合，搭建完整的四组对照实验框架。

### Step 4.1 — 联合配置 (B, s) 系统

**目标**：实现论文表6-4的配置集合。

**新建文件**：`sycl/experiment_config.h`

**关键实现**：
```cpp
enum class ExperimentGroup {
    BASELINE,   // 全 FP64，固定 B，无回收
    DG_ONLY,    // 全 FP64，动态 B，有回收
    MP_ONLY,    // 混合精度，固定 B，无回收
    DG_MP       // 混合精度，动态 B，有回收
};

struct ExperimentConfig {
    ExperimentGroup group;
    int B;
    int B_min, B_max;
    bool enableRecycle;
    bool enableMixedPrecision;
    PhasePolicyConfig phasePolicy;
    int maxWalkLength;
    float epsilonShell;
    int nWalksPerPoint;

    static ExperimentConfig baseline(int B, int maxWalk, int nWalks);
    static ExperimentConfig dgOnly(int B, int Bmin, int Bmax, int maxWalk, int nWalks);
    static ExperimentConfig mpOnly(int B, int maxWalk, int nWalks, float tau_in, float tau_out);
    static ExperimentConfig dgMp(int B, int Bmin, int Bmax, int maxWalk, int nWalks,
                                  float tau_in, float tau_out);
};
```

**验证方式**：四种配置均能正确创建并打印参数。

---

### Step 4.2 — 算例 A：解析对照（2D Laplace 圆域）

**目标**：实现第一个测试算例，有解析解可对照。

**问题定义**：
- 2D Laplace 方程 Δu = 0
- 单位圆域 Ω = {(x,y) : x² + y² < 1}
- Dirichlet 边界条件 u(x,y) = x（在 ∂Ω 上）
- 解析解 u(x,y) = x

**新建文件**：`sycl/test_case_a.h`

**验证标准**：四组配置的估计值与解析解的相对误差均 < 2%。

---

### Step 4.3 — 算例 B：近边界高频（2D Poisson）

**目标**：测试近边界区域的精度敏感性。

**问题定义**：
- 2D Poisson 方程 Δu = f
- 源项 f(x,y) = -2（使得解析解 u = x² + y² - 1 在单位圆上为 0）
- 采样点分布：50% 在域内部，50% 在近边界（dist < 0.01）

**新建文件**：`sycl/test_case_b.h`

**验证标准**：
- 近边界点的混合精度误差不超过全 FP64 的 2 倍
- 动态粒度在近边界点密集区域的尾部比例显著降低

---

### Step 4.4 — 算例 C：复杂三维网格

**目标**：在 3D 网格上验证系统的完整性。

**问题定义**：
- 3D Laplace 方程
- 从 OBJ 文件加载边界网格
- Dirichlet 边界条件

**新建文件**：`sycl/test_case_c.h`

**验证标准**：3D 模式正常运行，结果与 CPU Zombie 一致。

---

### Step 4.5 — 四组对照实验驱动程序

**目标**：产出论文第7章实验的完整驱动程序。

**新建文件**：`sycl/experiment_runner.cpp`

**执行流程**：
```
for each test_case in [A, B, C]:
    for each config in [baseline, DG, MP, DG+MP]:
        results = run_experiment(test_case, config)
        metrics = collect_metrics(results)
        output_table(test_case, config, metrics)
```

**输出格式**：
```
| 算例 | 配置 | θ (轨迹/s) | S | E | ρ_tail | ε_rel |
|------|------|-----------|---|---|--------|-------|
| A    | BL   | ...       |...|...|  ...   | ...   |
| A    | DG   | ...       |...|...|  ...   | ...   |
| A    | MP   | ...       |...|...|  ...   | ...   |
| A    | DG+MP| ...       |...|...|  ...   | ...   |
```

**验证标准**：
- 所有配置均正常运行完毕
- DG 配置的 ρ_tail < baseline 的 ρ_tail
- MP 配置的 θ > baseline 的 θ
- DG+MP 配置同时获得两者的收益

---

## 开发顺序与里程碑

| 里程碑 | 包含 Steps | 产出物 | 预期验证 |
|--------|-----------|--------|---------|
| M1: 3D + RNG | 1.1, 1.2 | 模板化 WalkStateLite + pcg32 | 编译运行，2D/3D 均正常 |
| M2: 真实 WoS | 1.3, 1.4, 1.5, 1.6 | Green 函数 + FCPW 几何 + 网格加载 | 解析解对照误差 < 2% |
| M3: 基线完成 | 1.7, 1.8 | `wos_sycl_baseline` | 与 CPU Zombie 一致 |
| M4: 动态粒度 | 2.1-2.6 | `wos_sycl_dg` | 尾部比例显著降低 |
| M5: 混合精度 | 3.1-3.6 | `wos_sycl_mp` | 误差可控，性能提升 |
| M6: 联合实验 | 4.1-4.5 | `experiment_runner` | 四组对照完整输出 |

## 编译与运行速查

```bash
# 环境初始化
source /home/tools/intel/oneapi/setvars.sh

# 构建
cd /home/powerzhang/zombie/build_icpx
cmake .. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) <target>

# 运行示例
./wos_sycl_baseline          # Phase 1 基线
./wos_sycl_dg                # Phase 2 动态粒度
./wos_sycl_mp                # Phase 3 混合精度
./experiment_runner           # Phase 4 完整实验
```

## 文件清单（新增/修改）

| 文件 | Phase | 类型 | 说明 |
|------|-------|------|------|
| `pcg32_device.h` | 1 | 新建 | 设备侧 PCG32 RNG |
| `greens_function_device.h` | 1 | 新建 | 设备侧 Green 函数 |
| `geometry_host.h` | 1 | 新建 | Host 侧 FCPW 封装 |
| `mesh_loader.h` | 1 | 新建 | OBJ 网格加载 |
| `statistics.h` | 1 | 新建 | Welford 在线统计 |
| `wos_sycl_baseline.cpp` | 1 | 新建 | WoS 基线程序 |
| `walkstate_bridge.h` | 1 | 修改 | 3D 支持 + PCG32 状态 |
| `task_runtime.h` | 1-3 | 修改 | 真实算子 + 混合精度 |
| `phase_policy.h` | 3 | 修改 | 滞回完善 |
| `task_init.h` | 1 | 修改 | 网格初始化 |
| `task_queue.h` | 2 | 新建 | 任务队列模型 |
| `metrics.h` | 2-3 | 新建 | 性能指标采集 |
| `precision_ops.h` | 3 | 新建 | FP32/FP64 算子 |
| `wos_sycl_dg.cpp` | 2 | 新建 | 动态粒度版本 |
| `wos_sycl_mp.cpp` | 3 | 新建 | 混合精度版本 |
| `experiment_config.h` | 4 | 新建 | 实验配置 |
| `test_case_a.h` | 4 | 新建 | 算例 A |
| `test_case_b.h` | 4 | 新建 | 算例 B |
| `test_case_c.h` | 4 | 新建 | 算例 C |
| `experiment_runner.cpp` | 4 | 新建 | 实验驱动程序 |
| `CMakeLists.txt` | 1-4 | 修改 | 新增编译目标 |
