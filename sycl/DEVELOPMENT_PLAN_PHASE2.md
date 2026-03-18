# Zombie SYCL 第二阶段开发计划

## 前置状态

Phase 1-4 已完成，当前具备：
- WoS 核心算法 SYCL 真实化（Green 函数、PCG32、host-device 迭代）
- 动态粒度调度（task_queue.h、尾部检测、回收重封装）
- 阶段感知混合精度（precision_ops.h、P0-P4 精度切换、滞回机制）
- 四组对照实验框架（experiment_runner.cpp、两个解析算例）

所有程序在 RTX 3090 + Intel SYCL 上编译运行通过。

## 待推进方向

按优先级排序，每个 Phase 内部按依赖关系排列。

---

## Phase 5：批量并行化 — 提升 GPU 利用率

> 当前瓶颈：每条 walk 单独提交一个 SYCL kernel（1 个 work-item），
> GPU 利用率极低。需要改为一次 kernel 批量执行多条 walk。

### Step 5.1 — 批量 kernel：单步多 walk 并行

**目标**：一次 kernel 调用同时推进 N 条 walk 的一步。

**改动文件**：`task_runtime.h`

**关键变化**：
- `run_task_once` 已经支持 `parallel_for(N)`，但当前外层循环是逐 walk 串行
- 改为：对同一采样点的所有 walk 打包成一个大 buffer，一次 kernel 推进一步
- host-device 循环变为：host 批量更新距离 → device 批量推进一步 → 循环

**验证标准**：与逐 walk 版本估计值一致，吞吐提升 10x 以上。

---

### Step 5.2 — 多采样点批量执行

**目标**：多个采样点的 walk 混合在同一个 buffer 中并行执行。

**改动文件**：
- 新建 `sycl/batch_solver.h`：批量求解器
- `experiment_runner.cpp`：使用批量求解器

**关键设计**：
```
buffer 布局: [pt0_walk0, pt0_walk1, ..., pt0_walkN, pt1_walk0, ..., ptM_walkN]
kernel: parallel_for(M * N)，每个 work-item 推进一条 walk 一步
host 循环: 批量更新距离 → 批量 kernel → 检查终止 → 循环
```

**验证标准**：结果与逐点版本一致，端到端时间显著缩短。

---

### Step 5.3 — 性能计时框架

**目标**：精确测量各阶段耗时，输出吞吐 θ 和加速比 S。

**改动文件**：`metrics.h`、`experiment_runner.cpp`

**测量项**：
- host 距离计算时间
- device kernel 执行时间
- host-device 数据传输时间
- 总端到端时间
- 吞吐 θ = 总完成 walk 数 / 总时间

**验证标准**：批量版本的 θ 显著高于逐 walk 版本。

---

## Phase 6：真实网格几何 — 接入 FCPW

> 当前使用解析单位圆几何，需要接入真实 OBJ 网格。

### Step 6.1 — FCPW 网格加载与距离查询

**目标**：用 `HostGeometry<DIM>` 加载 OBJ 文件，替代 `AnalyticUnitSphereGeometry`。

**改动文件**：
- `geometry_host.h`（已有 `HostGeometry`，需验证）
- 新建 `sycl/wos_mesh_test.cpp`：网格几何验证程序

**测试网格**：
- 2D: `deps/fcpw/tests/input/plus-shape.obj`
- 3D: `deps/fcpw/tests/input/bunny.obj`

**验证标准**：网格加载成功，域内点距离查询返回正值。

---

### Step 6.2 — 网格域上的完整 WoS 求解

**目标**：在真实网格域上运行 WoS，与 CPU Zombie 结果对比。

**改动文件**：
- 新建 `sycl/wos_mesh_solve.cpp`

**验证标准**：与 CPU Zombie 在相同网格、相同 PDE 上的估计值相对误差 < 2%。

---

### Step 6.3 — 终止时边界投影

**目标**：walk 终止时用 FCPW 投影到最近边界点，获取精确的 Dirichlet 值。

**改动文件**：`geometry_host.h`（`host_update_terminal_contributions` 已有骨架）

**验证标准**：投影后的边界点确实在网格面上，法向正确。

---

## Phase 7：长尾场景 — 验证动态粒度回收效果

> 当前解析圆域上所有 walk 步数相近，回收未被触发。
> 需要构造长尾场景让回收机制真正发挥作用。

### Step 7.1 — 构造长尾几何场景

**目标**：设计一个几何域，使部分采样点的 walk 步数远大于其他点。

**方案**：
- 窄通道几何：域包含一个细长通道，通道内的 walk 需要大量步数才能到达边界
- 或使用 `plus-shape.obj`，其凹角区域会产生长尾 walk

**新建文件**：`sycl/test_longtail.cpp`

**验证标准**：walk 步数分布呈现明显长尾（P99/P50 > 5）。

---

### Step 7.2 — 回收触发验证

**目标**：在长尾场景下验证回收机制被正确触发。

**验证标准**：
- `totalRecycles > 0`
- 回收后粒度 B 从 B_init 降到 B_min
- 回收前后轨迹状态连续（rng、position 不变）

---

### Step 7.3 — DG vs Baseline 尾部比例对比

**目标**：量化动态粒度调度对尾部比例 ρ_tail 的改善。

**验证标准**：DG 的 ρ_tail < Baseline 的 ρ_tail。

---

## Phase 8：3D 模式实测

> 代码已支持 3D（`ZOMBIE_SYCL_DIM=3`），但尚未实测。

### Step 8.1 — 3D 编译与基础验证

**目标**：以 `ZOMBIE_SYCL_DIM=3` 编译，在 3D 单位球上运行 WoS。

**改动文件**：`CMakeLists.txt`（添加 3D 编译目标）

**验证标准**：3D Laplace `u=x` 在 `(0.5,0,0)` 处估计值 ≈ 0.5。

---

### Step 8.2 — 3D 网格求解

**目标**：加载 3D OBJ 网格（bunny），运行 WoS。

**验证标准**：程序正常运行，输出合理的估计值。

---

## Phase 9：实验数据完善 — 对齐论文第7章

> 补全论文实验所需的全部数据。

### Step 9.1 — 参数扫描：B 的敏感性

**目标**：扫描 B ∈ {4, 8, 16, 32, 64, 128}，输出 θ、ρ_tail、ε_rel。

---

### Step 9.2 — 参数扫描：τ_in/τ_out 的敏感性

**目标**：扫描多组 (τ_in, τ_out)，输出 FP64 覆盖比例与误差。

---

### Step 9.3 — 开销分解

**目标**：将端到端时间分解为 T_wait + T_exec + T_commit + T_recycle + T_switch。

---

### Step 9.4 — 结果输出为 CSV

**目标**：所有实验结果输出为 CSV 格式，便于绘图。

---

## 开发顺序与里程碑

| 里程碑 | 包含 Steps | 核心产出 | 预期效果 |
|--------|-----------|---------|---------|
| M7: 批量并行 | 5.1-5.3 | batch_solver.h | 吞吐提升 10x+ |
| M8: 真实网格 | 6.1-6.3 | wos_mesh_solve.cpp | OBJ 网格上 WoS 跑通 |
| M9: 长尾验证 | 7.1-7.3 | test_longtail.cpp | 回收触发，ρ_tail 降低 |
| M10: 3D 实测 | 8.1-8.2 | 3D 编译目标 | 3D 模式跑通 |
| M11: 实验数据 | 9.1-9.4 | CSV 输出 | 论文第7章数据完备 |

## 编译与运行速查

```bash
source /home/tools/intel/oneapi/setvars.sh
export LD_LIBRARY_PATH=/home/tools/intel/oneapi/2025.2/lib:$LD_LIBRARY_PATH
cd /home/powerzhang/zombie/build_icpx
cmake .. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) <target>
```
