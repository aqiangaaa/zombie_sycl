#pragma once

// ============================================================
// Host 侧几何查询封装
//
// 封装 FCPW 加速结构，提供：
// 1. 从 OBJ 文件或顶点/索引构建加速结构
// 2. 计算任意点到吸收边界的距离
// 3. 投影到最近边界点（用于终止时获取边界位置）
// 4. 批量更新 WalkStateLite 数组中的边界距离
//
// FCPW 只能在 host 侧运行，不进入 SYCL kernel。
// ============================================================

#include <zombie/zombie.h>
#include <zombie/utils/fcpw_geometric_queries.h>
#include "walkstate_bridge.h"

#include <vector>
#include <string>

// ============================================================
// Host 侧几何数据与查询
// ============================================================
template <int D>
struct HostGeometry {
    zombie::FcpwDirichletBoundaryHandler<D> dirichletHandler;
    zombie::GeometricQueries<D> queries;

    std::vector<zombie::Vector<D>> positions;
    std::vector<zombie::Vectori<D>> indices;

    bool isBuilt = false;

    // 从 OBJ 文件加载并构建加速结构
    void loadAndBuild(const std::string& objFile, bool normalizePositions = true) {
        zombie::loadBoundaryMesh<D>(objFile, positions, indices);
        if (normalizePositions) {
            zombie::normalize<D>(positions);
        }
        build();
    }

    // 从已有顶点/索引构建加速结构
    void build() {
        dirichletHandler.buildAccelerationStructure(positions, indices, true, false);
        zombie::populateGeometricQueriesForDirichletBoundary<D>(dirichletHandler, queries);
        isBuilt = true;
    }

    // 计算单点到吸收边界的距离
    float distToAbsorbing(const zombie::Vector<D>& pt) const {
        return queries.computeDistToAbsorbingBoundary(pt, false);
    }

    // 投影到最近吸收边界点，返回是否成功
    bool projectToAbsorbing(zombie::Vector<D>& pt,
                            zombie::Vector<D>& normal,
                            float& dist) const {
        return queries.projectToAbsorbingBoundary(pt, normal, dist, false);
    }
};

// ============================================================
// 批量更新 WalkStateLite 数组中的边界距离
//
// 在每一步 device kernel 执行前调用：
// 1. 从 WalkStateLite 读取当前位置
// 2. 用 FCPW 计算到吸收边界的真实距离
// 3. 写回 distToAbsorbingBoundary
// ============================================================
template <int D>
inline void host_update_boundary_distances(
    std::vector<WalkStateLite>& states,
    const HostGeometry<D>& geom
) {
    for (auto& s : states) {
        if (s.term) continue;

        zombie::Vector<D> pt = zombie::Vector<D>::Zero();
        for (int d = 0; d < D; ++d) {
            pt[d] = s.currentPt[d];
        }

        s.distToAbsorbingBoundary = geom.distToAbsorbing(pt);
        s.geometryDistance = s.distToAbsorbingBoundary;
    }
}

// ============================================================
// 在轨迹终止时，投影到边界并更新 Dirichlet 值
//
// 对已终止的轨迹：
// 1. 投影当前位置到最近吸收边界
// 2. 用 PDE 回调计算该边界点的 Dirichlet 值
// 3. 重新计算 terminal contribution
// ============================================================
template <int D>
inline void host_update_terminal_contributions(
    std::vector<WalkStateLite>& states,
    const HostGeometry<D>& geom,
    const zombie::PDE<float, D>& pde
) {
    for (auto& s : states) {
        if (!s.term) continue;
        if (s.completionCode == WALK_TERMINATED_BY_POSITION_RULE) {
            // 投影到最近边界点
            zombie::Vector<D> pt = zombie::Vector<D>::Zero();
            zombie::Vector<D> normal = zombie::Vector<D>::Zero();
            float dist = 0.0f;
            for (int d = 0; d < D; ++d) {
                pt[d] = s.currentPt[d];
            }

            if (geom.projectToAbsorbing(pt, normal, dist)) {
                // 用投影后的边界点计算 Dirichlet 值
                if (pde.dirichlet) {
                    s.dirichletValue = pde.dirichlet(pt, false);
                }
            }
        }
    }
}

// ============================================================
// 用于无真实网格时的解析几何：单位圆/球
//
// 提供一个不依赖 FCPW 的简单距离函数，
// 用于解析对照实验（算例 A）。
// ============================================================
template <int D>
struct AnalyticUnitSphereGeometry {
    // 到单位球边界的距离：1 - |x|
    float distToAbsorbing(const zombie::Vector<D>& pt) const {
        return 1.0f - static_cast<float>(pt.norm());
    }

    // 投影到单位球面
    bool projectToAbsorbing(zombie::Vector<D>& pt,
                            zombie::Vector<D>& normal,
                            float& dist) const {
        float norm = static_cast<float>(pt.norm());
        if (norm < 1e-8f) {
            // 在原点，任意方向投影
            normal = zombie::Vector<D>::Zero();
            normal[0] = 1.0f;
            pt = normal;
            dist = 1.0f;
            return true;
        }
        normal = pt.normalized();
        pt = normal; // 投影到单位球面
        dist = 1.0f - norm;
        return true;
    }
};

// 解析几何版本的批量距离更新
template <int D>
inline void host_update_boundary_distances_analytic(
    std::vector<WalkStateLite>& states,
    const AnalyticUnitSphereGeometry<D>& geom
) {
    for (auto& s : states) {
        if (s.term) continue;

        zombie::Vector<D> pt = zombie::Vector<D>::Zero();
        for (int d = 0; d < D; ++d) {
            pt[d] = s.currentPt[d];
        }

        float dist = geom.distToAbsorbing(pt);
        // 确保距离非负（轨迹可能逃逸到域外）
        s.distToAbsorbingBoundary = std::max(0.0f, dist);
        s.geometryDistance = s.distToAbsorbingBoundary;
    }
}

// 解析几何版本的终止贡献更新
template <int D>
inline void host_update_terminal_contributions_analytic(
    std::vector<WalkStateLite>& states,
    const AnalyticUnitSphereGeometry<D>& geom,
    const zombie::PDE<float, D>& pde
) {
    for (auto& s : states) {
        if (!s.term) continue;
        if (s.completionCode == WALK_TERMINATED_BY_POSITION_RULE) {
            zombie::Vector<D> pt = zombie::Vector<D>::Zero();
            zombie::Vector<D> normal = zombie::Vector<D>::Zero();
            float dist = 0.0f;
            for (int d = 0; d < D; ++d) {
                pt[d] = s.currentPt[d];
            }

            if (geom.projectToAbsorbing(pt, normal, dist)) {
                if (pde.dirichlet) {
                    s.dirichletValue = pde.dirichlet(pt, false);
                }
            }
        }
    }
}
