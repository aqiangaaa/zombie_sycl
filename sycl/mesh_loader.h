#pragma once

// ============================================================
// 网格加载与域内采样点生成
//
// 提供：
// 1. 从 OBJ 文件加载 2D/3D 边界网格（复用 zombie 已有接口）
// 2. 在域内生成均匀网格采样点
// 3. 在单位圆/球内生成随机采样点
// ============================================================

#include <zombie/zombie.h>
#include "walkstate_bridge.h"
#include "pcg32_device.h"
#include "geometry_host.h"

#include <vector>
#include <string>
#include <cmath>

// ============================================================
// 在单位圆/球内生成均匀网格采样点
//
// 2D: 在 [-1,1]² 网格上取落在单位圆内的点
// 3D: 在 [-1,1]³ 网格上取落在单位球内的点
// ============================================================
template <int D>
inline std::vector<zombie::SamplePoint<float, D>>
generate_grid_sample_points_unit_sphere(int gridRes, float shrink = 0.95f) {
    std::vector<zombie::SamplePoint<float, D>> points;

    if constexpr (D == 2) {
        for (int i = 0; i < gridRes; ++i) {
            for (int j = 0; j < gridRes; ++j) {
                float x = -shrink + 2.0f * shrink * (i + 0.5f) / gridRes;
                float y = -shrink + 2.0f * shrink * (j + 0.5f) / gridRes;
                float r2 = x * x + y * y;
                if (r2 < shrink * shrink) {
                    zombie::Vector<D> pt;
                    pt[0] = x;
                    pt[1] = y;
                    zombie::Vector<D> normal = zombie::Vector<D>::Zero();
                    float distA = 1.0f - std::sqrt(r2);
                    float distR = distA; // 无反射边界时与 distA 相同
                    points.emplace_back(
                        pt, normal,
                        zombie::SampleType::InDomain,
                        zombie::EstimationQuantity::Solution,
                        1.0f, distA, distR);
                }
            }
        }
    } else {
        for (int i = 0; i < gridRes; ++i) {
            for (int j = 0; j < gridRes; ++j) {
                for (int k = 0; k < gridRes; ++k) {
                    float x = -shrink + 2.0f * shrink * (i + 0.5f) / gridRes;
                    float y = -shrink + 2.0f * shrink * (j + 0.5f) / gridRes;
                    float z = -shrink + 2.0f * shrink * (k + 0.5f) / gridRes;
                    float r2 = x * x + y * y + z * z;
                    if (r2 < shrink * shrink) {
                        zombie::Vector<D> pt;
                        pt[0] = x;
                        pt[1] = y;
                        pt[2] = z;
                        zombie::Vector<D> normal = zombie::Vector<D>::Zero();
                        float distA = 1.0f - std::sqrt(r2);
                        float distR = distA;
                        points.emplace_back(
                            pt, normal,
                            zombie::SampleType::InDomain,
                            zombie::EstimationQuantity::Solution,
                            1.0f, distA, distR);
                    }
                }
            }
        }
    }

    return points;
}

// ============================================================
// 在 FCPW 网格域内生成采样点
//
// 在 bounding box 内均匀撒点，用 insideDomain 过滤。
// 对每个域内点用 FCPW 计算到吸收边界的距离。
// ============================================================
template <int D>
inline std::vector<zombie::SamplePoint<float, D>>
generate_grid_sample_points_mesh(const HostGeometry<D>& geom, int gridRes) {
    std::vector<zombie::SamplePoint<float, D>> points;

    // 计算 bounding box
    zombie::Vector<D> bbMin = zombie::Vector<D>::Constant(1e10f);
    zombie::Vector<D> bbMax = zombie::Vector<D>::Constant(-1e10f);
    for (const auto& p : geom.positions) {
        for (int d = 0; d < D; ++d) {
            if (p[d] < bbMin[d]) bbMin[d] = p[d];
            if (p[d] > bbMax[d]) bbMax[d] = p[d];
        }
    }

    // 稍微收缩避免边界上的点
    zombie::Vector<D> margin = (bbMax - bbMin) * 0.02f;
    bbMin += margin;
    bbMax -= margin;

    if constexpr (D == 2) {
        for (int i = 0; i < gridRes; ++i) {
            for (int j = 0; j < gridRes; ++j) {
                zombie::Vector<D> pt;
                pt[0] = bbMin[0] + (bbMax[0] - bbMin[0]) * (i + 0.5f) / gridRes;
                pt[1] = bbMin[1] + (bbMax[1] - bbMin[1]) * (j + 0.5f) / gridRes;

                // 检查是否在域内
                if (geom.queries.insideDomain && geom.queries.insideDomain(pt)) {
                    float distA = geom.distToAbsorbing(pt);
                    if (distA > 0.0f) {
                        zombie::Vector<D> normal = zombie::Vector<D>::Zero();
                        points.emplace_back(
                            pt, normal,
                            zombie::SampleType::InDomain,
                            zombie::EstimationQuantity::Solution,
                            1.0f, distA, distA);
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < gridRes; ++i) {
            for (int j = 0; j < gridRes; ++j) {
                for (int k = 0; k < gridRes; ++k) {
                    zombie::Vector<D> pt;
                    pt[0] = bbMin[0] + (bbMax[0] - bbMin[0]) * (i + 0.5f) / gridRes;
                    pt[1] = bbMin[1] + (bbMax[1] - bbMin[1]) * (j + 0.5f) / gridRes;
                    pt[2] = bbMin[2] + (bbMax[2] - bbMin[2]) * (k + 0.5f) / gridRes;

                    if (geom.queries.insideDomain && geom.queries.insideDomain(pt)) {
                        float distA = geom.distToAbsorbing(pt);
                        if (distA > 0.0f) {
                            zombie::Vector<D> normal = zombie::Vector<D>::Zero();
                            points.emplace_back(
                                pt, normal,
                                zombie::SampleType::InDomain,
                                zombie::EstimationQuantity::Solution,
                                1.0f, distA, distA);
                        }
                    }
                }
            }
        }
    }

    return points;
}

// ============================================================
// 生成单个指定位置的采样点（用于单点验证）
// ============================================================
template <int D>
inline zombie::SamplePoint<float, D>
make_single_sample_point(const zombie::Vector<D>& pt, float distA) {
    zombie::Vector<D> normal = zombie::Vector<D>::Zero();
    return zombie::SamplePoint<float, D>(
        pt, normal,
        zombie::SampleType::InDomain,
        zombie::EstimationQuantity::Solution,
        1.0f, distA, distA);
}
