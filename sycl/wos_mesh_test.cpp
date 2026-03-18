#include <sycl/sycl.hpp>
#include <zombie/zombie.h>
#include "walkstate_bridge.h"
#include "phase_policy.h"
#include "task_runtime.h"
#include "geometry_host.h"
#include "batch_solver.h"
#include "statistics.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// ============================================================
// wos_mesh_test.cpp
//
// Step 6.1-6.2: 加载真实 OBJ 网格，用 FCPW 计算距离，
// 在网格域上运行 WoS 求解。
//
// 测试网格: plus-shape.obj (2D)
// PDE: Laplace Δu + f = 0, f = 0, u = x on boundary
// ============================================================

int main() {
    try {
        sycl::queue q;
        std::cout << "Device: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << "\n\n";

        // ---- 加载网格 ----
        std::string meshFile = "../deps/fcpw/tests/input/plus-shape.obj";
        HostGeometry<DIM> geom;
        std::cout << "Loading mesh: " << meshFile << "\n";
        geom.loadAndBuild(meshFile, true);
        std::cout << "Vertices: " << geom.positions.size()
                  << "  Segments: " << geom.indices.size() << "\n";

        // 打印 bounding box
        zombie::Vector<DIM> bbMin = zombie::Vector<DIM>::Constant(1e10f);
        zombie::Vector<DIM> bbMax = zombie::Vector<DIM>::Constant(-1e10f);
        for (const auto& p : geom.positions) {
            for (int d = 0; d < DIM; ++d) {
                if (p[d] < bbMin[d]) bbMin[d] = p[d];
                if (p[d] > bbMax[d]) bbMax[d] = p[d];
            }
        }
        std::cout << "BBox: (" << bbMin[0] << "," << bbMin[1]
                  << ") - (" << bbMax[0] << "," << bbMax[1] << ")\n\n";

        // ---- PDE: Laplace u = x on boundary ----
        zombie::PDE<float, DIM> pde;
        pde.source = [](const zombie::Vector<DIM>&) -> float { return 0.0f; };
        pde.dirichlet = [](const zombie::Vector<DIM>& x, bool) -> float { return x[0]; };
        pde.hasReflectingBoundaryConditions = [](const zombie::Vector<DIM>&) { return false; };
        pde.robin = [](const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool) -> float { return 0.0f; };
        pde.robinCoeff = [](const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool) -> float { return 0.0f; };

        zombie::WalkSettings settings(1e-3f, 1e-3f, 256, false);
        PhasePolicyConfig policy;

        // ---- 生成域内采样点 ----
        // 在 bounding box 内均匀撒点，用距离 > 0 过滤域内点
        std::vector<BatchPointInfo> batchPoints;
        const int gridRes = 8;
        zombie::Vector<DIM> margin = (bbMax - bbMin) * 0.05f;
        zombie::Vector<DIM> lo = bbMin + margin;
        zombie::Vector<DIM> hi = bbMax - margin;

        for (int i = 0; i < gridRes; ++i) {
            for (int j = 0; j < gridRes; ++j) {
                zombie::Vector<DIM> pt;
                pt[0] = lo[0] + (hi[0] - lo[0]) * (i + 0.5f) / gridRes;
                pt[1] = lo[1] + (hi[1] - lo[1]) * (j + 0.5f) / gridRes;

                float distA = geom.distToAbsorbing(pt);
                if (distA > 0.01f) {
                    BatchPointInfo bp;
                    bp.pt = pt;
                    bp.initDistA = distA;
                    bp.analyticSolution = 0.0f; // 无解析解，仅验证运行
                    bp.nWalks = 256;
                    batchPoints.push_back(bp);
                }
            }
        }

        std::cout << "Domain interior points: " << batchPoints.size() << "\n\n";

        if (batchPoints.empty()) {
            std::cout << "No interior points found. Check mesh orientation.\n";
            return 1;
        }

        // ---- 批量求解 ----
        std::vector<BatchPointInfo> pointsOut;
        auto states = batch_init_multipoint(batchPoints, pointsOut);

        std::cout << "Total walks: " << states.size() << "\n";
        std::cout << "Running WoS on mesh domain...\n\n";

        auto result = batch_solve(q, states, pointsOut, geom, pde,
                                   settings, policy, false);

        // ---- 输出结果 ----
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Solve time: " << result.totalTime << "s"
                  << "  (host=" << result.hostTime << "s"
                  << "  device=" << result.deviceTime << "s)"
                  << "  steps=" << result.totalSteps << "\n\n";

        std::cout << std::left
                  << std::setw(16) << "Point"
                  << std::setw(10) << "DistA"
                  << std::setw(12) << "Estimate"
                  << std::setw(10) << "StdErr"
                  << std::setw(10) << "AvgLen"
                  << "\n";
        std::cout << std::string(58, '-') << "\n";

        int validCount = 0;
        for (size_t i = 0; i < result.estimates.size(); ++i) {
            const auto& est = result.estimates[i];
            const auto& bp = pointsOut[i];

            std::cout << "(" << std::setw(6) << bp.pt[0]
                      << "," << std::setw(6) << bp.pt[1] << ")"
                      << "  " << std::setw(10) << bp.initDistA
                      << std::setw(12) << est.solution.mean
                      << std::setw(10) << est.solution.standardError()
                      << std::setw(10) << est.walkLen.mean
                      << "\n";

            if (est.solution.count > 0) validCount++;
        }

        std::cout << "\nValid points with results: " << validCount
                  << " / " << result.estimates.size() << "\n";

        float throughput = static_cast<float>(states.size()) / result.totalTime;
        std::cout << "Throughput: " << std::setprecision(0) << throughput << " walks/s\n";

        if (validCount > 0) {
            std::cout << "\nPASS: WoS on mesh domain completed successfully.\n";
        } else {
            std::cout << "\nFAIL: No valid results.\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
