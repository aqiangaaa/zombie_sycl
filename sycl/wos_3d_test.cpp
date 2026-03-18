#include <sycl/sycl.hpp>

// 3D 模式：编译期切换
#define ZOMBIE_SYCL_DIM 3

#include <zombie/zombie.h>
#include "walkstate_bridge.h"
#include "phase_policy.h"
#include "task_runtime.h"
#include "task_init.h"
#include "geometry_host.h"
#include "batch_solver.h"
#include "statistics.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// ============================================================
// wos_3d_test.cpp
//
// Step 8.1: 3D 单位球上的 WoS 验证
//
// 问题：3D Laplace Δu = 0，单位球域
// 边界条件：u(x,y,z) = x on ∂Ω
// 解析解：u(x,y,z) = x
// ============================================================

// 3D 单位球解析几何
struct AnalyticUnitSphere3D {
    float distToAbsorbing(const zombie::Vector<3>& pt) const {
        return std::max(0.0f, 1.0f - static_cast<float>(pt.norm()));
    }

    bool projectToAbsorbing(zombie::Vector<3>& pt,
                            zombie::Vector<3>& normal,
                            float& dist) const {
        float norm = static_cast<float>(pt.norm());
        if (norm < 1e-8f) {
            normal = zombie::Vector<3>::Zero();
            normal[0] = 1.0f;
            pt = normal;
            dist = 1.0f;
            return true;
        }
        normal = pt.normalized();
        pt = normal;
        dist = 1.0f - norm;
        return true;
    }
};

int main() {
    try {
        sycl::queue q;
        std::cout << "Device: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << "\n";
        std::cout << "DIM = " << DIM << "\n\n";

        // ---- PDE: Laplace u=x ----
        zombie::PDE<float, DIM> pde;
        pde.source = [](const zombie::Vector<DIM>&) -> float { return 0.0f; };
        pde.dirichlet = [](const zombie::Vector<DIM>& x, bool) -> float { return x[0]; };
        pde.hasReflectingBoundaryConditions = [](const zombie::Vector<DIM>&) { return false; };
        pde.robin = [](const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool) -> float { return 0.0f; };
        pde.robinCoeff = [](const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool) -> float { return 0.0f; };

        zombie::WalkSettings settings(1e-3f, 1e-3f, 128, false);
        PhasePolicyConfig policy;
        AnalyticUnitSphere3D geom;

        // ---- 测试点 ----
        struct TP { float x, y, z, analytic; };
        std::vector<TP> testPoints = {
            { 0.5f,  0.0f,  0.0f,  0.5f},
            {-0.5f,  0.0f,  0.0f, -0.5f},
            { 0.3f,  0.3f,  0.0f,  0.3f},
            { 0.7f,  0.0f,  0.0f,  0.7f},
            { 0.0f,  0.0f,  0.5f,  0.0f},
            { 0.3f,  0.3f,  0.3f,  0.3f},
        };

        const int nWalks = 1024;

        std::cout << "============ 3D WoS on Unit Sphere ============\n";
        std::cout << "Walks per point: " << nWalks << "\n\n";

        for (const auto& tp : testPoints) {
            zombie::Vector<DIM> pt;
            pt[0] = tp.x; pt[1] = tp.y; pt[2] = tp.z;

            double ptTime;
            auto est = batch_solve_single_point(
                q, pt, tp.analytic, geom, pde, settings, policy,
                false, nWalks, &ptTime);

            std::cout << "  pt=(" << tp.x << "," << tp.y << "," << tp.z << ")"
                      << "  analytic=" << std::fixed << std::setprecision(3) << tp.analytic
                      << "  estimate=" << est.solution.mean
                      << "  stdErr=" << est.solution.standardError()
                      << "  relErr=" << std::setprecision(1) << (est.relativeError() * 100.0f) << "%"
                      << "  avgLen=" << est.walkLen.mean
                      << "  time=" << std::setprecision(3) << ptTime << "s"
                      << "\n";
        }

        std::cout << "\n============ Done ============\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
