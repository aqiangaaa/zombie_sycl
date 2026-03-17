#include <sycl/sycl.hpp>
#include <zombie/zombie.h>
#include <array>
#include <iostream>
#include <vector>

using Real = float;
constexpr int DIM = 2;

struct WalkStateLite {
    float totalReflectingBoundaryContribution;
    float totalSourceContribution;
    std::array<float, 2> currentPt;
    std::array<float, 2> currentNormal;
    std::array<float, 2> prevDirection;
    float prevDistance;
    float throughput;
    int walkLength;
    int onReflectingBoundary;
};

static WalkStateLite packState(const zombie::WalkState<Real, DIM>& s) {
    WalkStateLite out{};
    out.totalReflectingBoundaryContribution = s.totalReflectingBoundaryContribution;
    out.totalSourceContribution = s.totalSourceContribution;
    out.currentPt = {s.currentPt[0], s.currentPt[1]};
    out.currentNormal = {s.currentNormal[0], s.currentNormal[1]};
    out.prevDirection = {s.prevDirection[0], s.prevDirection[1]};
    out.prevDistance = s.prevDistance;
    out.throughput = s.throughput;
    out.walkLength = s.walkLength;
    out.onReflectingBoundary = s.onReflectingBoundary ? 1 : 0;
    return out;
}

static void unpackState(const WalkStateLite& in, zombie::WalkState<Real, DIM>& s) {
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

int main() {
    try {
        sycl::queue q;
        std::cout << "Device: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << std::endl;

        constexpr int B = 8;  // 一个最小任务块
        std::vector<zombie::WalkState<Real, DIM>> hostStates(B);

        for (int i = 0; i < B; ++i) {
            hostStates[i].currentPt[0] = float(i);
            hostStates[i].currentPt[1] = float(i + 100);
            hostStates[i].currentNormal[0] = 0.0f;
            hostStates[i].currentNormal[1] = 1.0f;
            hostStates[i].prevDirection[0] = 1.0f;
            hostStates[i].prevDirection[1] = 0.0f;
            hostStates[i].prevDistance = 0.5f;
            hostStates[i].throughput = 1.0f;
            hostStates[i].walkLength = 0;
            hostStates[i].onReflectingBoundary = false;
        }

        std::vector<WalkStateLite> task(B);
        for (int i = 0; i < B; ++i) {
            task[i] = packState(hostStates[i]);
        }

        {
            sycl::buffer<WalkStateLite, 1> buf(task.data(), sycl::range<1>(B));
            q.submit([&](sycl::handler& h) {
                auto acc = buf.get_access<sycl::access::mode::read_write>(h);
                h.parallel_for(sycl::range<1>(B), [=](sycl::id<1> idx) {
                    int i = idx[0];

                    // 模拟一次最小 walk-task 执行
                    acc[i].currentPt[0] += 0.25f;
                    acc[i].currentPt[1] += 0.75f;
                    acc[i].throughput *= 0.95f;
                    acc[i].walkLength += 1;
                    acc[i].totalSourceContribution += 0.1f;
                });
            });
            q.wait();
        }

        for (int i = 0; i < B; ++i) {
            unpackState(task[i], hostStates[i]);
        }

        for (int i = 0; i < B; ++i) {
            std::cout << i
                      << ": pt=(" << hostStates[i].currentPt[0]
                      << ", " << hostStates[i].currentPt[1] << ")"
                      << ", throughput=" << hostStates[i].throughput
                      << ", walkLength=" << hostStates[i].walkLength
                      << ", source=" << hostStates[i].totalSourceContribution
                      << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "SYCL error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}