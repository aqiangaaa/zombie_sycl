#include <sycl/sycl.hpp>
#include <array>
#include <iostream>

struct WalkStateLite {
    std::array<float, 2> currentPt;
    std::array<float, 2> currentNormal;
    std::array<float, 2> prevDirection;
    float prevDistance;
    float throughput;
    int walkLength;
    int onReflectingBoundary;
};

int main() {
    try {
        sycl::queue q;
        std::cout << "Device: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << std::endl;

        constexpr int N = 8;
        WalkStateLite states[N];

        for (int i = 0; i < N; ++i) {
            states[i].currentPt = {float(i), float(i + 1)};
            states[i].currentNormal = {0.0f, 1.0f};
            states[i].prevDirection = {1.0f, 0.0f};
            states[i].prevDistance = 0.5f;
            states[i].throughput = 1.0f;
            states[i].walkLength = 0;
            states[i].onReflectingBoundary = 0;
        }

        {
            sycl::buffer<WalkStateLite, 1> buf(states, sycl::range<1>(N));
            q.submit([&](sycl::handler& h) {
                auto acc = buf.get_access<sycl::access::mode::read_write>(h);
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                    int i = idx[0];
                    acc[i].currentPt[0] += 0.25f;
                    acc[i].currentPt[1] += 0.50f;
                    acc[i].throughput *= 0.9f;
                    acc[i].walkLength += 1;
                });
            });
            q.wait();
        }

        for (int i = 0; i < N; ++i) {
            std::cout << i
                      << ": pt=(" << states[i].currentPt[0] << ", " << states[i].currentPt[1] << ")"
                      << ", throughput=" << states[i].throughput
                      << ", walkLength=" << states[i].walkLength
                      << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "SYCL error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}