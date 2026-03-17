#include <sycl/sycl.hpp>
#include <zombie/zombie.h>
#include <iostream>

int main() {
    try {
        sycl::queue q;
        std::cout << "Device: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << std::endl;

        zombie::WalkSettings settings(1e-3f, 1e-3f, 16, false);
        settings.russianRouletteThreshold = 0.1f;

        zombie::Vector<2> pt = zombie::Vector<2>::Zero();
        pt << 1.0f, 2.0f;

        zombie::Vector<2> normal = zombie::Vector<2>::Zero();
        normal << 0.0f, 1.0f;

        zombie::SamplePoint<float, 2> p(
            pt,
            normal,
            zombie::SampleType::InDomain,
            zombie::EstimationQuantity::Solution,
            1.0f,   // pdf
            0.5f,   // distToAbsorbingBoundary
            0.5f    // distToReflectingBoundary
        );

        std::cout << "maxWalkLength = " << settings.maxWalkLength << std::endl;
        std::cout << "point = (" << p.pt[0] << ", " << p.pt[1] << ")" << std::endl;

        int data[4] = {1, 2, 3, 4};
        {
            sycl::buffer<int, 1> buf(data, sycl::range<1>(4));
            q.submit([&](sycl::handler& h) {
                auto acc = buf.get_access<sycl::access::mode::read_write>(h);
                h.parallel_for(sycl::range<1>(4), [=](sycl::id<1> i) {
                    acc[i] += 1;
                });
            });
            q.wait();
        }

        std::cout << data[0] << " " << data[1] << " "
                  << data[2] << " " << data[3] << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "SYCL error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}