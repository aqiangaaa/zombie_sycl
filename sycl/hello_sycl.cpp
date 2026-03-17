#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    try {
        sycl::queue q;
        std::cout << "Device: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << std::endl;

        int data[4] = {1, 2, 3, 4};
        {
            sycl::buffer<int, 1> buf(data, sycl::range<1>(4));
            q.submit([&](sycl::handler& h) {
                auto acc = buf.get_access<sycl::access::mode::read_write>(h);
                h.parallel_for(sycl::range<1>(4), [=](sycl::id<1> i) {
                    acc[i] *= 2;
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