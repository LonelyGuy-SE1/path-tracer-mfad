#include <Eigen/Dense>
#include <iostream>
#include <omp.h>

int main() {
    std::cout << "Path Tracer (MFAD) initialized." << std::endl;
    std::cout << "Eigen version: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "."
              << EIGEN_MINOR_VERSION << std::endl;
    std::cout << "OpenMP maximum threads available: " << omp_get_max_threads() << std::endl;

    // Quick sanity check of Eigen vector operations
    Eigen::Vector3d v(1.0, 2.0, 3.0);
    std::cout << "Sanity check vector norm: " << v.norm() << std::endl;

    return 0;
}
