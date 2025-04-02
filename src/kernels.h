#ifndef KERNELS_H
#define KERNELS_H

#include "kernel_base.h"
#include <cmath>
#include <memory>
#include <string>

namespace flexKernel
{

/**
 * Gaussian (RBF) kernel: k(x, y) = exp(-||x - y||^2 / (2 * bandwidth^2))
 *
 * This kernel is infinitely differentiable, making it suitable for modeling
 * smooth functions. The bandwidth parameter controls the width of the kernel,
 * with smaller values leading to more localized effects.
 */
class GaussianKernel : public KernelBase
{
  private:
    double bandwidth; // Kernel bandwidth parameter (sigma)

  public:
    /**
     * Create a Gaussian kernel with the specified bandwidth.
     *
     * @param bandwidth Kernel bandwidth parameter (sigma)
     * @throws std::invalid_argument if bandwidth is not positive
     */
    explicit GaussianKernel(double bandwidth);

    void evaluateSubmatrix(const Eigen::MatrixXd &X1, const Eigen::MatrixXd &X2, Eigen::MatrixXd &K) const override;

    void multiplyInPlace(const Eigen::MatrixXd &X, const Eigen::VectorXd &v, Eigen::VectorXd &result) const override;

    double evaluateScalar(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) const override;

    std::unique_ptr<KernelBase> clone() const override;

    std::string toString() const override;

    // Getter for bandwidth
    double getBandwidth() const
    {
        return bandwidth;
    }
};

/**
 * Sinc kernel: k(x, y) = prod_i sinc((x_i - y_i) / bandwidth)
 * where sinc(z) = sin(pi * z) / (pi * z)
 *
 * This kernel corresponds to a box filter in the frequency domain
 * and is often used in signal processing applications.
 */
class SincKernel : public KernelBase
{
  private:
    double bandwidth; // Kernel bandwidth parameter

  public:
    /**
     * Create a Sinc kernel with the specified bandwidth.
     *
     * @param bandwidth Kernel bandwidth parameter
     * @throws std::invalid_argument if bandwidth is not positive
     */
    explicit SincKernel(double bandwidth);

    void evaluateSubmatrix(const Eigen::MatrixXd &X1, const Eigen::MatrixXd &X2, Eigen::MatrixXd &K) const override;

    void multiplyInPlace(const Eigen::MatrixXd &X, const Eigen::VectorXd &v, Eigen::VectorXd &result) const override;

    double evaluateScalar(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) const override;

    std::unique_ptr<KernelBase> clone() const override;

    std::string toString() const override;

    // Getter for bandwidth
    double getBandwidth() const
    {
        return bandwidth;
    }
};

/**
 * Helper function for sinc calculation
 * Computes sin(pi*x)/(pi*x) with appropriate handling for x=0
 *
 * @param x Input value
 * @return sinc(x)
 */
inline double sinc(double x)
{
    if (std::abs(x) < 1e-10)
    {
        return 1.0; // limit of sin(pi*x)/(pi*x) as x approaches 0
    }
    else
    {
        const double pi_x = M_PI * x;
        return std::sin(pi_x) / pi_x;
    }
}

} // namespace flexKernel

#endif // KERNELS_H
