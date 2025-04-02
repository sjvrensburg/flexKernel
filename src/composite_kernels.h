#ifndef COMPOSITE_KERNELS_H
#define COMPOSITE_KERNELS_H

#include "kernel_base.h"
#include <memory>
#include <string>
#include <vector>

namespace flexKernel
{

/**
 * Sum of kernels: k(x, y) = k_1(x, y) + k_2(x, y) + ... + k_n(x, y)
 *
 * This class combines multiple kernels through addition, which is useful for
 * modeling functions with multiple characteristic length scales or behaviors.
 */
class SumKernel : public KernelBase
{
  private:
    std::vector<std::unique_ptr<KernelBase>> kernels;

  public:
    /**
     * Create an empty sum kernel.
     * Kernels can be added later using addKernel().
     */
    SumKernel() = default;

    /**
     * Create a sum kernel from a vector of kernels.
     *
     * @param kernels Vector of kernels to be summed
     */
    explicit SumKernel(std::vector<std::unique_ptr<KernelBase>> kernels);

    /**
     * Add a kernel to the sum.
     *
     * @param kernel Kernel to add
     */
    void addKernel(std::unique_ptr<KernelBase> kernel);

    void evaluateSubmatrix(const Eigen::MatrixXd &X1, const Eigen::MatrixXd &X2, Eigen::MatrixXd &K) const override;

    void multiplyInPlace(const Eigen::MatrixXd &X, const Eigen::VectorXd &v, Eigen::VectorXd &result) const override;

    double evaluateScalar(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) const override;

    std::unique_ptr<KernelBase> clone() const override;

    std::string toString() const override;

    // Get number of kernels
    size_t size() const
    {
        return kernels.size();
    }

    // Access individual kernels (for testing/debugging)
    const KernelBase &getKernel(size_t idx) const
    {
        if (idx >= kernels.size())
        {
            throw std::out_of_range("Kernel index out of bounds");
        }
        return *kernels[idx];
    }
};

/**
 * Product of kernels: k(x, y) = k_1(x, y) * k_2(x, y) * ... * k_n(x, y)
 *
 * This class combines multiple kernels through multiplication, which can be used to
 * model functions that have multiple independent constraints (e.g., smoothness in
 * some dimensions but not others).
 */
class ProductKernel : public KernelBase
{
  private:
    std::vector<std::unique_ptr<KernelBase>> kernels;

  public:
    /**
     * Create an empty product kernel.
     * Kernels can be added later using addKernel().
     */
    ProductKernel() = default;

    /**
     * Create a product kernel from a vector of kernels.
     *
     * @param kernels Vector of kernels to be multiplied
     */
    explicit ProductKernel(std::vector<std::unique_ptr<KernelBase>> kernels);

    /**
     * Add a kernel to the product.
     *
     * @param kernel Kernel to add
     */
    void addKernel(std::unique_ptr<KernelBase> kernel);

    void evaluateSubmatrix(const Eigen::MatrixXd &X1, const Eigen::MatrixXd &X2, Eigen::MatrixXd &K) const override;

    void multiplyInPlace(const Eigen::MatrixXd &X, const Eigen::VectorXd &v, Eigen::VectorXd &result) const override;

    double evaluateScalar(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) const override;

    std::unique_ptr<KernelBase> clone() const override;

    std::string toString() const override;

    // Get number of kernels
    size_t size() const
    {
        return kernels.size();
    }

    // Access individual kernels (for testing/debugging)
    const KernelBase &getKernel(size_t idx) const
    {
        if (idx >= kernels.size())
        {
            throw std::out_of_range("Kernel index out of bounds");
        }
        return *kernels[idx];
    }
};

/**
 * Scaled kernel: k(x, y) = scale * k_base(x, y)
 *
 * This class applies a constant scaling factor to another kernel, which can be
 * used to adjust the overall magnitude of the kernel's influence.
 */
class ScaledKernel : public KernelBase
{
  private:
    double scale;
    std::unique_ptr<KernelBase> base_kernel;

  public:
    /**
     * Create a scaled kernel.
     *
     * @param scale Scaling factor
     * @param base_kernel Base kernel to be scaled
     * @throws std::invalid_argument if scale is not positive
     */
    ScaledKernel(double scale, std::unique_ptr<KernelBase> base_kernel);

    void evaluateSubmatrix(const Eigen::MatrixXd &X1, const Eigen::MatrixXd &X2, Eigen::MatrixXd &K) const override;

    void multiplyInPlace(const Eigen::MatrixXd &X, const Eigen::VectorXd &v, Eigen::VectorXd &result) const override;

    double evaluateScalar(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) const override;

    std::unique_ptr<KernelBase> clone() const override;

    std::string toString() const override;

    // Getters
    double getScale() const
    {
        return scale;
    }
    const KernelBase &getBaseKernel() const
    {
        return *base_kernel;
    }
};

} // namespace flexKernel

#endif // COMPOSITE_KERNELS_H
