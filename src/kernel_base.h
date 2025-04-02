#ifndef KERNEL_BASE_H
#define KERNEL_BASE_H

#include <RcppEigen.h>
#include <memory>
#include <string>

namespace flexKernel {

/**
 * Abstract base class for kernels.
 *
 * This class defines the interface for all kernel implementations.
 * Concrete kernel classes must implement the evaluateSubmatrix,
 * multiplyInPlace, evaluateScalar, clone, and toString methods.
 */
class KernelBase {
public:
  virtual ~KernelBase() = default;

  /**
   * Compute the kernel matrix for two sets of data points.
   *
   * @param X1 First set of data points (each row is a data point)
   * @param X2 Second set of data points (each row is a data point)
   * @param K Output kernel matrix (will be overwritten)
   * @throws std::invalid_argument if dimensions are incompatible
   */
  virtual void evaluateSubmatrix(
      const Eigen::MatrixXd& X1,
      const Eigen::MatrixXd& X2,
      Eigen::MatrixXd& K) const = 0;

  /**
   * Perform matrix-vector multiplication K*v where K is the kernel matrix.
   * This avoids explicitly forming the full kernel matrix.
   *
   * @param X Data matrix (each row is a data point)
   * @param v Vector to multiply with
   * @param result Output vector (will be overwritten)
   * @throws std::invalid_argument if dimensions are incompatible
   */
  virtual void multiplyInPlace(
      const Eigen::MatrixXd& X,
      const Eigen::VectorXd& v,
      Eigen::VectorXd& result) const = 0;

  /**
   * Evaluate the kernel function for a single pair of data points.
   *
   * @param x1 First data point (row vector)
   * @param x2 Second data point (row vector)
   * @return Kernel evaluation k(x1, x2)
   * @throws std::invalid_argument if vectors have different dimensions
   */
  virtual double evaluateScalar(
      const Eigen::VectorXd& x1,
      const Eigen::VectorXd& x2) const = 0;

  /**
   * Create a clone of this kernel object.
   *
   * @return A unique_ptr to a new instance of the same kernel with
   *         identical parameters
   */
  virtual std::unique_ptr<KernelBase> clone() const = 0;

  /**
   * Get a string representation of the kernel.
   *
   * @return A human-readable string describing the kernel and its parameters
   */
  virtual std::string toString() const = 0;
};

} // namespace flexKernel

#endif // KERNEL_BASE_H
