#include "composite_kernels.h"
#include <sstream>
#include <stdexcept>

namespace flexKernel
{

// SumKernel implementation
SumKernel::SumKernel(std::vector<std::unique_ptr<KernelBase>> kernels)
{
    if (kernels.empty())
    {
        throw std::invalid_argument("Cannot create a sum kernel with empty kernels vector");
    }

    for (auto &kernel : kernels)
    {
        if (!kernel)
        {
            throw std::invalid_argument("Null kernel pointer in kernels vector");
        }
        this->kernels.push_back(kernel->clone());
    }
}

void SumKernel::addKernel(std::unique_ptr<KernelBase> kernel)
{
    if (!kernel)
    {
        throw std::invalid_argument("Cannot add null kernel pointer");
    }
    kernels.push_back(std::move(kernel));
}

void SumKernel::evaluateSubmatrix(const Eigen::MatrixXd &X1, const Eigen::MatrixXd &X2, Eigen::MatrixXd &K) const
{

    if (X1.cols() != X2.cols())
    {
        throw std::invalid_argument("X1 and X2 must have the same number of columns");
    }

    if (kernels.empty())
    {
        // Empty sum kernel returns zeros
        K.setZero(X1.rows(), X2.rows());
        return;
    }

    const int n1 = X1.rows();
    const int n2 = X2.rows();

    // Resize output matrix if needed
    if (K.rows() != n1 || K.cols() != n2)
    {
        K.resize(n1, n2);
    }

    K.setZero(); // Initialize output matrix to zeros

    // Temporary matrix for individual kernel evaluations
    Eigen::MatrixXd K_temp(n1, n2);

    // Sum up all kernel evaluations
    for (const auto &kernel : kernels)
    {
        kernel->evaluateSubmatrix(X1, X2, K_temp);
        K += K_temp;
    }
}

void SumKernel::multiplyInPlace(const Eigen::MatrixXd &X, const Eigen::VectorXd &v, Eigen::VectorXd &result) const
{

    if (X.rows() != v.size())
    {
        throw std::invalid_argument("X.rows() must match v.size()");
    }

    const int n = X.rows();

    // Resize output vector if needed
    if (result.size() != n)
    {
        result.resize(n);
    }

    result.setZero(); // Initialize result vector to zeros

    if (kernels.empty())
    {
        return; // Empty sum kernel returns zeros
    }

    // Temporary vector for individual kernel multiplications
    Eigen::VectorXd temp_result(n);

    // Sum up all kernel-vector multiplications
    for (const auto &kernel : kernels)
    {
        kernel->multiplyInPlace(X, v, temp_result);
        result += temp_result;
    }
}

double SumKernel::evaluateScalar(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) const
{

    if (x1.size() != x2.size())
    {
        throw std::invalid_argument("Vectors must have the same dimension");
    }

    if (kernels.empty())
    {
        return 0.0; // Empty sum is zero
    }

    double sum = 0.0;
    for (const auto &kernel : kernels)
    {
        sum += kernel->evaluateScalar(x1, x2);
    }
    return sum;
}

std::unique_ptr<KernelBase> SumKernel::clone() const
{
    std::vector<std::unique_ptr<KernelBase>> cloned_kernels;
    cloned_kernels.reserve(kernels.size());

    for (const auto &kernel : kernels)
    {
        cloned_kernels.push_back(kernel->clone());
    }

    return std::make_unique<SumKernel>(std::move(cloned_kernels));
}

std::string SumKernel::toString() const
{
    std::ostringstream oss;
    oss << "SumKernel(";

    if (kernels.empty())
    {
        oss << "empty";
    }
    else
    {
        for (size_t i = 0; i < kernels.size(); ++i)
        {
            if (i > 0)
            {
                oss << " + ";
            }
            oss << kernels[i]->toString();
        }
    }

    oss << ")";
    return oss.str();
}

// ProductKernel implementation
ProductKernel::ProductKernel(std::vector<std::unique_ptr<KernelBase>> kernels)
{
    if (kernels.empty())
    {
        throw std::invalid_argument("Cannot create a product kernel with empty kernels vector");
    }

    for (auto &kernel : kernels)
    {
        if (!kernel)
        {
            throw std::invalid_argument("Null kernel pointer in kernels vector");
        }
        this->kernels.push_back(kernel->clone());
    }
}

void ProductKernel::addKernel(std::unique_ptr<KernelBase> kernel)
{
    if (!kernel)
    {
        throw std::invalid_argument("Cannot add null kernel pointer");
    }
    kernels.push_back(std::move(kernel));
}

void ProductKernel::evaluateSubmatrix(const Eigen::MatrixXd &X1, const Eigen::MatrixXd &X2, Eigen::MatrixXd &K) const
{

    if (X1.cols() != X2.cols())
    {
        throw std::invalid_argument("X1 and X2 must have the same number of columns");
    }

    const int n1 = X1.rows();
    const int n2 = X2.rows();

    // Resize output matrix if needed
    if (K.rows() != n1 || K.cols() != n2)
    {
        K.resize(n1, n2);
    }

    if (kernels.empty())
    {
        K.setOnes(); // Empty product is one
        return;
    }

    // Initialize with the first kernel
    kernels[0]->evaluateSubmatrix(X1, X2, K);

    // Temporary matrix for subsequent kernel evaluations
    Eigen::MatrixXd K_temp(n1, n2);

    // Multiply by all remaining kernels
    for (size_t i = 1; i < kernels.size(); ++i)
    {
        kernels[i]->evaluateSubmatrix(X1, X2, K_temp);
        K.array() *= K_temp.array(); // Element-wise multiplication
    }
}

void ProductKernel::multiplyInPlace(const Eigen::MatrixXd &X, const Eigen::VectorXd &v, Eigen::VectorXd &result) const
{

    if (X.rows() != v.size())
    {
        throw std::invalid_argument("X.rows() must match v.size()");
    }

    const int n = X.rows();

    // For product kernels, we can't directly use the matrix-vector multiplication
    // trick without forming the explicit kernel matrix, so we form the matrix.
    // This is less efficient but maintains correctness.

    // Compute the full kernel matrix
    Eigen::MatrixXd K(n, n);
    evaluateSubmatrix(X, X, K);

    // Resize output vector if needed
    if (result.size() != n)
    {
        result.resize(n);
    }

    // Perform matrix-vector multiplication
    result = K * v;
}

double ProductKernel::evaluateScalar(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) const
{

    if (x1.size() != x2.size())
    {
        throw std::invalid_argument("Vectors must have the same dimension");
    }

    if (kernels.empty())
    {
        return 1.0; // Empty product is one
    }

    double product = 1.0;
    for (const auto &kernel : kernels)
    {
        product *= kernel->evaluateScalar(x1, x2);
    }
    return product;
}

std::unique_ptr<KernelBase> ProductKernel::clone() const
{
    std::vector<std::unique_ptr<KernelBase>> cloned_kernels;
    cloned_kernels.reserve(kernels.size());

    for (const auto &kernel : kernels)
    {
        cloned_kernels.push_back(kernel->clone());
    }

    return std::make_unique<ProductKernel>(std::move(cloned_kernels));
}

std::string ProductKernel::toString() const
{
    std::ostringstream oss;
    oss << "ProductKernel(";

    if (kernels.empty())
    {
        oss << "empty";
    }
    else
    {
        for (size_t i = 0; i < kernels.size(); ++i)
        {
            if (i > 0)
            {
                oss << " * ";
            }
            oss << kernels[i]->toString();
        }
    }

    oss << ")";
    return oss.str();
}

// ScaledKernel implementation
ScaledKernel::ScaledKernel(double scale, std::unique_ptr<KernelBase> base_kernel)
    : scale(scale), base_kernel(std::move(base_kernel))
{

    if (scale <= 0.0)
    {
        throw std::invalid_argument("Scale must be positive");
    }

    if (!this->base_kernel)
    {
        throw std::invalid_argument("Base kernel cannot be null");
    }
}

void ScaledKernel::evaluateSubmatrix(const Eigen::MatrixXd &X1, const Eigen::MatrixXd &X2, Eigen::MatrixXd &K) const
{

    // Compute the base kernel matrix
    base_kernel->evaluateSubmatrix(X1, X2, K);

    // Scale the kernel matrix
    K *= scale;
}

void ScaledKernel::multiplyInPlace(const Eigen::MatrixXd &X, const Eigen::VectorXd &v, Eigen::VectorXd &result) const
{

    // Apply the base kernel's multiplication
    base_kernel->multiplyInPlace(X, v, result);

    // Scale the result
    result *= scale;
}

double ScaledKernel::evaluateScalar(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) const
{

    return scale * base_kernel->evaluateScalar(x1, x2);
}

std::unique_ptr<KernelBase> ScaledKernel::clone() const
{
    return std::make_unique<ScaledKernel>(scale, base_kernel->clone());
}

std::string ScaledKernel::toString() const
{
    std::ostringstream oss;
    oss << "ScaledKernel(scale=" << scale << ", base_kernel=" << base_kernel->toString() << ")";
    return oss.str();
}

} // namespace flexKernel
