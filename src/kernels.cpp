#include "kernels.h"
#include <sstream>
#include <stdexcept>

namespace flexKernel
{

// GaussianKernel implementation
GaussianKernel::GaussianKernel(double bandwidth) : bandwidth(bandwidth)
{
    if (bandwidth <= 0.0)
    {
        throw std::invalid_argument("Bandwidth must be positive");
    }
}

void GaussianKernel::evaluateSubmatrix(const Eigen::MatrixXd &X1, const Eigen::MatrixXd &X2, Eigen::MatrixXd &K) const
{

    if (X1.cols() != X2.cols())
    {
        throw std::invalid_argument("X1 and X2 must have the same number of columns");
    }

    auto const n1 = X1.rows();
    auto const n2 = X2.rows();

    // Resize output matrix if needed
    if (K.rows() != n1 || K.cols() != n2)
    {
        K.resize(n1, n2);
    }

    // Option 1: Direct computation for small matrices
    // Precompute 1/(2*bandwidth^2) for efficiency
    const double scale = 1.0 / (2.0 * bandwidth * bandwidth);

    if (n1 < 100 || n2 < 100)
    {
        // Direct computation for small matrices
        for (int i = 0; i < n1; ++i)
        {
            const Eigen::VectorXd &x1i = X1.row(i);

            for (int j = 0; j < n2; ++j)
            {
                const Eigen::VectorXd &x2j = X2.row(j);

                // Compute squared Euclidean distance
                double dist_sq = (x1i - x2j).squaredNorm();

                // Apply Gaussian kernel
                K(i, j) = std::exp(-dist_sq * scale);
            }
        }
    }
    else
    {
        // Option 2: Vectorized computation for large matrices
        // Uses identity ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
        Eigen::VectorXd sqnorm1 = X1.rowwise().squaredNorm();
        Eigen::VectorXd sqnorm2 = X2.rowwise().squaredNorm();

        // Compute distance matrix: sqnorm1_i + sqnorm2_j - 2*X1_i*X2_j^T
        K = -2 * X1 * X2.transpose();
        K.rowwise() += sqnorm1.transpose();  // Add row vector to each row
        K.colwise() += sqnorm2;              // Add column vector to each column

        // Apply exponential function
        K = (-K * scale).array().exp();
    }
}

void GaussianKernel::multiplyInPlace(const Eigen::MatrixXd &X, const Eigen::VectorXd &v, Eigen::VectorXd &result) const
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
    const double scale = 1.0 / (2.0 * bandwidth * bandwidth);

    // Perform matrix-vector multiplication without explicitly forming K
    for (int i = 0; i < n; ++i)
    {
        const Eigen::VectorXd &xi = X.row(i);

        for (int j = 0; j < n; ++j)
        {
            const Eigen::VectorXd &xj = X.row(j);

            // Compute squared Euclidean distance
            double dist_sq = (xi - xj).squaredNorm();

            // Apply Gaussian kernel and accumulate
            result(i) += v(j) * std::exp(-dist_sq * scale);
        }
    }
}

double GaussianKernel::evaluateScalar(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) const
{

    if (x1.size() != x2.size())
    {
        throw std::invalid_argument("Vectors must have the same dimension");
    }

    // Compute squared Euclidean distance
    double dist_sq = (x1 - x2).squaredNorm();

    // Apply Gaussian kernel function: exp(-||x-y||^2 / (2*sigma^2))
    return std::exp(-dist_sq / (2.0 * bandwidth * bandwidth));
}

std::unique_ptr<KernelBase> GaussianKernel::clone() const
{
    return std::make_unique<GaussianKernel>(bandwidth);
}

std::string GaussianKernel::toString() const
{
    std::ostringstream oss;
    oss << "GaussianKernel(bandwidth=" << bandwidth << ")";
    return oss.str();
}

// SincKernel implementation
SincKernel::SincKernel(double bandwidth) : bandwidth(bandwidth)
{
    if (bandwidth <= 0.0)
    {
        throw std::invalid_argument("Bandwidth must be positive");
    }
}

void SincKernel::evaluateSubmatrix(const Eigen::MatrixXd &X1, const Eigen::MatrixXd &X2, Eigen::MatrixXd &K) const
{

    if (X1.cols() != X2.cols())
    {
        throw std::invalid_argument("X1 and X2 must have the same number of columns");
    }

    const int n1 = X1.rows();
    const int n2 = X2.rows();
    const int d = X1.cols();

    // Resize output matrix if needed
    if (K.rows() != n1 || K.cols() != n2)
    {
        K.resize(n1, n2);
    }

    // Initialize K to ones (for product)
    K.setOnes();

    // For each dimension, compute sinc values and update K
    for (int k = 0; k < d; ++k)
    {
        // Create matrices of differences
        Eigen::MatrixXd diff_d = X1.col(k).replicate(1, n2) - X2.col(k).transpose().replicate(n1, 1);

        // Scale by bandwidth
        diff_d /= bandwidth;

        // Apply sinc function and multiply into K
        for (int i = 0; i < n1; ++i)
        {
            for (int j = 0; j < n2; ++j)
            {
                K(i, j) *= sinc(diff_d(i, j));
            }
        }
    }
}

void SincKernel::multiplyInPlace(const Eigen::MatrixXd &X, const Eigen::VectorXd &v, Eigen::VectorXd &result) const
{

    if (X.rows() != v.size())
    {
        throw std::invalid_argument("X.rows() must match v.size()");
    }

    const int n = X.rows();
    const int d = X.cols();

    // Resize output vector if needed
    if (result.size() != n)
    {
        result.resize(n);
    }

    result.setZero(); // Initialize result vector to zeros

    // Perform matrix-vector multiplication without explicitly forming K
    for (int i = 0; i < n; ++i)
    {
        const Eigen::VectorXd &xi = X.row(i);

        for (int j = 0; j < n; ++j)
        {
            const Eigen::VectorXd &xj = X.row(j);

            // Compute product of sinc values for each dimension
            double kernel_val = 1.0;
            for (int k = 0; k < d; ++k)
            {
                kernel_val *= sinc((xi(k) - xj(k)) / bandwidth);
            }

            // Accumulate result
            result(i) += v(j) * kernel_val;
        }
    }
}

double SincKernel::evaluateScalar(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) const
{

    if (x1.size() != x2.size())
    {
        throw std::invalid_argument("Vectors must have the same dimension");
    }

    // Compute product of sinc values for each dimension
    double kernel_val = 1.0;
    for (size_t d = 0; d < x1.size(); ++d)
    {
        double diff = x1(d) - x2(d);
        kernel_val *= sinc(diff / bandwidth);
    }

    return kernel_val;
}

std::unique_ptr<KernelBase> SincKernel::clone() const
{
    return std::make_unique<SincKernel>(bandwidth);
}

std::string SincKernel::toString() const
{
    std::ostringstream oss;
    oss << "SincKernel(bandwidth=" << bandwidth << ")";
    return oss.str();
}

} // namespace flexKernel
