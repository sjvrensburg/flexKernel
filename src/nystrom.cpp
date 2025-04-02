/**
 * @file nystrom.cpp
 * @brief Implementation of Nystrom approximation functions
 *
 * This file implements the functions for Nystrom approximation of large
 * kernel matrices, allowing for efficient kernel operations with large datasets.
 */

#include "nystrom.h"
#include <stdexcept>

namespace flexKernel
{

Eigen::MatrixXd computeNystromLandmarks(const Eigen::MatrixXd &X, size_t num_landmarks, size_t batch_size, size_t num_iterations,
                                        unsigned int seed)
{

    // Validate inputs
    if (X.rows() == 0 || X.cols() == 0)
    {
        throw std::invalid_argument("Input data matrix X cannot be empty");
    }

    if (num_landmarks == 0)
    {
        throw std::invalid_argument("Number of landmarks must be positive");
    }

    if (num_landmarks > X.rows())
    {
        Rcpp::warning("Number of landmarks exceeds number of data points, using all data points as landmarks");
        return X;
    }

    if (batch_size == 0)
    {
        throw std::invalid_argument("Batch size must be positive");
    }

    if (batch_size > X.rows())
    {
        batch_size = X.rows();
    }

    // Set random seed
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed = base_env["set.seed"];
    set_seed(seed);

    // Initialize means matrix for cluster centers
    Eigen::MatrixXd means(num_landmarks, X.cols());

    // Initialize cluster assignments
    Eigen::VectorXi cluster;

    // Run mini-batch k-means
    bool success =
      mini_batch_kmeans_eigen(means, X, static_cast<Eigen::Index>(num_landmarks), static_cast<Eigen::Index>(batch_size),
                              static_cast<Eigen::Index>(num_iterations),
                              1, // num_threads
                              cluster);

    if (!success)
    {
        throw std::runtime_error("Mini-batch k-means clustering failed");
    }

    return means;
}

NystromApproximation computeNystromApproximation(const Eigen::MatrixXd &X, const Eigen::MatrixXd &landmarks,
                                                 const KernelBase &kernel, double regularization)
{

    // Validate inputs
    if (X.rows() == 0 || X.cols() == 0)
    {
        throw std::invalid_argument("Input data matrix X cannot be empty");
    }

    if (landmarks.rows() == 0 || landmarks.cols() == 0)
    {
        throw std::invalid_argument("Landmarks matrix cannot be empty");
    }

    if (X.cols() != landmarks.cols())
    {
        throw std::invalid_argument("Data points and landmarks must have the same number of features");
    }

    if (regularization < 0)
    {
        throw std::invalid_argument("Regularization parameter must be non-negative");
    }

    // Create result structure
    NystromApproximation approx;
    approx.landmarks = landmarks;

    // Compute kernel matrix between data and landmarks: K_nm
    kernel.evaluateSubmatrix(X, landmarks, approx.K_nm);

    // Compute kernel matrix between landmarks: K_mm
    Eigen::MatrixXd K_mm;
    kernel.evaluateSubmatrix(landmarks, landmarks, K_mm);

    // Add regularization to diagonal for numerical stability
    for (size_t i = 0; i < K_mm.rows(); ++i)
    {
        K_mm(i, i) += regularization;
    }

    // Compute inverse of K_mm
    Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(K_mm);
    bool success = lu_decomp.isInvertible();

    // If inversion fails, use pseudo-inverse as fallback
    if (!success)
    {
      std::cerr << "Warning: Matrix inversion failed in Nystrom approximation, using pseudo-inverse" << std::endl;

      // Eigen equivalent of arma::pinv(K_mm) using Singular Value Decomposition (SVD)
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(K_mm, Eigen::ComputeThinU | Eigen::ComputeThinV);
      Eigen::VectorXd singularValues = svd.singularValues();
      Eigen::MatrixXd singularValuesInv(singularValues.size(), singularValues.size());
      singularValuesInv.setZero();
      double epsilon = std::numeric_limits<double>::epsilon() * std::max(K_mm.rows(), K_mm.cols()) * singularValues.array().abs().maxCoeff();
      for (int i = 0; i < singularValues.size(); ++i) {
        if (singularValues(i) > epsilon) {
          singularValuesInv(i, i) = 1.0 / singularValues(i);
        }
      }
      approx.K_mm_inv = svd.matrixV() * singularValuesInv * svd.matrixU().transpose();
    } else {
      approx.K_mm_inv = lu_decomp.inverse();
    }

    return approx;
}

} // namespace flexKernel
