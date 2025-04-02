/**
 * @file nystrom.h
 * @brief Nystrom approximation for large-scale kernel matrices
 *
 * This file contains structures and functions for implementing the Nystrom
 * approximation, which is used to efficiently approximate large kernel matrices
 * The approximation is based on selecting a subset of landmark points and
 * using them to construct a low-rank approximation of the full kernel matrix.
 */

#ifndef NYSTROM_H
#define NYSTROM_H

#include "kernel_base.h"
#include "mini_batch_kmeans.h"
#include <RcppEigen.h>

namespace flexKernel
{

/**
 * Compute landmark points for Nystrom approximation using mini-batch k-means.
 *
 * This function selects representative points from the dataset to use as
 * landmarks for the Nystrom approximation. It uses a mini-batch variant of
 * k-means clustering for efficiency on large datasets.
 *
 * @param X Data matrix (each row is a data point)
 * @param num_landmarks Number of landmark points to select
 * @param batch_size Size of mini-batches for k-means (default: 100)
 * @param num_iterations Number of k-means iterations (default: 100)
 * @param seed Random seed for reproducibility (default: 42)
 * @return Matrix of landmark points (each row is a landmark)
 * @throws std::invalid_argument if inputs are invalid
 */
Eigen::MatrixXd computeNystromLandmarks(const Eigen::MatrixXd &X, size_t num_landmarks, size_t batch_size = 100,
                                  size_t num_iterations = 100, unsigned int seed = 42);

/**
 * Structure containing the components needed for Nystrom approximation.
 *
 * The Nystrom approximation of a kernel matrix K is given by:
 * K ≈ K_nm * K_mm^(-1) * K_nm^T
 * where K_nm is the kernel matrix between data points and landmarks,
 * and K_mm is the kernel matrix between landmarks.
 */
struct NystromApproximation
{
    Eigen::MatrixXd landmarks; ///< Landmark points (each row is a landmark)
    Eigen::MatrixXd K_nm;      ///< Kernel matrix between data points and landmarks
    Eigen::MatrixXd K_mm_inv;  ///< Inverse of kernel matrix between landmarks

    /**
     * Multiply by the approximated kernel matrix without forming it explicitly.
     *
     * This efficiently computes K * v, where K is the approximated kernel matrix.
     * K * v ≈ K_nm * K_mm^(-1) * K_nm^T * v
     *
     * @param v Vector to multiply with
     * @return Result of K * v
     */
    Eigen::VectorXd multiply(const Eigen::VectorXd &v) const
    {
        return K_nm * (K_mm_inv * (K_nm.transpose() * v));
    }

    /**
     * Multiply by the approximated kernel matrix with projection.
     *
     * This efficiently computes P * K * v, where P is the projection matrix
     * P = I - W(W^T W)^(-1)W^T and K is the approximated kernel matrix.
     *
     * @param v Vector to multiply with
     * @param W Matrix of linear features for projection
     * @return Result of P * K * v
     */
    Eigen::VectorXd multiplyWithProjection(const Eigen::VectorXd &v, const Eigen::MatrixXd &W) const
    {
        // Calculate W(W^T W)^(-1)W^T * v
        Eigen::MatrixXd WtW = W.transpose() * W;
        Eigen::VectorXd Wv = W.transpose() * v;
        Eigen::VectorXd WtW_inv_Wv = WtW.fullPivLu().solve(Wv);
        Eigen::VectorXd W_WtW_inv_Wv = W * WtW_inv_Wv;

        // Return (I - W(W^T W)^(-1)W^T) * K * v
        return multiply(v - W_WtW_inv_Wv);
    }

    /**
     * Multiply the validation-to-training kernel matrix by a vector using Nyström approximation.
     *
     * This efficiently computes K_val_train * v, where K_val_train is the kernel matrix
     * between validation points and training points, approximated as:
     * K_val_train ≈ K_val_m * K_mm^(-1) * K_nm^T
     *
     * @param X_val Validation data matrix (each row is a validation point)
     * @param v Vector to multiply with (typically alpha coefficients)
     * @param kernel Kernel object to compute K_val_m
     * @return Result of K_val_train * v
     */
    Eigen::VectorXd multiplyValidation(const Eigen::MatrixXd &X_val, const Eigen::VectorXd &v, const KernelBase &kernel) const
    {
        // Validate input dimensions
        if (X_val.cols() != landmarks.cols())
        {
            throw std::invalid_argument("Validation data must have the same number of features as landmarks");
        }
        if (v.size() != K_nm.rows())
        {
            throw std::invalid_argument("Vector length must match the number of training points");
        }

        // Compute K_val_m: kernel matrix between validation points and landmarks
        Eigen::MatrixXd K_val_m;
        kernel.evaluateSubmatrix(X_val, landmarks, K_val_m);

        // Approximate K_val_train * v = K_val_m * K_mm^(-1) * K_nm^T * v
        return K_val_m * (K_mm_inv * (K_nm.transpose() * v));
    }
};

/**
 * Compute the Nystrom approximation for a kernel matrix.
 *
 * This function computes the components needed for the Nystrom approximation
 * of a kernel matrix using the specified landmark points and kernel function.
 *
 * @param X Data matrix (each row is a data point)
 * @param landmarks Landmark points (each row is a landmark)
 * @param kernel Kernel object to use
 * @param regularization Regularization parameter for numerical stability (default: 1e-6)
 * @return NystromApproximation structure containing the approximation components
 * @throws std::invalid_argument if inputs are invalid
 */
NystromApproximation computeNystromApproximation(const Eigen::MatrixXd &X, const Eigen::MatrixXd &landmarks,
                                                 const KernelBase &kernel, double regularization = 1e-6);

} // namespace flexKernel

#endif // NYSTROM_H
