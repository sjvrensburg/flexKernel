#ifndef MINI_BATCH_KMEANS_H_
#define MINI_BATCH_KMEANS_H_

#include <RcppEigen.h>
#include <omp.h>
#include <algorithm> // For std::min
#include <vector>    // For std::vector
#include <numeric>   // For std::iota
#include <limits>    // For std::numeric_limits

/**
 * mini_batch_kmeans_eigen
 *
 * @param means         (k x n_dims) matrix of cluster centers, updated in-place.
 * @param X             (n_rows x n_dims) data matrix, where each row is an observation.
 * @param k             number of clusters.
 * @param batch_size    mini-batch size.
 * @param max_epochs    number of full passes (epochs) over the dataset.
 * @param num_threads   number of OpenMP threads to use.
 * @param cluster       (n_rows) vector of cluster assignments (0-based indices), updated in-place.
 *
 * @return              true on success; false if data is invalid (e.g., too few points).
 *
 * Reproducibility Note:
 * All random draws (both initialization and mini-batch sampling) come from R's RNG,
 * ensuring that calls to set.seed() in R will fully determine the result.
 */
bool mini_batch_kmeans_eigen(
    Eigen::MatrixXd& means,
    const Eigen::MatrixXd& X,
    const Eigen::Index k,
    const Eigen::Index batch_size,
    const Eigen::Index max_epochs,
    const int num_threads,
    Eigen::VectorXi& cluster
);

//' @title Mini-Batch K-Means Clustering
//'
//' @description
//' Performs mini-batch k-means clustering on a given dataset using the k-means++ initialization method.
//' This function is a wrapper around a C++ implementation using RcppEigen for efficiency.
//'
//' @param XR The data matrix with observations in rows (standard R format).
//' @param k The number of clusters.
//' @param batch_size The size of each mini-batch.
//' @param max_epochs The number of full passes over the dataset.
//' @param num_threads The number of OpenMP threads to use for parallel processing.
//'
//' @return A list with the matrix of cluster centers, a vector of cluster assignments (1-based indices), and a status flag.
//'
//' @details
//' This function implements the mini-batch k-means algorithm, which is an approximation of the standard k-means
//' algorithm designed for large datasets. It processes the data in small random subsets (mini-batches) to update
//' the cluster centers incrementally, reducing computational cost and memory usage.
//'
//' The cluster centers are initialized using the k-means++ method, which selects initial centers that are spread out
//' across the dataset to improve convergence.
//'
//' The function uses OpenMP for parallel processing within each mini-batch, leveraging multiple CPU cores to speed up
//' computations.
//'
//' Random selections, including initialization and mini-batch sampling, are performed using R's random number generator
//' to ensure reproducibility with \code{set.seed()}.
//'
//' If any cluster does not receive points during an epoch, it is reinitialized to a random data point to prevent stagnation.
//'
//' @note The input matrix \code{XR} should have observations in rows and features in columns (standard R format).
//' The returned cluster centers have k rows (one per cluster) and the same number of columns as the input data.
//'
//' @examples
//' \dontrun{
//' set.seed(123)
//' X <- matrix(rnorm(1000 * 10), nrow = 1000)  # 1000 observations, 10 features
//' result <- mini_batch_kmeans_rcpp(X, k = 3, batch_size = 100, max_epochs = 10, num_threads = 2)
//' if (result$success) {
//' print(result$means)  # 3 rows (clusters) × 10 columns (features)
//' print(table(result$cluster))  # Distribution of observations across clusters
//' } else {
//' warning("Mini-batch k-means failed.")
//' }
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List mini_batch_kmeans_rcpp(
    const Rcpp::NumericMatrix& XR,
    const int k,
    const int batch_size,
    const int max_epochs,
    const int num_threads = 1
);

#endif // MINI_BATCH_KMEANS_H_
