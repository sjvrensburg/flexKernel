#include "mini_batch_kmeans.h"
#include <limits>    // For std::numeric_limits

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

bool mini_batch_kmeans_eigen(
    Eigen::MatrixXd& means,
    const Eigen::MatrixXd& X,
    const Eigen::Index k,
    const Eigen::Index batch_size,
    const Eigen::Index max_epochs,
    const int num_threads,
    Eigen::VectorXi& cluster
) {
    const Eigen::Index n_rows = X.rows();  // Number of observations
    const Eigen::Index n_dims = X.cols();  // Number of features

    // --- Safety check
    if (n_rows < k) {
        Rcpp::Rcerr << "mini_batch_kmeans: number of points less than number of clusters" << std::endl;
        return false;
    }

    // --------------------------------------------------------------------------
    // 1. Initialize cluster centers using k-means++
    //    Selects initial centers to be well-separated using R's RNG for reproducibility.
    // --------------------------------------------------------------------------
    means.resize(k, n_dims);  // k rows (clusters) × n_dims columns (features)

    // Select the first center randomly
    Rcpp::IntegerVector first_idx_R = Rcpp::sample(n_rows, 1, false) - 1;
    Eigen::Index first_idx = static_cast<Eigen::Index>(first_idx_R[0]);
    means.row(0) = X.row(first_idx);

    // Initialize min_dists with distances to the first center
    Eigen::VectorXd min_dists(n_rows);
    for (Eigen::Index i = 0; i < n_rows; ++i) {
        min_dists[i] = (X.row(i) - means.row(0)).squaredNorm();
    }

    // Select remaining centers
    for (Eigen::Index c = 1; c < k; ++c) {
        // Compute total sum of min_dists
        double total_sum = min_dists.sum();
        // Generate a random value between 0 and total_sum using R's RNG
        double rand_val = R::runif(0, 1) * total_sum;
        // Find the smallest index where the cumulative sum >= rand_val
        double cumsum = 0;
        Eigen::Index selected_idx = 0;
        for (Eigen::Index i = 0; i < n_rows; ++i) {
            cumsum += min_dists[i];
            if (cumsum >= rand_val) {
                selected_idx = i;
                break;
            }
        }
        // Add the selected point as the next center
        means.row(c) = X.row(selected_idx);
        // Update min_dists
        for (Eigen::Index i = 0; i < n_rows; ++i) {
            double dist = (X.row(i) - means.row(c)).squaredNorm();
            if (dist < min_dists[i]) {
                min_dists[i] = dist;
            }
        }
    }

    // --------------------------------------------------------------------------
    // 2. Initialize cluster counts (for the online update)
    // --------------------------------------------------------------------------
    Eigen::VectorXi counts = Eigen::VectorXi::Zero(k);

    // --------------------------------------------------------------------------
    // 3. Main loop over epochs
    //    Each epoch processes the entire dataset in mini-batches.
    // --------------------------------------------------------------------------
    for (Eigen::Index epoch = 0; epoch < max_epochs; ++epoch) {
        // (a) Shuffle all indices via R's RNG
        Rcpp::IntegerVector shuffled_R = Rcpp::sample(n_rows, n_rows, false) - 1;
        std::vector<Eigen::Index> shuffled_indices(n_rows);
        for (Eigen::Index i = 0; i < n_rows; ++i) {
            shuffled_indices[i] = static_cast<Eigen::Index>(shuffled_R[i]);
        }

        // (b) Loop over mini-batches
        for (Eigen::Index start_idx = 0; start_idx < n_rows; start_idx += batch_size) {
            Eigen::Index end_idx = std::min<Eigen::Index>(start_idx + batch_size, n_rows);
            Eigen::Index current_batch_size = end_idx - start_idx;

            // Extract the mini-batch rows
            Eigen::MatrixXd M(current_batch_size, n_dims);
            for (Eigen::Index i = 0; i < current_batch_size; ++i) {
                M.row(i) = X.row(shuffled_indices[start_idx + i]);
            }

            // (c) Assign points to clusters and accumulate sums in parallel
            //     Each thread processes a subset of the mini-batch points.
            //     For each point, find the nearest cluster center and add the point
            //     to the local sum and count for that cluster.
            //     After processing, merge the local sums and counts into global accumulators.
            Eigen::MatrixXd acc_sums = Eigen::MatrixXd::Zero(k, n_dims);
            Eigen::VectorXi acc_counts = Eigen::VectorXi::Zero(k);

            omp_set_num_threads(num_threads);

            #pragma omp parallel
            {
                // Initialize local accumulators for this thread
                Eigen::MatrixXd local_sums = Eigen::MatrixXd::Zero(k, n_dims);
                Eigen::VectorXi local_counts = Eigen::VectorXi::Zero(k);

                // Parallel loop over the mini-batch points
                #pragma omp for
                for (Eigen::Index i = 0; i < current_batch_size; ++i) {
                    // Initialize minimum distance and best cluster
                    double min_dist = std::numeric_limits<double>::infinity();
                    Eigen::Index best_g = 0;
                    // Find the nearest cluster center
                    for (Eigen::Index g = 0; g < k; ++g) {
                        // Compute squared Euclidean distance between observation i and cluster center g
                        double dist = (M.row(i) - means.row(g)).squaredNorm();
                        if (dist < min_dist) {
                            min_dist = dist;
                            best_g = g;
                        }
                    }
                    // Add the point to the local sum and count for the best cluster
                    local_sums.row(best_g) += M.row(i);
                    local_counts[best_g]++;
                }

                // Merge local accumulators into global accumulators
                #pragma omp critical
                {
                    acc_sums += local_sums;
                    for (Eigen::Index g = 0; g < k; ++g) {
                        acc_counts[g] += local_counts[g];
                    }
                }
            } // end parallel region

            // (d) Update cluster centers using the online update formula
            //     For each cluster, if it received points in this mini-batch (k_c > 0),
            //     update the center as a weighted average of the previous center and the mini-batch mean.
            //     Specifically, new_center = (n_c * old_center + sum_c) / (n_c + k_c)
            //     where n_c is the cumulative count so far, and k_c is the count in this mini-batch.
            //     Then, update the cumulative count: n_c += k_c
            for (Eigen::Index g = 0; g < k; ++g) {
                int k_c = acc_counts[g]; // number of points assigned to cluster g in this mini-batch
                if (k_c > 0) {
                    int n_c = counts[g]; // cumulative count of points assigned to cluster g so far
                    // Update the center
                    means.row(g) = (static_cast<double>(n_c) * means.row(g) + acc_sums.row(g))
                                  / static_cast<double>(n_c + k_c);
                    // Update the cumulative count
                    counts[g] = n_c + k_c;
                }
            }
        } // end mini-batch loop

        // (e) Reinitialize empty clusters
        //     If any cluster has not been assigned any points so far (counts[g] == 0),
        //     reinitialize it to a random data point.
        for (Eigen::Index g = 0; g < k; ++g) {
            if (counts[g] == 0) {
                // Select a random point using R's RNG
                Rcpp::IntegerVector rand_idx_R = Rcpp::sample(n_rows, 1, false) - 1;
                Eigen::Index rand_idx = static_cast<Eigen::Index>(rand_idx_R[0]);
                means.row(g) = X.row(rand_idx);
            }
        }
    } // end epoch loop

    // --------------------------------------------------------------------------
    // 4. Assign each observation to its nearest cluster center
    // --------------------------------------------------------------------------

    // Initialize the cluster assignment vector
    cluster.resize(n_rows);

    // Set up OpenMP parallelization
    omp_set_num_threads(num_threads);

    // Assign each observation to the nearest cluster center
    #pragma omp parallel for
    for (Eigen::Index i = 0; i < n_rows; ++i) {
        double min_dist = std::numeric_limits<double>::infinity();
        Eigen::Index best_cluster = 0;

        // Find the nearest cluster center
        for (Eigen::Index g = 0; g < k; ++g) {
            double dist = (X.row(i) - means.row(g)).squaredNorm();
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = g;
            }
        }

        // Assign the observation to the best cluster (0-based index)
        cluster[i] = static_cast<int>(best_cluster);
    }

    // --------------------------------------------------------------------------
    // 5. Final check
    // --------------------------------------------------------------------------
    // Check if any of the means contains NaN values
    for (Eigen::Index i = 0; i < means.rows(); ++i) {
        for (Eigen::Index j = 0; j < means.cols(); ++j) {
            if (std::isnan(means(i, j))) {
                return false;
            }
        }
    }

    return true;
}

Rcpp::List mini_batch_kmeans_rcpp(
    const Rcpp::NumericMatrix& XR,
    const int k,
    const int batch_size,
    const int max_epochs,
    const int num_threads
) {
    // 1. Convert R matrix to Eigen
    const Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd> >(XR));

    // 2. Initialize means matrix with correct dimensions (k rows × n_dims columns)
    Eigen::MatrixXd means(k, X.cols());

    // 3. Initialize cluster assignment vector
    Eigen::VectorXi cluster;

    // 4. Run mini-batch k-means
    bool success = mini_batch_kmeans_eigen(
        means,
        X,
        static_cast<Eigen::Index>(k),
        static_cast<Eigen::Index>(batch_size),
        static_cast<Eigen::Index>(max_epochs),
        num_threads,
        cluster
    );

    // 5. Convert 0-based cluster indices to 1-based for R
    Eigen::VectorXi r_cluster = cluster.array() + 1;

    return Rcpp::List::create(
      Rcpp::Named("means") = means,
      Rcpp::Named("cluster") = r_cluster,
      Rcpp::Named("success") = success
    );
}
