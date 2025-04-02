#include "kernel_factory.h"
#include "nystrom.h"
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' Compute the kernel matrix between two sets of data points.
//'
//' @param X1 First set of data points (each row is a data point)
//' @param X2 Second set of data points (each row is a data point)
//' @param kernel_spec R list describing the kernel
//' @return Kernel matrix K
// [[Rcpp::export]]
Eigen::MatrixXd kernelMatrix_cpp(const Eigen::MatrixXd &X1, const Eigen::MatrixXd &X2, const Rcpp::List &kernel_spec)
{
    try
    {
        // Create kernel object from specification
        std::unique_ptr<flexKernel::KernelBase> kernel = flexKernel::createKernel(kernel_spec);
        
        // Compute kernel matrix
        Eigen::MatrixXd K(X1.rows(), X2.rows());
        kernel->evaluateSubmatrix(X1, X2, K);
        return K;
    }
    catch (const std::exception &e)
    {
        Rcpp::stop("Error computing kernel matrix: %s", e.what());
    }
}

//' Compute the Nystrom approximation of a kernel matrix.
//'
//' @param X Data matrix (each row is a data point)
//' @param num_landmarks Number of landmark points to use
//' @param kernel_spec R list describing the kernel
//' @param regularization Regularization parameter for inverting K_mm
//' @param batch_size Size of mini-batches for k-means
//' @param max_iterations Maximum number of iterations for k-means
//' @param seed Random seed for initialization
//' @return List containing the Nystrom approximation components
// [[Rcpp::export]]
Rcpp::List computeNystromApproximation_cpp(const Eigen::MatrixXd &X, int num_landmarks, const Rcpp::List &kernel_spec,
                                          double regularization = 1e-6, int batch_size = 100, int max_iterations = 100,
                                          unsigned int seed = 42)
{
    try
    {
        // Validate inputs
        if (num_landmarks <= 0)
        {
            Rcpp::stop("Number of landmarks must be positive");
        }
        if (regularization < 0)
        {
            Rcpp::stop("Regularization parameter must be non-negative");
        }
        if (batch_size <= 0)
        {
            Rcpp::stop("Batch size must be positive");
        }
        if (max_iterations <= 0)
        {
            Rcpp::stop("Maximum iterations must be positive");
        }
        
        // Create kernel object from specification
        std::unique_ptr<flexKernel::KernelBase> kernel = flexKernel::createKernel(kernel_spec);
        
        // Compute landmarks using mini-batch k-means
        Eigen::MatrixXd landmarks = flexKernel::computeNystromLandmarks(X, num_landmarks, batch_size, max_iterations, seed);
        
        // Compute Nystrom approximation
        flexKernel::NystromApproximation approx = flexKernel::computeNystromApproximation(X, landmarks, *kernel, regularization);
        
        // Return results as a list
        return Rcpp::List::create(
            Rcpp::Named("landmarks") = approx.landmarks,
            Rcpp::Named("K_nm") = approx.K_nm,
            Rcpp::Named("K_mm_inv") = approx.K_mm_inv
        );
    }
    catch (const std::exception &e)
    {
        Rcpp::stop("Error computing Nystrom approximation: %s", e.what());
    }
}

//' Evaluate the kernel function for a single pair of data points.
//'
//' @param x1 First data point (vector)
//' @param x2 Second data point (vector)
//' @param kernel_spec R list describing the kernel
//' @return Kernel value k(x1, x2)
// [[Rcpp::export]]
double kernelScalar_cpp(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, const Rcpp::List &kernel_spec)
{
    try
    {
        // Create kernel object from specification
        std::unique_ptr<flexKernel::KernelBase> kernel = flexKernel::createKernel(kernel_spec);
        
        // Compute kernel value
        return kernel->evaluateScalar(x1, x2);
    }
    catch (const std::exception &e)
    {
        Rcpp::stop("Error computing kernel value: %s", e.what());
    }
}
