#' Nystrom Approximation for Kernel Matrices
#'
#' This function computes the Nystrom approximation for large kernel matrices,
#' which provides an efficient low-rank approximation using a subset of landmark points.
#' The approximation is given by K ≈ K_nm * K_mm^(-1) * K_nm^T where K_nm is the
#' kernel matrix between data points and landmarks, and K_mm is the kernel matrix
#' between landmarks.
#'
#' @param X Data matrix where each row is a data point
#' @param kernel Kernel specification created using `gaussian_kernel()`, `sinc_kernel()`,
#'        or combinations of kernels
#' @param num_landmarks Number of landmark points to use (default: min(300, nrow(X)/3))
#' @param regularization Regularization parameter for numerical stability (default: 1e-6)
#' @param batch_size Size of mini-batches for k-means clustering when selecting landmarks (default: 100)
#' @param max_iterations Maximum number of iterations for k-means (default: 100)
#' @param seed Random seed for reproducibility (default: 42)
#' @return An object of class "nystrom_approx" with components:
#'   \item{landmarks}{Matrix of landmark points}
#'   \item{K_nm}{Kernel matrix between data points and landmarks}
#'   \item{K_mm_inv}{Inverse of kernel matrix between landmarks}
#'   \item{kernel}{The kernel specification used}
#'   \item{X}{The original data matrix}
#'
#' @details
#' The Nystrom approximation is particularly useful for large datasets where
#' computing and storing the full kernel matrix is prohibitive. It selects a subset
#' of representative points (landmarks) from the data and constructs a low-rank
#' approximation of the full kernel matrix.
#'
#' Landmark points are selected using mini-batch k-means clustering to ensure
#' good coverage of the data space.
#'
#' @examples
#' # Create a dataset
#' X <- matrix(rnorm(1000 * 10), nrow = 1000)
#'
#' # Create a Gaussian kernel
#' kernel <- gaussian_kernel(1.0)
#'
#' # Compute Nystrom approximation
#' nystrom <- nystrom_approximation(X, kernel, num_landmarks = 50)
#'
#' # Predict for new data
#' X_new <- matrix(rnorm(100 * 10), nrow = 100)
#' pred <- predict(nystrom, X_new)
#'
#' @export
nystrom_approximation <- function(X, kernel, num_landmarks = min(300, nrow(X)/3),
                                 regularization = 1e-6, batch_size = 100,
                                 max_iterations = 100, seed = 42) {
  # Input validation
  if (!is.matrix(X)) {
    X <- as.matrix(X)
  }
  
  if (nrow(X) == 0 || ncol(X) == 0) {
    stop("Input data matrix X cannot be empty")
  }
  
  if (!inherits(kernel, "kernel_spec")) {
    stop("kernel must be a valid kernel specification")
  }
  
  if (num_landmarks <= 0) {
    stop("Number of landmarks must be positive")
  }
  
  if (num_landmarks > nrow(X)) {
    warning("Number of landmarks exceeds number of data points, using all data points as landmarks")
    num_landmarks <- nrow(X)
  }
  
  if (regularization < 0) {
    stop("Regularization parameter must be non-negative")
  }
  
  if (batch_size <= 0) {
    stop("Batch size must be positive")
  }
  
  if (batch_size > nrow(X)) {
    batch_size <- nrow(X)
  }
  
  if (max_iterations <= 0) {
    stop("Maximum iterations must be positive")
  }
  
  # Set seed for reproducibility
  old_seed <- NULL
  if (!is.null(seed)) {
    old_seed <- .Random.seed
    set.seed(seed)
  }
  
  # Compute Nystrom approximation using C++ function
  result <- computeNystromApproximation_cpp(
    X,
    as.integer(num_landmarks),
    kernel,
    regularization,
    as.integer(batch_size),
    as.integer(max_iterations),
    as.integer(seed)
  )
  
  # Restore old seed if needed
  if (!is.null(old_seed)) {
    .Random.seed <- old_seed
  }
  
  # Create return object with class for method dispatch
  nystrom <- list(
    landmarks = result$landmarks,
    K_nm = result$K_nm,
    K_mm_inv = result$K_mm_inv,
    kernel = kernel,
    X = X
  )
  
  class(nystrom) <- "nystrom_approx"
  return(nystrom)
}

#' Print method for Nystrom approximation objects
#'
#' @param x Nystrom approximation object
#' @param ... Additional arguments (not used)
#' @return Invisibly returns the object
#' @export
print.nystrom_approx <- function(x, ...) {
  cat("Nystrom Approximation:\n")
  cat(" - Data dimensions:", dim(x$X)[1], "x", dim(x$X)[2], "\n")
  cat(" - Number of landmarks:", nrow(x$landmarks), "\n")
  cat(" - Kernel: ")
  print_kernel(x$kernel)
  invisible(x)
}

#' Apply Nystrom approximation to new data points
#'
#' This function applies the Nystrom approximation to new data points,
#' computing an approximate kernel matrix between the new points and the
#' original training data.
#'
#' @param object Nystrom approximation object from `nystrom_approximation()`
#' @param newdata New data matrix (each row is a data point)
#' @param ... Additional arguments (not used)
#' @return Approximate kernel matrix between new data and original data
#'
#' @details
#' The approximation is computed as K_val_train ≈ K_val_m * K_mm^(-1) * K_nm^T,
#' where K_val_m is the kernel matrix between validation points and landmarks.
#'
#' @export
predict.nystrom_approx <- function(object, newdata, ...) {
  if (!is.matrix(newdata)) {
    newdata <- as.matrix(newdata)
  }
  
  if (ncol(newdata) != ncol(object$X)) {
    stop("New data must have the same number of features as the original data")
  }
  
  # Compute kernel matrix between new data and landmarks
  K_val_m <- kernel_matrix(newdata, object$landmarks, object$kernel)
  
  # Approximate K_val_train ≈ K_val_m * K_mm^(-1) * K_nm^T
  K_val_train <- K_val_m %*% object$K_mm_inv %*% t(object$K_nm)
  
  return(K_val_train)
}

#' Multiply by the approximated kernel matrix without forming it explicitly
#'
#' This function efficiently computes K * v, where K is the approximated kernel matrix.
#'
#' @param nystrom Nystrom approximation object from `nystrom_approximation()`
#' @param v Vector to multiply with (must have length equal to number of training points)
#' @return Result of K * v
#'
#' @details
#' The multiplication is performed as K * v ≈ K_nm * K_mm^(-1) * K_nm^T * v
#' without explicitly forming the full kernel matrix.
#'
#' @export
nystrom_multiply <- function(nystrom, v) {
  if (!inherits(nystrom, "nystrom_approx")) {
    stop("First argument must be a Nystrom approximation object")
  }
  
  if (!is.numeric(v)) {
    stop("Second argument must be a numeric vector")
  }
  
  if (length(v) != nrow(nystrom$X)) {
    stop("Vector length must match the number of training points")
  }
  
  # Compute K * v ≈ K_nm * K_mm^(-1) * K_nm^T * v
  result <- nystrom$K_nm %*% (nystrom$K_mm_inv %*% (t(nystrom$K_nm) %*% v))
  
  return(result)
}

#' Multiply with projection for the Nystrom approximation
#'
#' This function computes P * K * v, where P is a projection matrix
#' P = I - W(W^T W)^(-1)W^T and K is the approximated kernel matrix.
#'
#' @param nystrom Nystrom approximation object from `nystrom_approximation()`
#' @param v Vector to multiply with
#' @param W Matrix of linear features for projection
#' @return Result of P * K * v
#'
#' @details
#' This is useful for orthogonal projection when you want to compute a kernel
#' matrix that is orthogonal to a specific set of features.
#'
#' @export
nystrom_multiply_with_projection <- function(nystrom, v, W) {
  if (!inherits(nystrom, "nystrom_approx")) {
    stop("First argument must be a Nystrom approximation object")
  }
  
  if (!is.numeric(v)) {
    stop("Second argument must be a numeric vector")
  }
  
  if (length(v) != nrow(nystrom$X)) {
    stop("Vector length must match the number of training points")
  }
  
  if (!is.matrix(W)) {
    W <- as.matrix(W)
  }
  
  if (nrow(W) != nrow(nystrom$X)) {
    stop("W must have the same number of rows as the training data")
  }
  
  # Calculate W(W^T W)^(-1)W^T * v
  WtW <- t(W) %*% W
  Wv <- t(W) %*% v
  WtW_inv_Wv <- solve(WtW, Wv)
  W_WtW_inv_Wv <- W %*% WtW_inv_Wv
  
  # Return (I - W(W^T W)^(-1)W^T) * K * v
  result <- nystrom_multiply(nystrom, v - W_WtW_inv_Wv)
  
  return(result)
}

#' Compute an approximation of the full kernel matrix using Nystrom method
#'
#' This function computes an approximation of the full kernel matrix.
#' Use with caution for large datasets as it still creates a full n×n matrix.
#'
#' @param nystrom Nystrom approximation object
#' @return Approximated full kernel matrix
#'
#' @details
#' The approximation is computed as K ≈ K_nm * K_mm^(-1) * K_nm^T
#' 
#' @export
nystrom_full_matrix <- function(nystrom) {
  if (!inherits(nystrom, "nystrom_approx")) {
    stop("Argument must be a Nystrom approximation object")
  }
  
  # Calculate K ≈ K_nm * K_mm^(-1) * K_nm^T
  K_approx <- nystrom$K_nm %*% nystrom$K_mm_inv %*% t(nystrom$K_nm)
  
  return(K_approx)
}
