#' @title Kernel Specification Functions for KGF Regression
#' @description Functions for creating and manipulating kernel specifications.
#' This includes both basic kernel types and functions for combining kernels
#' through addition, multiplication, and scaling.

#' Create a Gaussian kernel specification
#'
#' The Gaussian (RBF) kernel is defined as:
#' \deqn{k(x, y) = \exp(-\|x - y\|^2 / (2 \sigma^2))}
#' where \eqn{\sigma} is the bandwidth parameter.
#'
#' @param bandwidth Kernel bandwidth parameter (sigma)
#' @return A list representing a Gaussian kernel specification
#' @examples
#' # Create a Gaussian kernel with bandwidth 1.0
#' kernel <- gaussian_kernel(1.0)
#' @export
gaussian_kernel <- function(bandwidth) {
  # Validate input
  if (!is.numeric(bandwidth)) {
    stop("Bandwidth must be numeric")
  }
  if (length(bandwidth) != 1) {
    stop("Bandwidth must be a scalar")
  }
  if (bandwidth <= 0) {
    stop("Bandwidth must be positive")
  }

  # Create kernel specification
  kernel <- list(type = "gaussian", bandwidth = bandwidth)
  class(kernel) <- c("kernel_spec", "list")
  return(kernel)
}

#' Create a Sinc kernel specification
#'
#' The Sinc kernel is defined as:
#' \deqn{k(x, y) = \prod_i \text{sinc}((x_i - y_i) / \text{bandwidth})}
#' where \eqn{\text{sinc}(z) = \sin(\pi z) / (\pi z)}.
#'
#' @param bandwidth Kernel bandwidth parameter
#' @return A list representing a Sinc kernel specification
#' @examples
#' # Create a Sinc kernel with bandwidth 1.0
#' kernel <- sinc_kernel(1.0)
#' @export
sinc_kernel <- function(bandwidth) {
  # Validate input
  if (!is.numeric(bandwidth)) {
    stop("Bandwidth must be numeric")
  }
  if (length(bandwidth) != 1) {
    stop("Bandwidth must be a scalar")
  }
  if (bandwidth <= 0) {
    stop("Bandwidth must be positive")
  }

  # Create kernel specification
  kernel <- list(type = "sinc", bandwidth = bandwidth)
  class(kernel) <- c("kernel_spec", "list")
  return(kernel)
}

#' Create a sum of kernels specification
#'
#' The sum kernel combines multiple kernels through addition:
#' \deqn{k(x, y) = k_1(x, y) + k_2(x, y) + \ldots + k_n(x, y)}
#'
#' @param ... Kernel specifications to be combined
#' @return A list representing a sum kernel specification
#' @examples
#' # Create a sum of two Gaussian kernels with different bandwidths
#' kernel <- sum_kernels(
#'   gaussian_kernel(1.0),
#'   gaussian_kernel(0.1)
#' )
#' # You can also use the + operator:
#' # kernel <- gaussian_kernel(1.0) + gaussian_kernel(0.1)
#' @export
sum_kernels <- function(...) {
  # Get kernels from arguments
  kernels <- list(...)

  # Validate inputs
  if (length(kernels) == 0) {
    stop("At least one kernel must be provided")
  }

  # Check that all arguments are valid kernel specifications
  for (i in seq_along(kernels)) {
    if (!inherits(kernels[[i]], "kernel_spec")) {
      stop(paste("Argument", i, "is not a valid kernel specification"))
    }
  }

  # Simplify if there's only one kernel in the list
  if (length(kernels) == 1) {
    return(kernels[[1]])
  }

  # Flatten nested sum kernels
  flat_kernels <- list()
  for (kernel in kernels) {
    if (kernel$type == "sum") {
      # Add each sub-kernel individually
      flat_kernels <- c(flat_kernels, kernel$kernels)
    } else {
      # Add non-sum kernel as is
      flat_kernels <- c(flat_kernels, list(kernel))
    }
  }

  # Create kernel specification
  kernel <- list(type = "sum", kernels = flat_kernels)
  class(kernel) <- c("kernel_spec", "list")
  return(kernel)
}

#' Create a product of kernels specification
#'
#' The product kernel combines multiple kernels through multiplication:
#' \deqn{k(x, y) = k_1(x, y) \times k_2(x, y) \times \ldots \times k_n(x, y)}
#'
#' @param ... Kernel specifications to be multiplied
#' @return A list representing a product kernel specification
#' @examples
#' # Create a product of a Gaussian kernel and a Sinc kernel
#' kernel <- product_kernels(
#'   gaussian_kernel(1.0),
#'   sinc_kernel(0.5)
#' )
#' # You can also use the * operator:
#' # kernel <- gaussian_kernel(1.0) * sinc_kernel(0.5)
#' @export
product_kernels <- function(...) {
  # Get kernels from arguments
  kernels <- list(...)

  # Validate inputs
  if (length(kernels) == 0) {
    stop("At least one kernel must be provided")
  }

  # Check that all arguments are valid kernel specifications
  for (i in seq_along(kernels)) {
    if (!inherits(kernels[[i]], "kernel_spec")) {
      stop(paste("Argument", i, "is not a valid kernel specification"))
    }
  }

  # Simplify if there's only one kernel in the list
  if (length(kernels) == 1) {
    return(kernels[[1]])
  }

  # Flatten nested product kernels
  flat_kernels <- list()
  for (kernel in kernels) {
    if (kernel$type == "product") {
      # Add each sub-kernel individually
      flat_kernels <- c(flat_kernels, kernel$kernels)
    } else {
      # Add non-product kernel as is
      flat_kernels <- c(flat_kernels, list(kernel))
    }
  }

  # Create kernel specification
  kernel <- list(type = "product", kernels = flat_kernels)
  class(kernel) <- c("kernel_spec", "list")
  return(kernel)
}

#' Scale a kernel specification
#'
#' The scaled kernel applies a constant scaling factor to another kernel:
#' \deqn{k(x, y) = \text{scale} \times k_{\text{base}}(x, y)}
#'
#' @param kernel Kernel specification to be scaled
#' @param scale Scaling factor
#' @return A list representing a scaled kernel specification
#' @examples
#' # Create a scaled Gaussian kernel
#' kernel <- scale_kernel(gaussian_kernel(1.0), 2.5)
#' # You can also use the * operator:
#' # kernel <- 2.5 * gaussian_kernel(1.0)
#' @export
scale_kernel <- function(kernel, scale) {
  # Validate inputs
  if (!inherits(kernel, "kernel_spec")) {
    stop("First argument must be a valid kernel specification")
  }

  if (!is.numeric(scale)) {
    stop("Scale must be numeric")
  }

  if (length(scale) != 1) {
    stop("Scale must be a scalar")
  }

  if (scale <= 0) {
    stop("Scale must be positive")
  }

  # Special case: if the kernel is already a scaled kernel,
  # just multiply the scales instead of nesting
  if (kernel$type == "scaled") {
    return(scale_kernel(kernel$kernel, scale * kernel$scale))
  }

  # Create kernel specification
  kernel <- list(type = "scaled", kernel = kernel, scale = scale)
  class(kernel) <- c("kernel_spec", "list")
  return(kernel)
}

#' Print a kernel specification
#'
#' Internal function to recursively print kernel specifications.
#'
#' @param kernel Kernel specification
#' @param indent Indentation level (for nested printing)
#' @return Invisible NULL
#' @keywords internal
print_kernel <- function(kernel, indent = 0) {
  # Create indentation string
  ind <- paste(rep("  ", indent), collapse = "")

  # Print kernel information based on type
  if (kernel$type == "gaussian") {
    cat(ind, "Gaussian kernel (bandwidth = ", kernel$bandwidth, ")\n", sep = "")
  } else if (kernel$type == "sinc") {
    cat(ind, "Sinc kernel (bandwidth = ", kernel$bandwidth, ")\n", sep = "")
  } else if (kernel$type == "scaled") {
    cat(ind, "Scaled kernel (scale = ", kernel$scale, ")\n", sep = "")
    print_kernel(kernel$kernel, indent + 1)
  } else if (kernel$type == "sum") {
    cat(ind, "Sum of kernels:\n", sep = "")
    for (k in kernel$kernels) {
      print_kernel(k, indent + 1)
    }
  } else if (kernel$type == "product") {
    cat(ind, "Product of kernels:\n", sep = "")
    for (k in kernel$kernels) {
      print_kernel(k, indent + 1)
    }
  } else {
    cat(ind, "Unknown kernel type: ", kernel$type, "\n", sep = "")
  }

  invisible(NULL)
}

#' Print method for kernel specifications
#'
#' @param x Kernel specification
#' @param ... Additional arguments (not used)
#' @return Invisible NULL
#' @export
print.kernel_spec <- function(x, ...) {
  cat("Kernel Specification:\n")
  print_kernel(x)
  invisible(x)
}

#' Overloaded operators for kernel specifications
#'
#' This function allows the use of arithmetic operators with kernel specifications,
#' providing an intuitive way to construct complex kernels.
#'
#' Supported operations:
#' - `kernel + kernel`: Creates a sum of kernels
#' - `kernel * kernel`: Creates a product of kernels
#' - `number * kernel` or `kernel * number`: Creates a scaled kernel
#'
#' @param e1 First operand
#' @param e2 Second operand (may be missing for unary operators)
#' @return A new kernel specification
#' @examples
#' # Create some basic kernels
#' g <- gaussian_kernel(1.0)
#' s <- sinc_kernel(0.5)
#'
#' # Create complex kernels using operators
#' sum_kernel <- g + s          # Sum of kernels
#' product_kernel <- g * s      # Product of kernels
#' scaled_kernel <- 2.5 * g     # Scaled kernel
#'
#' # Create a more complex kernel
#' complex_kernel <- 0.5 * g + 2.0 * (s * g)
#' @export
Ops.kernel_spec <- function(e1, e2) {
  op <- .Generic

  # Handle unary operators (not likely needed for kernels)
  if (missing(e2)) {
    switch(op,
           "+" = return(e1),
           "-" = stop("Unary negation not supported for kernels"),
           stop(paste("Unary operator", op, "not supported for kernels"))
    )
  }

  # Handle binary operators
  # Both arguments are kernels
  if (inherits(e1, "kernel_spec") && inherits(e2, "kernel_spec")) {
    switch(op,
           "+" = return(sum_kernels(e1, e2)),
           "*" = return(product_kernels(e1, e2)),
           stop(paste("Binary operator", op, "not supported for kernel-kernel operations"))
    )
  }

  # One argument is a kernel, one is a number
  if (inherits(e1, "kernel_spec") && is.numeric(e2)) {
    switch(op,
           "*" = return(scale_kernel(e1, e2)),
           stop(paste("Binary operator", op, "not supported for kernel-numeric operations"))
    )
  }

  if (is.numeric(e1) && inherits(e2, "kernel_spec")) {
    switch(op,
           "*" = return(scale_kernel(e2, e1)),
           stop(paste("Binary operator", op, "not supported for numeric-kernel operations"))
    )
  }

  # If we get here, the operation is not supported
  stop(paste("Operation", op, "not supported for these argument types"))
}

#' Create a kernel matrix from data points
#'
#' This function computes the kernel matrix for a given set of data points
#' using a specified kernel.
#'
#' @param X1 First set of data points (each row is a data point)
#' @param X2 Second set of data points (optional, defaults to X1)
#' @param kernel Kernel specification
#' @return Kernel matrix K
#' @examples
#' # Generate some data
#' X <- matrix(rnorm(20), 10, 2)
#'
#' # Create a Gaussian kernel
#' kernel <- gaussian_kernel(1.0)
#'
#' # Compute the kernel matrix
#' K <- kernel_matrix(X, kernel = kernel)
#' @export
kernel_matrix <- function(X1, X2 = NULL, kernel) {
  # Validate inputs
  if (!is.matrix(X1)) {
    X1 <- as.matrix(X1)
  }

  if (is.null(X2)) {
    X2 <- X1
  } else if (!is.matrix(X2)) {
    X2 <- as.matrix(X2)
  }

  if (ncol(X1) != ncol(X2)) {
    stop("X1 and X2 must have the same number of columns")
  }

  if (!inherits(kernel, "kernel_spec")) {
    stop("kernel must be a valid kernel specification")
  }

  # Compute kernel matrix using C++ function
  K <- kernelMatrix_cpp(X1, X2, kernel)

  return(K)
}

#' Generate kernel specifications for cross-validation
#'
#' This function creates a grid of kernel specifications for use in
#' cross-validation or grid search. It can generate multiple bandwidth
#' values for Gaussian kernels, or combinations of different kernel types.
#'
#' @param kernel_type Type of kernel to generate: "gaussian", "sinc", or "both"
#' @param bandwidths Vector of bandwidth values to try
#' @param scales Vector of scaling factors to try (optional)
#' @return List of kernel specifications
#' @examples
#' # Generate a grid of Gaussian kernels with different bandwidths
#' kernels <- generate_kernel_grid("gaussian", c(0.1, 1.0, 10.0))
#'
#' # Generate a grid with both Gaussian and Sinc kernels
#' kernels <- generate_kernel_grid("both", c(0.5, 2.0))
#'
#' # Generate a grid with scaled kernels
#' kernels <- generate_kernel_grid("gaussian", c(1.0, 5.0), scales = c(0.5, 2.0))
#' @export
generate_kernel_grid <- function(kernel_type = "gaussian", bandwidths = 10^(-1:1), scales = NULL) {
  # Validate inputs
  if (!is.character(kernel_type) || length(kernel_type) != 1) {
    stop("kernel_type must be a single character string")
  }

  if (!kernel_type %in% c("gaussian", "sinc", "both")) {
    stop("kernel_type must be one of 'gaussian', 'sinc', or 'both'")
  }

  if (!is.numeric(bandwidths) || length(bandwidths) == 0) {
    stop("bandwidths must be a non-empty numeric vector")
  }

  if (!is.null(scales) && (!is.numeric(scales) || length(scales) == 0)) {
    stop("scales must be NULL or a non-empty numeric vector")
  }

  # Generate basic kernels
  kernels <- list()

  if (kernel_type %in% c("gaussian", "both")) {
    # Add Gaussian kernels
    for (bw in bandwidths) {
      kernels <- c(kernels, list(gaussian_kernel(bw)))
    }
  }

  if (kernel_type %in% c("sinc", "both")) {
    # Add Sinc kernels
    for (bw in bandwidths) {
      kernels <- c(kernels, list(sinc_kernel(bw)))
    }
  }

  # Apply scaling if requested
  if (!is.null(scales)) {
    scaled_kernels <- list()
    for (kernel in kernels) {
      for (scale in scales) {
        scaled_kernels <- c(scaled_kernels, list(scale_kernel(kernel, scale)))
      }
    }
    kernels <- scaled_kernels
  }

  return(kernels)
}

#' Kernel distance between two sets of points
#'
#' This function computes the pairwise kernel distances between two sets
#' of points. The kernel distance is defined as:
#' \deqn{d_k(x, y) = \sqrt{k(x, x) + k(y, y) - 2k(x, y)}}
#'
#' @param X1 First set of data points (each row is a data point)
#' @param X2 Second set of data points (optional, defaults to X1)
#' @param kernel Kernel specification
#' @return Matrix of kernel distances
#' @examples
#' # Generate some data
#' X <- matrix(rnorm(20), 10, 2)
#'
#' # Create a Gaussian kernel
#' kernel <- gaussian_kernel(1.0)
#'
#' # Compute kernel distances
#' D <- kernel_distance(X, kernel = kernel)
#' @export
kernel_distance <- function(X1, X2 = NULL, kernel) {
  # Validate inputs
  if (!is.matrix(X1)) {
    X1 <- as.matrix(X1)
  }

  if (is.null(X2)) {
    X2 <- X1
    self_comparison <- TRUE
  } else {
    if (!is.matrix(X2)) {
      X2 <- as.matrix(X2)
    }
    self_comparison <- identical(X1, X2)
  }

  if (ncol(X1) != ncol(X2)) {
    stop("X1 and X2 must have the same number of columns")
  }

  if (!inherits(kernel, "kernel_spec")) {
    stop("kernel must be a valid kernel specification")
  }

  # Compute kernel matrices
  K12 <- kernel_matrix(X1, X2, kernel)

  if (self_comparison) {
    # If X1 == X2, we only need to compute one kernel matrix
    K11 <- diag(K12)
    K22 <- K11
  } else {
    # Compute diagonal elements of kernel matrices
    K11 <- diag(kernel_matrix(X1, X1, kernel))
    K22 <- diag(kernel_matrix(X2, X2, kernel))
  }

  # Compute kernel distances
  D <- matrix(0, nrow = nrow(X1), ncol = nrow(X2))
  for (i in 1:nrow(X1)) {
    for (j in 1:nrow(X2)) {
      D[i, j] <- sqrt(max(0, K11[i] + K22[j] - 2 * K12[i, j]))
    }
  }

  return(D)
}
