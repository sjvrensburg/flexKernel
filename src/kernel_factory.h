#ifndef KERNEL_FACTORY_H
#define KERNEL_FACTORY_H

#include "kernel_base.h"
#include <RcppEigen.h>
#include <memory>

namespace flexKernel
{

/**
 * Create a kernel object from an R list specification.
 *
 * This function takes an R list that describes a kernel configuration and
 * returns a C++ kernel object that implements the specified kernel.
 *
 * The list should have a "type" element that specifies the kernel type,
 * along with type-specific parameters:
 *
 * - Gaussian kernel: list(type = "gaussian", bandwidth = <value>)
 * - Sinc kernel: list(type = "sinc", bandwidth = <value>)
 * - Sum kernel: list(type = "sum", kernels = <list of kernel specs>)
 * - Product kernel: list(type = "product", kernels = <list of kernel specs>)
 * - Scaled kernel: list(type = "scaled", scale = <value>, kernel = <kernel spec>)
 *
 * @param kernel_spec R list describing the kernel configuration
 * @return A unique_ptr to the created kernel object
 * @throws std::invalid_argument if the specification is invalid or incomplete
 */
std::unique_ptr<KernelBase> createKernel(const Rcpp::List &kernel_spec);

} // namespace flexKernel

#endif // KERNEL_FACTORY_H
