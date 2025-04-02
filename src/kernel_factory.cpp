#include "kernel_factory.h"
#include "composite_kernels.h"
#include "kernels.h"
#include <stdexcept>

namespace flexKernel
{

std::unique_ptr<KernelBase> createKernel(const Rcpp::List &kernel_spec)
{
    // Check if the list has a "type" element
    if (!kernel_spec.containsElementNamed("type"))
    {
        throw std::invalid_argument("Kernel specification must contain a 'type' element");
    }

    std::string type = Rcpp::as<std::string>(kernel_spec["type"]);

    if (type == "gaussian")
    {
        // Check required parameters
        if (!kernel_spec.containsElementNamed("bandwidth"))
        {
            throw std::invalid_argument("Gaussian kernel specification must contain a 'bandwidth' parameter");
        }

        double bandwidth = Rcpp::as<double>(kernel_spec["bandwidth"]);
        if (bandwidth <= 0.0)
        {
            throw std::invalid_argument("Bandwidth must be positive");
        }

        return std::make_unique<GaussianKernel>(bandwidth);
    }
    else if (type == "sinc")
    {
        // Check required parameters
        if (!kernel_spec.containsElementNamed("bandwidth"))
        {
            throw std::invalid_argument("Sinc kernel specification must contain a 'bandwidth' parameter");
        }

        double bandwidth = Rcpp::as<double>(kernel_spec["bandwidth"]);
        if (bandwidth <= 0.0)
        {
            throw std::invalid_argument("Bandwidth must be positive");
        }

        return std::make_unique<SincKernel>(bandwidth);
    }
    else if (type == "sum")
    {
        // Check required parameters
        if (!kernel_spec.containsElementNamed("kernels"))
        {
            throw std::invalid_argument("Sum kernel specification must contain a 'kernels' list");
        }

        Rcpp::List kernel_list = Rcpp::as<Rcpp::List>(kernel_spec["kernels"]);
        if (kernel_list.size() == 0)
        {
            throw std::invalid_argument("Sum kernel must contain at least one sub-kernel");
        }

        std::vector<std::unique_ptr<KernelBase>> kernels;
        kernels.reserve(kernel_list.size());

        for (R_xlen_t i = 0; i < kernel_list.size(); ++i)
        {
            Rcpp::List sub_kernel_spec = Rcpp::as<Rcpp::List>(kernel_list[i]);
            kernels.push_back(createKernel(sub_kernel_spec));
        }

        return std::make_unique<SumKernel>(std::move(kernels));
    }
    else if (type == "product")
    {
        // Check required parameters
        if (!kernel_spec.containsElementNamed("kernels"))
        {
            throw std::invalid_argument("Product kernel specification must contain a 'kernels' list");
        }

        Rcpp::List kernel_list = Rcpp::as<Rcpp::List>(kernel_spec["kernels"]);
        if (kernel_list.size() == 0)
        {
            throw std::invalid_argument("Product kernel must contain at least one sub-kernel");
        }

        std::vector<std::unique_ptr<KernelBase>> kernels;
        kernels.reserve(kernel_list.size());

        for (R_xlen_t i = 0; i < kernel_list.size(); ++i)
        {
            Rcpp::List sub_kernel_spec = Rcpp::as<Rcpp::List>(kernel_list[i]);
            kernels.push_back(createKernel(sub_kernel_spec));
        }

        return std::make_unique<ProductKernel>(std::move(kernels));
    }
    else if (type == "scaled")
    {
        // Check required parameters
        if (!kernel_spec.containsElementNamed("scale"))
        {
            throw std::invalid_argument("Scaled kernel specification must contain a 'scale' parameter");
        }
        if (!kernel_spec.containsElementNamed("kernel"))
        {
            throw std::invalid_argument("Scaled kernel specification must contain a 'kernel' specification");
        }

        double scale = Rcpp::as<double>(kernel_spec["scale"]);
        if (scale <= 0.0)
        {
            throw std::invalid_argument("Scale must be positive");
        }

        Rcpp::List base_kernel_spec = Rcpp::as<Rcpp::List>(kernel_spec["kernel"]);
        std::unique_ptr<KernelBase> base_kernel = createKernel(base_kernel_spec);

        return std::make_unique<ScaledKernel>(scale, std::move(base_kernel));
    }
    else
    {
        throw std::invalid_argument("Unknown kernel type: " + type);
    }
}

} // namespace flexKernel
