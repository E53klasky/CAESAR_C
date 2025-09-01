#include "blocks.h"
#include <algorithm>

// rember to make a file to test this when done
//NOTE:  assuming kernel size is 1d 
torch::nn::Conv2d convLayer(int64_t inChannels, int64_t outChannels,
        int64_t kernelSize, int64_t stride,
        int64_t dilation, int64_t groups) {

    int64_t padding = (kernelSize - 1) / 2 * dilation;
  return torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, outChannels, kernelSize)
    .stride(stride)
    .padding(padding)
    .bias(true)
    .dilation(dilation)
    .groups(groups));  
  
}


BSConvU bluePrintConvLayer(int64_t inChannels, int64_t outChannels,
        int64_t kernelSize, int64_t stride, int64_t dilation) {

    int64_t padding = (kernelSize - 1) / 2 * dilation;
    
    return BSConvU(inChannels, outChannels, kernelSize, stride, padding, 1);
}


torch::nn::AnyModule norm(const std::string& norm_type, int64_t nc) {
    std::string norm_lower = norm_type;
    std::transform(norm_lower.begin(), norm_lower.end(), norm_lower.begin(), ::tolower);
    
    if (norm_lower == "batch") {
        auto layer = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(nc).affine(true));
        return torch::nn::AnyModule(layer);
    } else if (norm_lower == "instance") {
        auto layer = torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(nc).affine(false));
        return torch::nn::AnyModule(layer);
    } else {
        throw std::runtime_error("normalization layer [" + norm_type + "] is not found");
    }
}
