#include "blocks.h"
#include <algorithm>
#include <cctype>
#include <stdexcept>

// rember to make a file to TEST this when done
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

torch::nn::AnyModule pad(const std::string& paddingType, int64_t padding){
   if (padding == 0)
       // same as none
       return torch::nn::AnyModule();
   std::string lowerPadType = paddingType;

   std::transform(
    lowerPadType.begin(),
    lowerPadType.end(),
    lowerPadType.begin(),
    [](unsigned char c){ return std::tolower(c); }
);

   if(lowerPadType == "reflect"){
      return torch::nn::AnyModule(torch::nn::ReflectionPad2d(
                  torch::nn::ReflectionPad2dOptions(padding)));
              }

   else if(lowerPadType == "replicate"){
   return torch::nn::AnyModule(torch::nn::ReflectionPad2d(
            torch::nn::ReflectionPad2dOptions(padding)));
           }
   else{
   throw std::runtime_error("padding layer [" + lowerPadType + "] is not implemented");
   }
}

int64_t getValidPadding(int64_t kernelSize, int64_t dilation){
    kernelSize = kernelSize + ( kernelSize -1) * (dilation -1);
    int64_t padding = (kernelSize-1) / 2;
    return padding;
}


torch::nn::AnyModule activation(
    const std::string& act_type,
    bool inplace,
    double neg_slope,
    int64_t n_prelu
) {
    std::string lowerAct = act_type;
    std::transform(
        lowerAct.begin(), lowerAct.end(), lowerAct.begin(),
        [](unsigned char c){ return std::tolower(c); }
    );

    if (lowerAct == "relu") {
        return torch::nn::AnyModule(
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(inplace))
        );
    } 
    else if (lowerAct == "lrelu") {
        return torch::nn::AnyModule(
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions()
                .negative_slope(neg_slope)
                .inplace(inplace))
        );
    } 
    else if (lowerAct == "prelu") {
        return torch::nn::AnyModule(
            torch::nn::PReLU(torch::nn::PReLUOptions()
                .num_parameters(n_prelu)
                .init(neg_slope))
        );
    } 
    else if (lowerAct == "gelu") {
        return torch::nn::AnyModule(torch::nn::GELU());
    } 
    else {
        throw std::runtime_error(
            "activation layer [" + act_type + "] is not implemented"
        );
    }
}

//  on this one -> def conv_block  in CAESAR/CAESAR/models/BCRN/conv_block
//

ShortcutBlockImpl::ShortcutBlockImpl(torch::nn::AnyModule submodule) 
    : sub(submodule) {

    auto module_ptr = sub.ptr();
    if (module_ptr) {
        for (auto& param : module_ptr->named_parameters()) {
            register_parameter(param.key(), param.value());
        }
        for (auto& buffer : module_ptr->named_buffers()) {
            register_buffer(buffer.key(), buffer.value());
        }
    }
}

torch::Tensor ShortcutBlockImpl::forward(torch::Tensor x) {
    return x + sub.forward(x);
}

torch::Tensor meanChannels(const torch::Tensor& F){
    TORCH_CHECK(F.dim() == 4, "Expected 4D tensor (N,C,H,W)");
    auto spatialSum = F.sum(3,true).sum(2,true);
    return spatialSum/(F.size(2) * F.size(3));
}

torch::Tensor stdvChannels(const torch::Tensor& F){
   TORCH_CHECK(F.dim() == 4, "Expected 4D tensor (N,C,H,W)");
   auto FMean = meanChannels(F);
   auto FVariance = (F-FMean).pow(2).sum(3, true).sum(2,true)/ (F.size(2) * F.size(3));
   return FVariance;
}

// this is the clostest 1 to 1 that i can do btw pytorch and libtorch they are just different
// should work for what i need only two cases
 torch::nn::Sequential sequential(std::vector<torch::nn::AnyModule> args) {
    torch::nn::Sequential seq;

    for (auto& module : args) {
        if (module.ptr() != nullptr) {  
            seq->push_back(std::move(module));
        }
    }

    return seq;
}



torch::nn::Sequential convBlock(int64_t inChannel, int64_t outChannel, int64_t kernelSize,
        int64_t stride, int64_t dilation, int64_t groups, bool bias,
        const std::string& paddingType, const std::string& normType, const std::string& actType){
   
  int64_t padding =  getValidPadding(kernelSize, dilation);
   
torch::nn::AnyModule p = torch::nn::AnyModule(); 
if (!paddingType.empty() && paddingType != "zero") {
    p = pad(paddingType, padding);
}
if (paddingType != "zero") {
    padding = 0;
}

    auto conv_options = torch::nn::Conv2dOptions(inChannel, outChannel, kernelSize)
        .stride(stride)
        .padding(padding)
        .dilation(dilation)
        .groups(groups)
        .bias(bias);
   torch::nn::AnyModule c = torch::nn::AnyModule(torch::nn::Conv2d(conv_options));
   std::string myActType = actType;
   torch::nn::AnyModule a;
    if (!actType.empty()) {
        a = activation(myActType);
    }

    torch::nn::AnyModule n;
    if (!normType.empty()) {
        n = norm(normType, outChannel);
    }
    
    return sequential({p, c, n, a});
}




