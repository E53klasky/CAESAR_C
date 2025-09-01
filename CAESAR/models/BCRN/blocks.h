#pragma once
#include <torch/torch.h>
#include <tuple>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include "BSConvU.h"


torch::nn::Conv2d convLayer(int64_t inChannels, int64_t outChannels,
        int64_t kernelSize, int64_t stride = 1,
        int64_t dilation = 1, int64_t groups = 1);

BSConvU bluePrintConvLayer(int64_t inChannels, int64_t outChannels,
        int64_t kernelSize, int64_t stride = 1,
        int64_t dilation = 1);

torch::nn::AnyModule norm(const std::string& normType, int64_t nc);

torch::nn::AnyModule  pad(const std::string& padType, int64_t padding);

int64_t getValidPadding(int64_t kernelSize, int64_t dilation);


torch::nn::AnyModule activation(std::string& actType, bool inplace = true,
        double negSlope = 0.05,
        int64_t nPrelu =1);



torch::nn::Sequential convBlock(int64_t inChannel, int64_t outChannel, int64_t kernelSize,
        int64_t stride = 1, int64_t dilation=1, int64_t groups = 1, bool bias=true,
        const std::string& paddingType = "zero", const std::string& normType = "", const
        std::string& actType = "relu");


class ShortcutBlockImpl : public torch::nn::Module {
public:
    ShortcutBlockImpl(torch::nn::AnyModule submodule);
    torch::Tensor forward(torch::Tensor x);
private:
    torch::nn::AnyModule sub;
};
TORCH_MODULE(ShortcutBlock);

torch::Tensor meanChannels(const torch::Tensor& F);
torch::Tensor stdvChannels(const torch::Tensor& F);

torch::nn::Sequential sequential(std::vector<torch::nn::AnyModule> args);



struct ESAImpl : torch::nn::Module {

    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv_f{nullptr};
    torch::nn::Conv2d conv_max{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Conv2d conv3{nullptr};
    torch::nn::Conv2d conv3_{nullptr};
    torch::nn::Conv2d conv4{nullptr};
    torch::nn::Sigmoid sigmoid{nullptr};
    torch::nn::ReLU relu{nullptr};

 
    ESAImpl(int64_t n_feats);
  
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(ESA);

