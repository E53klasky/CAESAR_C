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

torch::nn::AnyModule  padding(const std::string& padType, int64_t padding);

int64_t getValidPadding(int64_t kernelSize, int64_t dilation);


torch::nn::AnyModule activation(std::string& actType, bool inplace = true,
        double negSlope = 0.05,
        int64_t nPrelu =1);



// for conv Block need this a class with the ---  sequential ---
/* def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)

    */
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


// convert this method then go to conv_blocks  in this file:
// CAESAR/CAESAR/models/BCRN/block.py
// def sequential(*args):
