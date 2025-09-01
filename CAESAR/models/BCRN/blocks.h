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
