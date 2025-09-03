#pragma once
#include <torch/torch.h>
#include "blocks.h"
#include "BSConvU.h"

struct BluePrintShortcutBlockImpl : torch::nn::Module {
    BSConvU conv{nullptr};
    Blocks convNextBlock{nullptr};
    ESA esa{nullptr};
    CCALayer cca{nullptr};

    BluePrintShortcutBlockImpl(int64_t inChannels, int64_t outChannels, int64_t kernelSize = 3);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(BluePrintShortcutBlock);  

struct BluePrintConvNeXtSRImpl : torch::nn::Module {
    BSConvU conv1{nullptr};
    BluePrintShortcutBlock convNext1{nullptr};
    BluePrintShortcutBlock convNext2{nullptr};
    BluePrintShortcutBlock convNext3{nullptr};
    BluePrintShortcutBlock convNext4{nullptr};
    BluePrintShortcutBlock convNext5{nullptr};
    BluePrintShortcutBlock convNext6{nullptr};

    BSConvU conv2{nullptr};
    PixelShuffleBlock upsampleBlock{nullptr};
    torch::nn::AnyModule act;

    BluePrintConvNeXtSRImpl(int64_t inChannels, int64_t outChannels,
                             int64_t upscaleFactor = 2, int64_t baseChannels = 64);

    torch::Tensor forward(torch::Tensor x);

// this method is seems like it is never used but here for debuging
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
    loadPartModel(const std::string& pretrainPath);
};

TORCH_MODULE(BluePrintConvNeXtSR); 

