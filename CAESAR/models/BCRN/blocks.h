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



// note KernlSize is assumed to be an int64_t not a kernel
torch::nn::Conv2d convLayer(int64_t inChannels, int64_t outChannels,
                            int64_t kernelSize, int64_t stride = 1,
                            int64_t dilation = 1, int64_t groups = 1);

BSConvU bluePrintConvLayer(int64_t inChannels, int64_t outChannels,
        int64_t kernelSize, int64_t stride = 1,
        int64_t dilation = 1);

torch::nn::AnyModule norm(const std::string& normType, int64_t nc);

torch::nn::AnyModule  pad(const std::string& padType, int64_t padding);

int64_t getValidPadding(int64_t kernelSize, int64_t dilation);

torch::nn::AnyModule activation(const std::string& actType, bool inplace = true,
        double negSlope = 0.05,
        int64_t nPrelu = 1);



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


// defeined wrong???????????????????-
struct ESAImpl : torch::nn::Module {

BSConvU conv1{nullptr};
BSConvU conv_f{nullptr};
BSConvU conv_max{nullptr};
BSConvU conv2{nullptr};
BSConvU conv3{nullptr};
BSConvU conv3_{nullptr};
BSConvU conv4{nullptr};
   torch::nn::Sigmoid sigmoid{nullptr};
    torch::nn::ReLU relu{nullptr};
 
    ESAImpl(int64_t n_feats);
  
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(ESA);



struct PixelShuffleBlockImpl : torch::nn::Module {
    BSConvU conv{nullptr};                   
    torch::nn::PixelShuffle pixel_shuffle{nullptr};


    PixelShuffleBlockImpl(int64_t in_channels, int64_t out_channels,
                          int64_t upscale_factor = 2,
                          int64_t kernel_size = 3, int64_t stride = 1);

        torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(PixelShuffleBlock);


inline torch::nn::Conv2d conv_layer(int64_t inChannels, int64_t outChannels, int64_t kernelSize,
                                    int64_t stride = 1, int64_t padding = 0, int64_t groups = 1) {
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, outChannels, kernelSize)
                                 .stride(stride)
                                 .padding(padding)
                                 .groups(groups)
                                 .bias(true));
}

struct BlocksImpl : torch::nn::Module {
    torch::nn::Conv2d c1_d_3{nullptr}, c1_r_1{nullptr}, c1_r_2{nullptr};
    torch::nn::AnyModule act; 

    BlocksImpl(int64_t dim, int64_t kernelSize = 3) {

        int64_t padding = (kernelSize - 1) / 2; 
        c1_d_3 = conv_layer(dim, dim, kernelSize, 1, padding, dim); 
        c1_r_1 = conv_layer(dim, dim * 4, 1);
        c1_r_2 = conv_layer(dim * 4, dim, 1);
        
        std::string actName = "gelu";
        act = activation(actName); 

        register_module("c1_d_3", c1_d_3);
        register_module("c1_r_1", c1_r_1);
        register_module("c1_r_2", c1_r_2);
    }

    torch::Tensor forward(torch::Tensor x) {
        
        auto shortcut = x.clone();
        x = c1_d_3->forward(x);
        x = c1_r_1->forward(x);
        x = act.forward(x);        
        x = c1_r_2->forward(x);
        x = x + shortcut;
        return x;
    }
};

TORCH_MODULE(Blocks);

struct saLayerImpl : torch::nn::Module {
    int64_t groups;
    torch::nn::AdaptiveAvgPool2d avgPool{nullptr};
    torch::Tensor cweight, cbias, sweight, sbias;
    torch::nn::Sigmoid sigmoid{nullptr};
    
    torch::nn::GroupNorm gn{nullptr};

    saLayerImpl(int64_t numFeats, int64_t groups = 6);

    torch::Tensor forward(torch::Tensor x);
    static torch::Tensor channelShuffle(torch::Tensor x, int64_t groups);
};

TORCH_MODULE(saLayer);


struct CCALayerImpl : torch::nn::Module{

    torch::nn::AdaptiveAvgPool2d avgPool{nullptr};
    torch::nn::Sequential convDu;
    int64_t channel;
    int64_t reduction;

    CCALayerImpl(int64_t channel_, int64_t reduction_ = 16) :
        channel(channel_), reduction(reduction_) {

        avgPool = torch::nn::AdaptiveAvgPool2d(
            torch::nn::AdaptiveAvgPool2dOptions({1, 1}));

        convDu = torch::nn::Sequential( 
// review logic
            BSConvU(channel, channel / reduction, 1, 1, 0, 1, true),  
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            BSConvU(channel / reduction, channel, 1, 1, 0, 1, true), 
            torch::nn::Sigmoid()
        );

        register_module("avgPool", avgPool);
        register_module("convDu", convDu);
     }

    torch::Tensor forward(torch::Tensor x){
        auto y = avgPool->forward(x);
        y = convDu->forward(y);
        return x*y;
    }
};

TORCH_MODULE(CCALayer);
