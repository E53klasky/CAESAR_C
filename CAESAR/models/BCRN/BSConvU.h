#pragma once
#include <torch/torch.h>
#include <tuple>
#include <string>
#include <vector>

struct BSConvUImpl : public torch::nn::Module {
    torch::nn::Conv2d pw{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
    torch::nn::Conv2d dw{nullptr};
    bool has_bn;

   
    BSConvUImpl(
        int64_t in_channels,
        int64_t out_channels,
        int64_t kernel_size,
        int64_t stride = 1,
        int64_t padding = 0,
        int64_t dilation = 1,
        bool bias = true,
        const std::string& padding_mode = "zeros",
        bool with_bn = false
    ) : BSConvUImpl(in_channels, out_channels, std::make_tuple(kernel_size, kernel_size), 
                    stride, padding, dilation, bias, padding_mode, with_bn) {
    }

    
    BSConvUImpl(
        int64_t in_channels,
        int64_t out_channels,
        std::tuple<int64_t, int64_t> kernel_size,
        int64_t stride = 1,
        int64_t padding = 0,
        int64_t dilation = 1,
        bool bias = true,
        const std::string& padding_mode = "zeros",
        bool with_bn = false
    ) : has_bn(with_bn) {
        auto [kh, kw] = kernel_size;
        
     
        torch::nn::detail::conv_padding_mode_t torch_padding_mode = torch::kZeros;
        if (padding_mode == "reflect") {
            torch_padding_mode = torch::kReflect;
        } else if (padding_mode == "replicate") {
            torch_padding_mode = torch::kReplicate;
        } else if (padding_mode == "circular") {
            torch_padding_mode = torch::kCircular;
        }

      
        pw = register_module("pw", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels, std::vector<int64_t>{1, 1})
                .stride(1)
                .padding(0)
                .dilation(1)
                .groups(1)
                .bias(false)
        ));

       
        if (with_bn) {
            bn = register_module("bn", torch::nn::BatchNorm2d(
                torch::nn::BatchNorm2dOptions(out_channels)
            ));
        }

        
        dw = register_module("dw", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(out_channels, out_channels, std::vector<int64_t>{kh, kw})
                .stride(stride)
                .padding(padding)
                .dilation(dilation)
                .groups(out_channels)
                .bias(bias)
                .padding_mode(torch_padding_mode)
        ));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = pw->forward(x);
        if (has_bn) {
            x = bn->forward(x);
        }
        x = dw->forward(x);
        return x;
    }
};

TORCH_MODULE(BSConvU);
