// testBcrn.cpp
#include <iostream>
#include <torch/torch.h>
#include "../CAESAR/models/BCRN/bcrnModel.h"

int main() {
    try {
        // Example: Super-resolution with input channels = 3 (RGB), output channels = 3
        BluePrintConvNeXtSR model(3, 3, 2, 64);

        // Dummy input: batch=1, channels=3, height=16, width=16
        auto input = torch::randn({1, 3, 16, 16});

        auto output = model->forward(input);

        std::cout << "Input shape: " << input.sizes() << std::endl;
        std::cout << "Output shape: " << output.sizes() << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error during forward pass: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

