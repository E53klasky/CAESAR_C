#include <torch/torch.h>
#include <iostream>
#include "../CAESAR/models/BCRN/blocks.h"

void run_test(const std::string& padType,
              const std::string& normType,
              const std::string& actType,
              int stride = 1,
              bool bias = true) {
    auto x = torch::randn({2, 3, 32, 32});  // [N, C, H, W], bigger input
    auto block = convBlock(
        3, 16, 3,     // in=3, out=16, kernel=3
        stride, 1, 1, // stride, dilation, groups
        bias,
        padType, normType, actType
    );
    auto y = block->forward(x);

    std::cout << "==== Test ====" << std::endl;
    std::cout << "Pad=" << padType
              << ", Norm=" << (normType.empty() ? "None" : normType)
              << ", Act=" << (actType.empty() ? "None" : actType)
              << ", Stride=" << stride
              << ", Bias=" << (bias ? "true" : "false")
              << std::endl;
    std::cout << "Input shape: " << x.sizes()
              << " -> Output shape: " << y.sizes() << std::endl;
    std::cout << "--------------" << std::endl;
}

int main() {
    torch::manual_seed(42);

    // Padding types
    run_test("zero", "", "relu");
    run_test("reflect", "", "relu");
    run_test("replicate", "", "relu");

    // Activations
    run_test("zero", "", "");
    run_test("zero", "", "relu");
    run_test("zero", "", "lrelu");
    run_test("zero", "", "prelu");

    // Norms (enable once you implement norm())
    // run_test("zero", "batch", "relu");
    // run_test("zero", "instance", "relu");

    // Stride and bias variations
    run_test("zero", "", "relu", 2, true);   // stride=2
    run_test("zero", "", "relu", 1, false);  // bias=false

    return 0;
}

