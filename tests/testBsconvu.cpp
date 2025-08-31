#include <torch/torch.h>
#include <cassert>
#include <iostream>
#include <iomanip>
#include "../CAESAR/models/BCRN/BSConvU.h" // Adjust path as needed

void test_basic_construction() {
    std::cout << "Testing basic construction..." << std::endl;
    
    // Test basic construction
    BSConvU model(16, 32, std::make_tuple(3, 3));
    assert(model.get() != nullptr);
    std::cout << "  Basic construction passed" << std::endl;
}

void test_forward_pass() {
    std::cout << "Testing forward pass..." << std::endl;
    
    BSConvU model(16, 32, std::make_tuple(3, 3), 1, 1);
    
    // Create input tensor: batch_size=2, channels=16, height=32, width=32
    torch::Tensor input = torch::randn({2, 16, 32, 32});
    
    // Forward pass - now we can use -> since we have explicit forward method
    torch::Tensor output = model->forward(input);
    
    // Check output shape
    assert(output.dim() == 4);
    assert(output.size(0) == 2);  // batch size
    assert(output.size(1) == 32); // output channels
    assert(output.size(2) == 32); // height (same due to padding=1, kernel=3, stride=1)
    assert(output.size(3) == 32); // width
    
    std::cout << "  Forward pass shape test passed" << std::endl;
    std::cout << "  Input shape: " << input.sizes() << std::endl;
    std::cout << "  Output shape: " << output.sizes() << std::endl;
}

void test_different_parameters() {
    std::cout << "Testing different parameters..." << std::endl;
    
    // Test with stride=2
    {
        BSConvU model(16, 32, std::make_tuple(3, 3), 2, 1);
        torch::Tensor input = torch::randn({1, 16, 32, 32});
        torch::Tensor output = model->forward(input);
        
        assert(output.size(0) == 1);
        assert(output.size(1) == 32);
        assert(output.size(2) == 16); // 32/2 due to stride=2
        assert(output.size(3) == 16);
        std::cout << "✓ Stride=2 test passed" << std::endl;
    }
    
    // Test with different kernel size
    {
        BSConvU model(8, 16, std::make_tuple(5, 5), 1, 2);
        torch::Tensor input = torch::randn({1, 8, 32, 32});
        torch::Tensor output = model->forward(input);
        
        assert(output.size(0) == 1);
        assert(output.size(1) == 16);
        assert(output.size(2) == 32); // Same due to padding=2, kernel=5
        assert(output.size(3) == 32);
        std::cout << "✓ Different kernel size test passed" << std::endl;
    }
}

void test_with_batchnorm() {
    std::cout << "Testing with batch normalization..." << std::endl;
    
    BSConvU model_with_bn(16, 32, std::make_tuple(3, 3), 1, 1, 1, true, "zeros", true);
    BSConvU model_without_bn(16, 32, std::make_tuple(3, 3), 1, 1, 1, true, "zeros", false);
    
    torch::Tensor input = torch::randn({2, 16, 32, 32});
    
    // Both should work
    torch::Tensor output_with_bn = model_with_bn->forward(input);
    torch::Tensor output_without_bn = model_without_bn->forward(input);
    
    // Same output shape
    assert(output_with_bn.sizes() == output_without_bn.sizes());
    std::cout << "  Batch normalization test passed" << std::endl;
}

void test_padding_modes() {
    std::cout << "Testing different padding modes..." << std::endl;
    
    // Test all padding modes
    std::vector<std::string> padding_modes = {"zeros", "reflect", "replicate", "circular"};
    
    for (const auto& mode : padding_modes) {
        BSConvU model(16, 32, std::make_tuple(3, 3), 1, 1, 1, true, mode);
        torch::Tensor input = torch::randn({1, 16, 16, 16});
        torch::Tensor output = model->forward(input);
        
        assert(output.size(0) == 1);
        assert(output.size(1) == 32);
        assert(output.size(2) == 16);
        assert(output.size(3) == 16);
        
        std::cout << " Padding mode '" << mode << "' test passed" << std::endl;
    }
}

void test_gradient_flow() {
    std::cout << "Testing gradient flow..." << std::endl;
    
    BSConvU model(8, 16, std::make_tuple(3, 3));
    
    torch::Tensor input = torch::randn({1, 8, 16, 16}, torch::requires_grad());
    torch::Tensor output = model->forward(input);
    
    // Compute a simple loss
    torch::Tensor loss = output.sum();
    loss.backward();
    
    // Check that gradients exist
    assert(input.grad().defined());
    assert(input.grad().sizes() == input.sizes());
    
    std::cout << " Gradient flow test passed" << std::endl;
}

void test_parameter_count() {
    std::cout << "Testing parameter count..." << std::endl;
    
    int64_t in_channels = 16;
    int64_t out_channels = 32;
    int64_t kh = 3, kw = 3;
    
    BSConvU model(in_channels, out_channels, std::make_tuple(kh, kw));
    
    auto parameters = model->parameters();
    
    // Should have parameters from both conv layers
    // Conv1: in_channels * out_channels * 1 * 1 (no bias)
    // Conv2: out_channels * 1 * kh * kw + out_channels (if bias=true)
    
    int param_count = 0;
    for (const auto& param : parameters) {
        param_count += param.numel();
    }
    
    // Expected: 16*32*1*1 + 32*1*3*3 + 32 = 512 + 288 + 32 = 832
    int expected = in_channels * out_channels + out_channels * kh * kw + out_channels;
    
    std::cout << "  Parameter count: " << param_count << std::endl;
    std::cout << "  Expected: " << expected << std::endl;
    assert(param_count == expected);
    
    std::cout << "✓ Parameter count test passed" << std::endl;
}

void test_module_structure() {
    std::cout << "Testing module structure..." << std::endl;

    BSConvU model_without_bn(16, 32, std::make_tuple(3, 3), 1, 1, 1, true, "zeros", false);
    BSConvU model_with_bn(16, 32, std::make_tuple(3, 3), 1, 1, 1, true, "zeros", true);

    // Model without BN should have 2 children (pw, dw)
    auto children_without_bn = model_without_bn->children();
    int count_without_bn = 0;
    for (auto& child : children_without_bn) {
        count_without_bn++;
    }
    assert(count_without_bn == 2);  // pw and dw

    // Model with BN should have 3 children (pw, bn, dw)
    auto children_with_bn = model_with_bn->children();
    int count_with_bn = 0;
    for (auto& child : children_with_bn) {
        count_with_bn++;
    }
    assert(count_with_bn == 3);  // pw, bn, and dw

    std::cout << "  Module structure test passed" << std::endl;
    std::cout << "  Without BN: " << count_without_bn << " modules" << std::endl;
    std::cout << "  With BN: " << count_with_bn << " modules" << std::endl;
}


int main() {
    std::cout << "Running BSConvU tests..." << std::endl;
    std::cout << "=========================" << std::endl;
    
    try {
        test_basic_construction();
        test_forward_pass();
        test_different_parameters();
        test_with_batchnorm();
        test_padding_modes();
        test_gradient_flow();
        test_parameter_count();
        test_module_structure();
        
        std::cout << std::endl;
        std::cout << "All tests passed successfully!" << std::endl;
        std::cout << "Your BSConvU implementation is working correctly." << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "Test failed with unknown exception" << std::endl;
        return 1;
    }
    
    return 0;
}
