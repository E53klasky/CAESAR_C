#include "../CAESAR/pretrainedModelIO/CaesarModelLoader.h"
#include <iostream>
#include <string>

/**
 * @brief Test program for CaesarModelLoader class
 * 
 * This program demonstrates how to use the CaesarModelLoader to load
 * and inspect pre-trained CAESAR model tensors.
 */
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <tensors_directory>" << std::endl;
        std::cout << "Example: " << argv[0] << " /path/to/caesar_v_tensors" << std::endl;
        return 1;
    }

    std::string tensors_dir = argv[1];

    // Create the model loader instance
    CaesarModelLoader loader(tensors_dir);

    // Load all tensors from the directory
    if (!loader.loadAllTensors()) {
        std::cerr << "Failed to load tensors" << std::endl;
        return 1;
    }

    // Print summary statistics
    loader.printSummary();

    // Demonstrate tensor access methods
    std::cout << "\n=== Example Tensor Access ===" << std::endl;

    try {
        // Access entropy model tensors
        auto entropy_tensors = loader.getTensorsByModule("entropy_model");
        std::cout << "Found " << entropy_tensors.size() << " entropy model tensors" << std::endl;

        for (size_t i = 0; i < std::min(size_t(3), entropy_tensors.size()); ++i) {
            const auto& [name, tensor] = entropy_tensors[i];
            std::cout << "  " << name << ": [";
            for (int64_t j = 0; j < tensor.dim(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << tensor.size(j);
            }
            std::cout << "] " << tensor.dtype() << std::endl;
        }

        // Access SR model tensors
        auto sr_tensors = loader.getTensorsByModule("sr_model");
        std::cout << "\nFound " << sr_tensors.size() << " SR model tensors" << std::endl;

        for (size_t i = 0; i < std::min(size_t(3), sr_tensors.size()); ++i) {
            const auto& [name, tensor] = sr_tensors[i];
            std::cout << "  " << name << ": [";
            for (int64_t j = 0; j < tensor.dim(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << tensor.size(j);
            }
            std::cout << "] " << tensor.dtype() << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Error accessing module tensors: " << e.what() << std::endl;
    }

    // Demonstrate individual tensor access and statistics
    std::cout << "\n=== Example Tensor Usage ===" << std::endl;

    try {
        auto tensor_names = loader.getTensorNames();
        
        // Find and analyze a convolution weight tensor
        for (const auto& name : tensor_names) {
            if (name.find("conv") != std::string::npos && name.find("weight") != std::string::npos) {
                auto tensor = loader.getTensor(name);
                std::cout << "Found conv weight: " << name << std::endl;
                std::cout << "Shape: [";
                for (int64_t i = 0; i < tensor.dim(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << tensor.size(i);
                }
                std::cout << "]" << std::endl;
                std::cout << "Data type: " << tensor.dtype() << std::endl;
                std::cout << "Device: " << tensor.device() << std::endl;
                std::cout << "Min value: " << tensor.min() << std::endl;
                std::cout << "Max value: " << tensor.max() << std::endl;
                std::cout << "Mean value: " << tensor.mean() << std::endl;
                std::cout << "Standard deviation: " << tensor.std() << std::endl;
                break;
            }
        }

        // Test tensor existence check
        std::cout << "\n=== Testing Tensor Existence ===" << std::endl;
        std::string test_name = tensor_names.empty() ? "nonexistent" : tensor_names[0];
        std::cout << "Does tensor '" << test_name << "' exist? " << 
                     (loader.hasTensor(test_name) ? "Yes" : "No") << std::endl;
        
        if (!tensor_names.empty()) {
            std::cout << "Does tensor 'definitely_nonexistent' exist? " << 
                         (loader.hasTensor("definitely_nonexistent") ? "Yes" : "No") << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Error accessing individual tensors: " << e.what() << std::endl;
    }

    // Final summary
    std::cout << "\n=== Test Results ===" << std::endl;
    std::cout << "Successfully loaded and inspected CAESAR model tensors!" << std::endl;
    std::cout << "Total tensors loaded: " << loader.size() << std::endl;
    std::cout << "Tensor directory: " << loader.getTensorDirectory() << std::endl;
    
    std::cout << "\nNext steps:" << std::endl;
    std::cout << "1. Use loader.getTensor(name) to get specific weights" << std::endl;
    std::cout << "2. Apply them in your neural network operations" << std::endl;
    std::cout << "3. Build your inference pipeline using these pre-trained weights" << std::endl;

    return 0;
}
