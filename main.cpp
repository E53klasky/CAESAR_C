#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <filesystem>

/* I think we should add external C here for conpatiblity with the GNC compiler so someone can use it for different lagnues and compile wiht other ABI rules i.e C,cpp,fortan


*/

class CaesarModelLoader {
private:
    std::unordered_map<std::string, torch::Tensor> tensors;
    std::string tensor_dir;

public:
    CaesarModelLoader(const std::string& tensors_directory) 
        : tensor_dir(tensors_directory) {}

    bool loadAllTensors() {
        try {
            std::cout << "Loading tensors from directory: " << tensor_dir << std::endl;

            
            if (!std::filesystem::exists(tensor_dir)) {
                std::cerr << "Directory does not exist: " << tensor_dir << std::endl;
                return false;
            }

            // Read the parameter mapping file to get original names
            std::string mapping_file = tensor_dir + "/parameter_map.txt";
            std::unordered_map<std::string, std::string> name_to_file;
            
            if (std::filesystem::exists(mapping_file)) {
                std::ifstream mapping(mapping_file);
                std::string line;
                while (std::getline(mapping, line)) {
                    if (line[0] == '#' || line.empty()) continue;
                    
                    size_t arrow_pos = line.find(" -> ");
                    if (arrow_pos != std::string::npos) {
                        std::string original_name = line.substr(0, arrow_pos);
                        std::string filename = line.substr(arrow_pos + 4);
                        name_to_file[original_name] = filename;
                    }
                }
                mapping.close();
            }

            int loaded_count = 0;
            int failed_count = 0;

            // Load each tensor file
            for (const auto& [original_name, filename] : name_to_file) {
                std::string full_path = tensor_dir + "/" + filename;
                
                try {
                    
                    std::vector<char> buffer;
                    std::ifstream file(full_path, std::ios::binary);
                    
                    if (!file.is_open()) {
                        std::cerr << "Cannot open tensor file: " << full_path << std::endl;
                        failed_count++;
                        continue;
                    }

                    file.seekg(0, std::ios::end);
                    size_t file_size = file.tellg();
                    file.seekg(0, std::ios::beg);

                    buffer.resize(file_size);
                    file.read(buffer.data(), file_size);
                    file.close();

                    // Load tensor using pickle_load
                    auto loaded_data = torch::pickle_load(buffer);
                    
                    if (loaded_data.isTensor()) {
                        tensors[original_name] = loaded_data.toTensor();
                        loaded_count++;
                        
                        if (loaded_count <= 5) {
                            std::cout << "Loaded " << original_name << " with shape: [";
                            auto tensor = tensors[original_name];
                            for (int64_t i = 0; i < tensor.dim(); ++i) {
                                if (i > 0) std::cout << ", ";
                                std::cout << tensor.size(i);
                            }
                            std::cout << "]" << std::endl;
                        } else if (loaded_count == 6) {
                            std::cout << "... (continuing to load remaining tensors)" << std::endl;
                        }
                    } else {
                        std::cerr << "Loaded data is not a tensor for: " << original_name << std::endl;
                        failed_count++;
                    }

                } catch (const std::exception& e) {
                    std::cerr << "Failed to load " << original_name << ": " << e.what() << std::endl;
                    failed_count++;
                }
            }

            std::cout << "Successfully loaded " << loaded_count << " tensors" << std::endl;
            if (failed_count > 0) {
                std::cout << "Failed to load " << failed_count << " tensors" << std::endl;
            }

            return loaded_count > 0;

        } catch (const std::exception& e) {
            std::cerr << "Error during tensor loading: " << e.what() << std::endl;
            return false;
        }
    }

    
    torch::Tensor getTensor(const std::string& name) const {
        auto it = tensors.find(name);
        if (it != tensors.end()) {
            return it->second;
        } else {
            throw std::runtime_error("Tensor not found: " + name);
        }
    }

   
    bool hasTensor(const std::string& name) const {
        return tensors.find(name) != tensors.end();
    }

 
    std::vector<std::string> getTensorNames() const {
        std::vector<std::string> names;
        for (const auto& pair : tensors) {
            names.push_back(pair.first);
        }
        return names;
    }

  
    std::vector<std::pair<std::string, torch::Tensor>> getTensorsByModule(const std::string& module_prefix) const {
        std::vector<std::pair<std::string, torch::Tensor>> result;
        for (const auto& pair : tensors) {
            if (pair.first.find(module_prefix) == 0) {
                result.push_back(pair);
            }
        }
        return result;
    }

    // Print summary
    void printSummary() const {
        std::cout << "\n=== CAESAR Model Summary ===" << std::endl;
        std::cout << "Total tensors loaded: " << tensors.size() << std::endl;


        std::unordered_map<std::string, int> module_counts;
        int64_t total_params = 0;

        for (const auto& pair : tensors) {
            total_params += pair.second.numel();
            
            // Extract main module name
            std::string name = pair.first;
            size_t first_dot = name.find('.');
            if (first_dot != std::string::npos) {
                std::string main_module = name.substr(0, first_dot);
                module_counts[main_module]++;
            }
        }

        std::cout << "Total parameters: " << total_params << std::endl;
        std::cout << "\nParameters by module:" << std::endl;
        for (const auto& pair : module_counts) {
            std::cout << "  " << pair.first << ": " << pair.second << " tensors" << std::endl;
        }
    }

    // Example of using tensors for inference (entropy model encoder layer)
    torch::Tensor applyEntropyEncLayer(const torch::Tensor& input, int layer_idx, int block_idx) const {
        try {
            // Get the weights and biases for this layer
            std::string base_name = "entropy_model.enc." + std::to_string(layer_idx) + "." + std::to_string(block_idx);
            
            if (hasTensor(base_name + ".block1.block.0.weight")) {
                auto weight = getTensor(base_name + ".block1.block.0.weight");
                auto bias = getTensor(base_name + ".block1.block.0.bias");
                
                std::cout << "Applying layer " << base_name << " with weight shape: [";
                for (int64_t i = 0; i < weight.dim(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << weight.size(i);
                }
                std::cout << "]" << std::endl;

                // Apply convolution (this is just an example - you'd need proper conv parameters)
                if (weight.dim() == 4) {
                    // 2D convolution
                    return torch::conv2d(input, weight, bias, /*stride=*/1, /*padding=*/1);
                } else if (weight.dim() == 5) {
                    // 3D convolution  
                    return torch::conv3d(input, weight, bias, /*stride=*/1, /*padding=*/1);
                }
            }
            
            throw std::runtime_error("Layer not found or unsupported: " + base_name);

        } catch (const std::exception& e) {
            std::cerr << "Error applying layer: " << e.what() << std::endl;
            throw;
        }
    }

    size_t size() const {
        return tensors.size();
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <tensors_directory>" << std::endl;
        std::cout << "Example: " << argv[0] << " /path/to/caesar_v_tensors" << std::endl;
        return 1;
    }

    std::string tensors_dir = argv[1];


    CaesarModelLoader loader(tensors_dir);


    if (!loader.loadAllTensors()) {
        std::cerr << "Failed to load tensors" << std::endl;
        return 1;
    }

 
    loader.printSummary();

  
    std::cout << "\n=== Example Tensor Access ===" << std::endl;

    try {
      
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


        std::cout << "\n=== Example Tensor Usage ===" << std::endl;
        

        auto tensor_names = loader.getTensorNames();
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
                std::cout << "Min value: " << tensor.min() << std::endl;
                std::cout << "Max value: " << tensor.max() << std::endl;
                std::cout << "Mean value: " << tensor.mean() << std::endl;
                break;
            }
        }

    } catch (const std::exception& e) {
        std::cout << "Error accessing tensors: " << e.what() << std::endl;
    }

    std::cout << "\n Successfully loaded and inspected CAESAR model tensors!" << std::endl;
    std::cout << "You now have access to all " << loader.size() << " model tensors in LibTorch." << std::endl;
    std::cout << "\nNext steps:" << std::endl;
    std::cout << "1. Use loader.getTensor(name) to get specific weights" << std::endl;
    std::cout << "2. Apply them in your neural network operations" << std::endl;
    std::cout << "3. Build your inference pipeline using these pre-trained weights" << std::endl;

    return 0;
}
