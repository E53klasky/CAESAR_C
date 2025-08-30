#include "CaesarModelLoader.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>

CaesarModelLoader::CaesarModelLoader(const std::string& tensors_directory)
    : tensor_dir(tensors_directory) {}

bool CaesarModelLoader::loadAllTensors() {
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

torch::Tensor CaesarModelLoader::getTensor(const std::string& name) const {
    auto it = tensors.find(name);
    if (it != tensors.end()) {
        return it->second;
    } else {
        throw std::runtime_error("Tensor not found: " + name);
    }
}

bool CaesarModelLoader::hasTensor(const std::string& name) const {
    return tensors.find(name) != tensors.end();
}

std::vector<std::string> CaesarModelLoader::getTensorNames() const {
    std::vector<std::string> names;
    names.reserve(tensors.size());
    for (const auto& pair : tensors) {
        names.push_back(pair.first);
    }
    return names;
}

std::vector<std::pair<std::string, torch::Tensor>> CaesarModelLoader::getTensorsByModule(const std::string& module_prefix) const {
    std::vector<std::pair<std::string, torch::Tensor>> result;
    for (const auto& pair : tensors) {
        if (pair.first.find(module_prefix) == 0) {
            result.push_back(pair);
        }
    }
    return result;
}

void CaesarModelLoader::printSummary() const {
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

size_t CaesarModelLoader::size() const {
    return tensors.size();
}

const std::string& CaesarModelLoader::getTensorDirectory() const {
    return tensor_dir;
}
