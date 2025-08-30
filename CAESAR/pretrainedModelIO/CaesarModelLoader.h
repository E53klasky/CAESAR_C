#ifndef CAESAR_MODEL_LOADER_H
#define CAESAR_MODEL_LOADER_H

#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

// External C compatibility for GNC compiler and different languages/ABI rules
//#ifdef __cplusplus
//extern "C" {
//#endif

/**
 * @class CaesarModelLoader
 * @brief A class for loading and managing pre-trained CAESAR model tensors
 * 
 * This class provides functionality to load PyTorch tensors from a directory,
 * manage them in memory, and provide various access methods for model weights.
 */
class CaesarModelLoader {
private:
    std::unordered_map<std::string, torch::Tensor> tensors;
    std::string tensor_dir;

public:
    /**
     * @brief Constructor
     * @param tensors_directory Path to the directory containing tensor files
     */
    explicit CaesarModelLoader(const std::string& tensors_directory);

    /**
     * @brief Load all tensors from the specified directory
     * @return true if at least one tensor was loaded successfully, false otherwise
     */
    bool loadAllTensors();

    /**
     * @brief Get a specific tensor by name
     * @param name The name of the tensor to retrieve
     * @return The requested tensor
     * @throws std::runtime_error if tensor is not found
     */
    torch::Tensor getTensor(const std::string& name) const;

    /**
     * @brief Check if a tensor with the given name exists
     * @param name The name to check
     * @return true if tensor exists, false otherwise
     */
    bool hasTensor(const std::string& name) const;

    /**
     * @brief Get all tensor names
     * @return Vector containing all tensor names
     */
    std::vector<std::string> getTensorNames() const;

    /**
     * @brief Get tensors that belong to a specific module
     * @param module_prefix The module prefix to filter by
     * @return Vector of pairs containing tensor names and tensors
     */
    std::vector<std::pair<std::string, torch::Tensor>> getTensorsByModule(const std::string& module_prefix) const;

    /**
     * @brief Print a summary of loaded tensors
     */
    void printSummary() const;

    /**
     * @brief Get the number of loaded tensors
     * @return Number of tensors currently loaded
     */
    size_t size() const;

    /**
     * @brief Get the tensor directory path
     * @return The directory path where tensors are stored
     */
    const std::string& getTensorDirectory() const;
};

//#ifdef __cplusplus
//}
//#endif

#endif // CAESAR_MODEL_LOADER_H
