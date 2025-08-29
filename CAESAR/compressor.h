#ifndef COMPRESSOR_H
#define COMPRESSOR_H

#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include  <map>

class Compressor {
public:
    Compressor(const std::string& modelPath,
               bool useDiffusion = true,
               const std::string& device = "cuda",
               int nFrames = 16,
               int interpolationRate = 3,
               int diffusionSteps = 32);
    
   std::map<std::string, torch::Tensor> removeModulePrefix(const std::map<std::string, torch::Tensor>& stateDict);

private:

   void loadModels();
   void loadCaesarVCompressor();
   void loadCaesarDCompressor(); // later to add in 

    std::string pretrainedPath;
    bool useDiffusion;
    std::string device;
    int nFrames;
    int interpolationRate;
    int diffusionSteps;

    torch::Tensor condIdx;
    torch::Tensor predIdx;

    torch::jit::script::Module model;

};

#endif 

