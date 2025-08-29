#ifndef COMPRESSOR_H
#define COMPRESSOR_H

#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include  <map>
#include <utility>


struct CompressionResult {
  // A bit of a guess for right now 
    std::vector<float> latent;
    std::vector<uint8_t> postProcess;
    std::map<std::string, int> meta_data;
    std::vector<int> shape;
    std::vector<int> padding;
    std::vector<int> filteredBlocks;

};


// for compression method
using CompressReturn = std::pair<CompressionResult, int>;

class Compressor {
public:
    Compressor(const std::string& modelPath,
               bool useDiffusion = true,
               const std::string& device = "cuda",
               int nFrames = 16,
               int interpolationRate = 3,
               int diffusionSteps = 32);
    
   std::map<std::string, torch::Tensor> removeModulePrefix(const std::map<std::string, torch::Tensor>& stateDict);

   // what the hell is a dataloader look into it more this also needs to be saved
 //  CompressReturn compress(DataLoader& data)






private:

   void loadModels();
 
   //both here
   // py def compress(self, dataloader, eb = 1e-3):
   

   // CAESAR V methods ------------------------
   void loadCaesarVCompressor();






   // CAESAR D methods down ---------------------
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

