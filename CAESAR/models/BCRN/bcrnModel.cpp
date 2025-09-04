#include "bcrnModel.h"

// TODO: debug something is wrong here -----------------------------------------------
BluePrintShortcutBlockImpl::BluePrintShortcutBlockImpl(int64_t inChannels,
        int64_t outChannels, int64_t kernelSize){
    conv = bluePrintConvLayer(inChannels, outChannels, kernelSize);
    convNextBlock = Blocks(outChannels, kernelSize);
    esa = ESA(outChannels);
    cca = CCALayer(outChannels);
   
    register_module("conv", conv);
    register_module("convNextBlock", convNextBlock);
    register_module("esa", esa);
    register_module("cca", cca);
}

torch::Tensor BluePrintShortcutBlockImpl::forward(torch::Tensor x){
    std::cout << "=== BluePrintShortcutBlock forward start ===" << std::endl;
    std::cout << "Input to shortcut block: " << x.sizes() << std::endl;
    
    std::cout << "Starting conv layer..." << std::endl;
    x = conv->forward(x);
    std::cout << "After conv: " << x.sizes() << std::endl;

    std::cout << "Starting convNextBlock..." << std::endl;
    x = convNextBlock->forward(x);
      std::cout << "After convNextBlock: " << x.sizes() << std::endl;

     std::cout << "Starting ESA..." << std::endl;
    x = esa->forward(x); // fail???
   std::cout << "After ESA: " << x.sizes() << std::endl;
    
std::cout << "Starting CCA..." << std::endl;
   x = cca->forward(x);
 std::cout << "After CCA: " << x.sizes() << std::endl;

 std::cout<<"DOne BluePrintShortcutBlock foward"<<std::endl;


    return x;
}

    BluePrintConvNeXtSRImpl::BluePrintConvNeXtSRImpl(int64_t inChannels, int64_t outChannels,
                                                 int64_t upscaleFactor, int64_t baseChannels)
    : convNext1(BluePrintShortcutBlock(baseChannels, baseChannels, 3)),
      convNext2(BluePrintShortcutBlock(baseChannels, baseChannels, 3)),
      convNext3(BluePrintShortcutBlock(baseChannels, baseChannels, 3)),
      convNext4(BluePrintShortcutBlock(baseChannels, baseChannels, 3)),
      convNext5(BluePrintShortcutBlock(baseChannels, baseChannels, 3)),
      convNext6(BluePrintShortcutBlock(baseChannels, baseChannels, 3))
{

    conv1 = bluePrintConvLayer(inChannels, baseChannels, 3);
    conv2 = bluePrintConvLayer(baseChannels * 6, baseChannels, 3);
    upsampleBlock = PixelShuffleBlock(baseChannels, outChannels, upscaleFactor);
   
    act = activation("gelu");
    register_module("act", act.ptr());

    register_module("conv1", conv1);
    register_module("convNext1", convNext1);
    register_module("convNext2", convNext2);
    register_module("convNext3", convNext3);
    register_module("convNext4", convNext4);
    register_module("convNext5", convNext5);
    register_module("convNext6", convNext6);
    register_module("conv2", conv2);
    register_module("upsampleBlock", upsampleBlock);
}

torch::Tensor BluePrintConvNeXtSRImpl::forward(torch::Tensor x) {
     std::cout << "=== BluePrintConvNeXtSR forward start ===" << std::endl;
      std::cout << "Input x: " << x.sizes() << std::endl;
    
    auto outFea = conv1->forward(x);
        std::cout << "After conv1, outFea: " << outFea.sizes() << std::endl;
  
        std::cout << "Starting convNext1..." << std::endl;
    auto outC1 = convNext1->forward(outFea);
     std::cout << "After convNext1, outC1: " << outC1.sizes() << std::endl;

     std::cout<<"Starting ConvNext2..."<<std::endl;
    auto outC2 = convNext2->forward(outC1);
    std::cout<< "After convNexr2, outC2: "<<outC2.sizes() <<std::endl;

    std::cout<<"Staring convNext3..."<<std::endl;
    auto outC3 = convNext3->forward(outC2);
    std::cout<<"After convNext3, outC3: "<<outC3.sizes()<<std::endl;
    
    std::cout<<"Starting convNext4..."<<std::endl;
    auto outC4 = convNext4->forward(outC3);
    std::cout<<"After convNext4, outC4: "<<outC4.sizes()<<std::endl;

    std::cout<<"Starting convNext5..."<<std::endl;
    auto outC5 = convNext5->forward(outC4);
    std::cout<<"After convNext5, outC5" <<outC5.sizes()<<std::endl;
    
    std::cout<<"Starting convNext6..."<<std::endl;
    auto outC6 = convNext6->forward(outC5);
    std::cout<<"After convNext6, outC6: "<<outC6.sizes()<<std::endl;

    // NOTE: be cunation did not use act should be ok though ---------------------
    
    auto outCat = torch::cat({outC1, outC2, outC3, outC4, outC5, outC6}, 1);
    auto outC = act.forward(conv2->forward(outCat));
    auto outLr = outC + outFea;
    auto output = upsampleBlock->forward(outLr);
    std::cout<<"done"<<std::endl;
    return output;
}

