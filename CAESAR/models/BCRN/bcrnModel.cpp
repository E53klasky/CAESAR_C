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
    x = conv->forward(x);
    x = convNextBlock->forward(x);
    x = esa->forward(x);
    x = cca->forward(x);
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
    auto outFea = conv1->forward(x);
    auto outC1 = convNext1->forward(outFea);
    auto outC2 = convNext2->forward(outC1);
    auto outC3 = convNext3->forward(outC2);
    auto outC4 = convNext4->forward(outC3);
    auto outC5 = convNext5->forward(outC4);
    auto outC6 = convNext6->forward(outC5);

    // NOTE: be cunation did not use act should be ok though ---------------------
    auto outCat = torch::cat({outC1, outC2, outC3, outC4, outC5, outC6}, 1);
    auto outC = act.forward(conv2->forward(outCat));
    auto outLr = outC + outFea;
    auto output = upsampleBlock->forward(outLr);
    return output;
}

