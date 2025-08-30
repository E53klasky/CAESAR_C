#include <tuple>
#include <torch/torch.h>
#include <torch/script.h>

/// probly test this I think the sytax is right 
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
normalizeLatent(const torch::Tensor& x) {
    auto xMin = torch::amin(x, {1, 2, 3, 4}, true);
    auto xMax = torch::amax(x, {1, 2, 3, 4}, true);

    // idk why this is +1e-8 something with no 0 I guess
    auto scale = (xMax - xMin + 1e-8) / 2;
    auto offset = xMin + scale;

    auto xNorm = (x - offset) / scale;

    return {xNorm, offset, scale};
}

