#pragma once
#include <torch/torch.h>

namespace nn = torch::nn;

struct CandleCNNImpl final : nn::Module {
    nn::Conv2d conv1{nullptr}, conv2{nullptr};
    nn::Linear fc1{nullptr}, fc2{nullptr};

    explicit CandleCNNImpl(int64_t num_classes) {
        conv1 = register_module("conv1", nn::Conv2d(nn::Conv2dOptions(4,16,3).padding(1)));
        conv2 = register_module("conv2", nn::Conv2d(nn::Conv2dOptions(16,16,3).padding(1)));
        fc1 = register_module("fc1", nn::Linear(16 * 5 * 5, 128));
        fc2 = register_module("fc2", nn::Linear(128, num_classes));
    }
    torch::Tensor forward(torch::Tensor input) {
        input = torch::max_pool2d(torch::relu(conv1->forward(input)), 2);
        input = torch::max_pool2d(torch::relu(conv2->forward(input)), 2);
        return fc2->forward(torch::relu(fc1->forward(input.flatten(1))));
    }
};
TORCH_MODULE(CandleCNN);
