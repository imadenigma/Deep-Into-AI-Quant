
#include <iostream>
#include <torch/torch.h>
#include "CandleCNN.h"

int main() {
    torch::Tensor X = torch::rand({2, 20});
    CandleCNN net(3);
    torch::NoGradGuard ng;                       // no autograd graph for a forward-only check
    auto out = net->forward(torch::randn({1, 4, 20, 20}));
    std::cout << out.sizes() << "\n" << out << "\n";   // [1, 3]
    return 0;
}
