//
// Created by imad on 6/12/26.
//
#include <iostream>
#include <vector>
#include <fstream>
#include <torch/torch.h>
#include "CandleCNN.h"

template <typename T>
std::vector<T> read_bin(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    TORCH_CHECK(file.good(), "Could not open file " + path);
    auto bytes = file.tellg();
    file.seekg(0);
    std::vector<T> buf(bytes / sizeof(T));
    file.read(reinterpret_cast<char*>(buf.data()), bytes);
    return buf;

}

int main() {
    const std::string DATA = "/home/imad/CLionProjects/Deep-Into-AI-Quant/CNNs - Candlestick Pattern Recognition/";
    auto xb = read_bin<float>(DATA + "X_gaf.bin");
    auto yb = read_bin<int64_t>(DATA + "y.bin");
    const int64_t N = static_cast<int64_t>(yb.size());

    auto X = torch::from_blob(xb.data(), {N, 4, 20, 20}, torch::kFloat32).clone();
    auto y = torch::from_blob(yb.data(), {N}, torch::kInt64).clone();

    const int64_t n_train = N * 0.8; // train/test split 80;20
    auto Xtr = X.slice(0, 0, n_train), ytr = y.slice(0, 0, n_train);
    auto Xtest = X.slice(0, n_train, N), ytest = y.slice(0, n_train, N);

    // ------ MODEL / LOSS / OPTIMIZER ---------//
    CandleCNN net(3);
    auto opt = torch::optim::Adam(net->parameters(), torch::optim::AdamOptions(1e-3));
    const int64_t B = 64, epochs = 16;

    for (int64_t epoch = 1; epoch <= epochs; ++epoch) {
        // ---- Train ---- //
        net->train();
        auto perm = torch::randperm(n_train);
        double loss_sum = 0.0;
        for (int64_t i = 0; i < n_train; i += B) {
            auto idx = perm.slice(0, i, std::min(i + B, n_train));
            auto xb = Xtr.index_select(0, idx);
            auto yb = ytr.index_select(0, idx);

            opt.zero_grad();
            auto loss = torch::cross_entropy_loss(net->forward(xb), yb);
            loss.backward();
            opt.step();
            loss_sum += loss.item<double>() * idx.size(0);

        }
        // Validation (TEST)
        net->eval();
        torch::NoGradGuard ng;
        auto logits = net->forward(Xtest);
        auto acc = (logits.argmax(1) == ytest).to(torch::kFloat32).mean().item<double>();
        std::cout << "epoch " << epoch
                  << "  train_loss " << loss_sum / n_train
                  << "  val_acc " << acc << "\n";

    }
}

