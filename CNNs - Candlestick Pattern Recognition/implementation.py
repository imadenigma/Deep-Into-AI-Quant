import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, X):
        return self.output(F.relu(self.hidden(X)))

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
    def forward(self, X):
        for module in self.children():
            X = module(X)
        return X

mlp = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
# Random Initialization
def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

X = torch.rand(2, 20)

_ = mlp(X)
mlp.apply(my_init)
