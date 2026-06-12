# Builders' Guide in C++ (LibTorch) — Part 1: Layers and Modules

> Faithful C++ rewrite of d2l Chapter 6, §1 (`model-construction`), using **LibTorch** (the PyTorch C++ API).
> Each Python concept is translated *and* the framework machinery underneath it is explained.

---

## 0. Orientation: the two ideas that make C++ modules work

The Python book leans on two things that happen *invisibly*. In C++ they become explicit, so understanding them once explains every class in the whole chapter.

### Idea 1 — Reference semantics via a "holder"

In Python, `net = MLP()` gives you a reference. Passing it around, storing it in a list, or assigning it never copies the parameters. C++ defaults to *value* semantics (assigning copies), which is wrong for neural nets — you want every reference to point at the *same* weights.

LibTorch solves this with the **Impl + holder** pattern:

- You write your module logic in a class named `XxxImpl` that derives from `torch::nn::Module`.
- The macro `TORCH_MODULE(Xxx)` generates a thin wrapper `Xxx` that is essentially a `std::shared_ptr<XxxImpl>` with `operator->`.

So `MLP` is a shared pointer; `MLP a = b;` shares the same underlying weights, exactly like Python. You access members with `->` (`net->forward(X)`), because `net` is a (smart) pointer.

### Idea 2 — Explicit registration

Python's `self.hidden = nn.Linear(...)` triggers `nn.Module.__setattr__`, which secretly records the layer in an internal dict. That dict is what powers `net.parameters()`, `net.to(device)`, `net.state_dict()`, etc.

C++ has no `__setattr__`. So you must *tell* the parent module about each sub-module or tensor:

| Python (automatic) | LibTorch (explicit) | What it tracks |
|---|---|---|
| `self.layer = nn.Linear(...)` | `register_module("layer", ...)` | a sub-module (recurses into it) |
| a learnable tensor attribute | `register_parameter("w", ...)` | a leaf parameter (gets gradients) |
| a constant tensor attribute | `register_buffer("c", ...)` | state that is *not* trained |

If you forget to register something, it still works in `forward`, but it becomes invisible to `parameters()`, the optimizer, `.to(cuda)`, and saving. That invisibility is the #1 LibTorch beginner bug — and the reason Part 2 (parameters) and Part 6 (save/load) work at all.

### One real difference: no `LazyLinear`

The Python book uses `nn.LazyLinear(256)` — it infers the input width on the first forward pass. LibTorch has no lazy linear layer, so we pass **both** dimensions: `torch::nn::Linear(in_features, out_features)`. Keep this in mind; it is exactly the topic of Part 4 (Lazy Initialization), where we explain *why* the framework can defer this and how to emulate it in C++.

Our running input is `X` of shape `(2, 20)`, so the first layer is `Linear(20, ...)`.

---

## 1. The built-in `Sequential`

**Python**
```python
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
X = torch.rand(2, 20)
net(X).shape          # torch.Size([2, 10])
```

**C++ (LibTorch)**
```cpp
#include <torch/torch.h>
#include <iostream>

int main() {
    torch::nn::Sequential net(
        torch::nn::Linear(20, 256),
        torch::nn::ReLU(),
        torch::nn::Linear(256, 10)
    );

    torch::Tensor X = torch::rand({2, 20});
    std::cout << net->forward(X).sizes() << std::endl;   // [2, 10]
}
```

What to notice:

- `torch::nn::Sequential` is itself a module holder, so we call through it with `net->forward(X)`. (The `->` is the holder dereference from Idea 1.)
- We replaced `net(X)` with `net->forward(X)`. Python's `net(X)` is sugar for `__call__`, which runs hooks and then `forward`. In C++ you normally call `forward` directly.
- `.shape` becomes `.sizes()`, which returns an `IntArrayRef` (prints as `[2, 10]`).
- `nn.ReLU()` → `torch::nn::ReLU()`. It is a real (parameter-free) module, just like in Python.

Under the hood, `Sequential` stores its children type-erased and, in its own `forward`, threads the tensor through each child in order — which is precisely what we reimplement by hand in §3.

---

## 2. A custom module: `MLP`

This is the heart of the section: subclass the base module, declare sub-layers, and define `forward`.

**Python**
```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.out = nn.LazyLinear(10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
```

**C++ (LibTorch)**
```cpp
#include <torch/torch.h>

struct MLPImpl : torch::nn::Module {
    // Members are empty holders until we assign them in the constructor.
    torch::nn::Linear hidden{nullptr}, out{nullptr};

    MLPImpl() {
        // register_module is the explicit version of Python's `self.x = ...`.
        // It (a) stores the child so parameters()/to()/save() can find it,
        // and (b) returns the layer so we can keep our own typed handle.
        hidden = register_module("hidden", torch::nn::Linear(20, 256));
        out    = register_module("out",    torch::nn::Linear(256, 10));
    }

    torch::Tensor forward(torch::Tensor X) {
        return out->forward(torch::relu(hidden->forward(X)));
    }
};
TORCH_MODULE(MLP);   // generates the `MLP` holder around MLPImpl
```

Use it:
```cpp
MLP net;                                  // default-constructs MLPImpl inside a shared_ptr
std::cout << net->forward(X).sizes();     // [2, 10]
```

Line-by-line mapping:

- `class MLP(nn.Module)` splits into `struct MLPImpl : torch::nn::Module` **plus** `TORCH_MODULE(MLP)`. The split is the price of reference semantics (Idea 1).
- `super().__init__()` is automatic — the base `torch::nn::Module` default constructor runs for you. There is nothing to call.
- `self.hidden = nn.Linear(...)` → `hidden = register_module("hidden", ...)`. The **string name** ("hidden") is what you'll see later in `named_parameters()` (e.g. `hidden.weight`) and in saved files. In Python that name came for free from the attribute name; in C++ you supply it.
- `F.relu(...)` → `torch::relu(...)`. The `torch::` free functions are the C++ equivalent of `torch.nn.functional`.
- `self.out(...)` → `out->forward(...)`. Again, `->forward` because `out` is a holder.

Why declare the members with `{nullptr}`? A `torch::nn::Linear` holder cannot be default-constructed empty *and* valid; `{nullptr}` makes a deliberately empty holder that we fill in the constructor body. This is the standard LibTorch idiom.

The conceptual payoff is identical to Python: instantiating `MLP net1; MLP net2;` gives two independent sets of weights, because each `MLP()` runs the constructor and builds fresh `Linear` layers.

---

## 3. Reimplementing `Sequential` → `MySequential`

The book builds a minimal `Sequential` to show there's no magic: store children in order, then chain them in `forward`.

**Python**
```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, X):
        for module in self.children():
            X = module(X)
        return X
```

**C++ (LibTorch)**
```cpp
#include <torch/torch.h>
#include <vector>
#include <string>

struct MySequentialImpl : torch::nn::Module {
    // Why AnyModule? See explanation below.
    std::vector<torch::nn::AnyModule> layers_;

    MySequentialImpl(std::vector<torch::nn::AnyModule> layers)
        : layers_(std::move(layers)) {
        for (size_t i = 0; i < layers_.size(); ++i) {
            // The numeric string name mirrors Python's add_module(str(idx), ...)
            register_module(std::to_string(i), layers_[i].ptr());
        }
    }

    torch::Tensor forward(torch::Tensor X) {
        for (auto& layer : layers_) {
            X = layer.forward<torch::Tensor>(X);
        }
        return X;
    }
};
TORCH_MODULE(MySequential);
```

Use it:
```cpp
MySequential net(std::vector<torch::nn::AnyModule>{
    torch::nn::AnyModule(torch::nn::Linear(20, 256)),
    torch::nn::AnyModule(torch::nn::ReLU()),
    torch::nn::AnyModule(torch::nn::Linear(256, 10))
});
std::cout << net->forward(X).sizes();   // [2, 10]
```

The one genuinely new C++ concept here: **`torch::nn::AnyModule`**.

In Python, `for module in self.children(): X = module(X)` works because Python is dynamically typed — any object with a `forward` will do. C++ is statically typed, and the base `torch::nn::Module` deliberately has **no** `forward` (different layers have different signatures). So you cannot loop over heterogeneous modules and call `forward` on the base type.

`AnyModule` is LibTorch's answer: a **type-erased** wrapper that remembers how to call the contained module's `forward`. `layer.forward<torch::Tensor>(X)` says "call the wrapped module's forward, expecting a `torch::Tensor` back." This is the C++ stand-in for Python's duck typing.

Two registration details that map straight back to the Python:

- `register_module(std::to_string(i), layers_[i].ptr())` — `.ptr()` gets the underlying `shared_ptr<Module>` so the parent can track it. This is the literal counterpart of `add_module(str(idx), module)`. Storing the layers in a plain `std::vector` *without* registering would be the C++ version of "store modules in a Python list" — and would break `parameters()` and `.to(device)`. (That's the book's Exercise 1, and now you can see exactly why it breaks.)

Behaviorally this is indistinguishable from §1's built-in `Sequential`, which is the whole point: the built-in one does the same loop, just with nicer construction syntax.

---

## 4. Running arbitrary code in `forward`: `FixedHiddenMLP`

Sequential only daisy-chains. Real architectures need control flow, constants, and reused layers inside `forward`. This class demonstrates all three.

**Python**
```python
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20))   # constant, not trained
        self.linear = nn.LazyLinear(20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(X @ self.rand_weight + 1)
        X = self.linear(X)                         # reuse => weight sharing
        while X.abs().sum() > 1:                   # control flow
            X /= 2
        return X.sum()
```

**C++ (LibTorch)**
```cpp
#include <torch/torch.h>

struct FixedHiddenMLPImpl : torch::nn::Module {
    torch::Tensor rand_weight;        // a constant tensor (a "buffer")
    torch::nn::Linear linear{nullptr};

    FixedHiddenMLPImpl() {
        // register_buffer: state that travels with the module (saved, moved to
        // GPU) but is NOT a parameter -> no gradient, never touched by the optimizer.
        rand_weight = register_buffer("rand_weight", torch::rand({20, 20}));
        linear = register_module("linear", torch::nn::Linear(20, 20));
    }

    torch::Tensor forward(torch::Tensor X) {
        X = linear->forward(X);
        X = torch::relu(torch::matmul(X, rand_weight) + 1);
        X = linear->forward(X);                       // reuse => weight sharing
        while (X.abs().sum().item<float>() > 1.0f) {  // control flow
            X = X / 2;
        }
        return X.sum();
    }
};
TORCH_MODULE(FixedHiddenMLP);
```

The three teaching points, made explicit:

1. **Constant tensors → buffers.** The Python code sets `self.rand_weight = torch.rand(...)` as a *plain attribute*. That happens to keep it out of `parameters()` (so no gradients — the intended effect), but it's also subtly broken: a plain attribute will **not** follow the module to the GPU on `net.to(cuda)`. The correct, faithful-in-spirit C++ is `register_buffer`, which gives you both properties on purpose: excluded from training, included in `.to(device)` and in saved state. This buffer-vs-parameter distinction is the core of Part 2.

2. **Weight sharing is just calling the same layer twice.** `linear->forward(X)` appears twice and refers to the *same* weights — exactly like Python. No copying happens (Idea 1). Gradients from both uses accumulate into the one weight tensor.

3. **Control flow needs an explicit scalar extraction.** Python's `while X.abs().sum() > 1` quietly converts a one-element tensor to a Python bool. C++ won't auto-convert a `Tensor` to `bool`, so you pull the scalar out with `.item<float>()`. `X @ W` becomes `torch::matmul(X, W)`; `X /= 2` becomes `X = X / 2`.

Run:
```cpp
FixedHiddenMLP net;
std::cout << net->forward(X) << std::endl;   // a 0-dim (scalar) tensor
```

---

## 5. Nesting modules: `NestMLP` and a "chimera"

Modules compose recursively: a module can hold a `Sequential`, which holds layers; and that whole thing can sit inside another `Sequential`. Registration recurses, so `parameters()` on the outermost module still finds everything.

**Python**
```python
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(64), nn.ReLU(),
                                 nn.LazyLinear(32), nn.ReLU())
        self.linear = nn.LazyLinear(16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.LazyLinear(20), FixedHiddenMLP())
chimera(X)
```

**C++ (LibTorch)**
```cpp
struct NestMLPImpl : torch::nn::Module {
    torch::nn::Sequential net{nullptr};
    torch::nn::Linear linear{nullptr};

    NestMLPImpl() {
        net = register_module("net", torch::nn::Sequential(
            torch::nn::Linear(20, 64), torch::nn::ReLU(),
            torch::nn::Linear(64, 32), torch::nn::ReLU()));
        linear = register_module("linear", torch::nn::Linear(32, 16));
    }

    torch::Tensor forward(torch::Tensor X) {
        return linear->forward(net->forward(X));
    }
};
TORCH_MODULE(NestMLP);

// A custom module drops straight into a built-in Sequential, because our holder
// has a forward(Tensor)->Tensor that Sequential can type-erase into an AnyModule.
torch::nn::Sequential chimera(
    NestMLP(),
    torch::nn::Linear(16, 20),
    FixedHiddenMLP()
);
std::cout << chimera->forward(X) << std::endl;
```

Notice the dimensions are now explicit and must line up by hand (the cost of no `LazyLinear`): `NestMLP` maps `20 -> 64 -> 32 -> 16`, then `Linear(16, 20)`, then `FixedHiddenMLP` (its internal `Linear(20, 20)`), which collapses to a scalar.

The deep point: `chimera` can mix a hand-written `NestMLP`, a built-in `Linear`, and the control-flow-heavy `FixedHiddenMLP` because **they are all just `torch::nn::Module`s**. Registration nests, so `chimera->parameters()` returns the trainable weights from all three, however deeply buried — and the `rand_weight` buffer is correctly *excluded* from that list. That recursion is the whole reason the abstraction scales to hundred-layer networks.

---

## 6. Build & run

`CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.18)
project(d2l_cpp)
find_package(Torch REQUIRED)
add_executable(part1 part1.cpp)
target_link_libraries(part1 "${TORCH_LIBRARIES}")
set_property(TARGET part1 PROPERTY CXX_STANDARD 17)
```

Build (point `CMAKE_PREFIX_PATH` at your unzipped LibTorch):
```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
./part1
```

Download LibTorch from pytorch.org (CPU or CUDA build). Everything above compiles against a standard release.

---

## 7. Exercises (the book's, translated to C++ thinking)

1. **What breaks if `MySequential` stores modules in a plain list instead of registering them?** In C++ terms: keep the `std::vector<AnyModule>` but delete the `register_module` loop. `forward` still runs, but `parameters()` returns nothing, the optimizer updates nothing, and `to(cuda)`/save ignore the layers. Registration *is* the bookkeeping.
2. **Parallel module.** Write `ParallelImpl` holding `net1` and `net2` (both registered), and in `forward` return `torch::cat({net1->forward(X), net2->forward(X)}, /*dim=*/1)`.
3. **Factory of repeated blocks.** Write a function returning a `torch::nn::Sequential` built in a loop (`for (...) seq->push_back(block_factory());`). `Sequential::push_back` is the C++ analog of repeatedly adding the same block.

---

## What's next

**Part 2 — Parameter Management.** Now that registration is clear, we'll use it: accessing weights by name (`named_parameters()`), reading a specific layer's `weight`/`bias`, tying parameters across layers, and the parameter-vs-buffer line we previewed in §4. Everything there is just *reading the registry* we built in this part.
