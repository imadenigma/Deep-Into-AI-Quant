cmake_minimum_required(VERSION 3.18)
project(Deep_Into_AI_Quant)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "/home/imad/libtorch")

find_package(Torch REQUIRED)
set(CMAKE_BUILD_RPATH "/home/imad/libtorch/lib")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)



add_executable(Deep_Into_AI_Quant
        "CNNs - Candlestick Pattern Recognition/main.cpp"
        "CNNs - Candlestick Pattern Recognition/CandleCNN.h"
)

target_include_directories(Deep_Into_AI_Quant PRIVATE "${TORCH_INCLUDE_DIRS}")
include_directories("CNNs - Candlestick Pattern Recognition")
target_link_libraries(Deep_Into_AI_Quant "${TORCH_LIBRARIES}")
set_property(TARGET Deep_Into_AI_Quant PROPERTY CXX_STANDARD 17)
target_precompile_headers(Deep_Into_AI_Quant PRIVATE <torch/torch.h>)
