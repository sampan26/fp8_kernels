#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <torch/extension.h>
#include <vector>


torch::Tensor q8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor a_scales, torch::Tensor b_scales, bool fuse_gelu);
