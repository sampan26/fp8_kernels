#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/extension.h>
#include <torch/python.h>

at::Tensor rms_norm(at::Tensor &x, at::Tensor& weights);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("rms_norm", &rms_norm, "caculate rms norm");
}