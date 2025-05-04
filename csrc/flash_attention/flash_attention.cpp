#include <torch/extension.h>
#include <torch/python.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

void flash_attention_cuda_mask(void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, int* BatchMask,
                                size_t bs, size_t nh, size_t N, size_t dim, float softmax_scale, cudaStream_t stream);

void flash_attention_cuda(void* Q_ptr, void* K_ptr, void* V_ptrx, void* O_ptr,
                                size_t bs, size_t nh, size_t N, size_t dim,  float softmax_scale, cudaStream_t stream);

tensor::Tensor flash_attention(torch::Tensor& q, torch::Tensor& k, torch::Tensor &v, 
                                float softmax_scale, c10::optional<at::Tensor>& batch_mask_) {
    int bs = q.size(0);
    int nh = q.size(1);
    int N = q.size(2);
    int d = q.size(3);
    auto opts = q.options();

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream.stream();

    auto out = torch::empty_like(q);
    
    if (batch_mask_.has_value()) {
        at::Tensor batch_mask;
        batch_mask = batch_mask_.value();
        TORCH_CHECK(batch_mask.dtype() == torch::kInt32, "flash attn: mask type error");
        TORCH_CHECK(batch_mask.size(0) == bs, "flash attn: mask batch error");
        flash_attention_cuda_mask(q.data_ptr(),
                                  k.data_ptr(),
                                  v.data_ptr(),
                                  o.data_ptr(),
                                  batch_mask.data_ptr<int>(),
                                  bs, nh, N, dim, softmax_scale, stream
                                );
    }
    else {
        flash_attention_cuda_mask(q.data_ptr(),
                                  k.data_ptr(),
                                  v.data_ptr(),
                                  o.data_ptr(),
                                  bs, nh, N, dim, softmax_scale, stream
                                );
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention", &flash_attention,
          "Flash attention v2 fp8");
}