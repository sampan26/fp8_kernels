#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/extension.h>
#include <torch/python.h>

#include <vector>

#include "rms_norm.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")


template<bool norm_affine, typename input_t>
void  rms_norm_cuda(RMSNormsParamsBase &params, cudaStream_t stream);


void set_rms_norm_params(RMSNormsParamsBase &params,
                         const size_t batch,
                         const size_t dim,
                         
                         const at::Tensor x,
                         const at::Tensor out,
                         const at::Tensor weights,
                         bool norm_affine
                         ) {

    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
  
    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    if(norm_affine){
        params.weights_ptr = weights.data_ptr();
    } else {
        params.weights_ptr = nullptr;
    }

    params.x_batch_stride = x.stride(0);
    params.out_batch_stride = out.stride(0);
}


at::Tensor rms_norm(at::Tensor &x, c10::optional<at::Tensor>& weights_) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float8_e4m3fn);
    TORCH_CHECK(x.is_cuda());

    const auto shapes_og = x.sizes();    
    const int dim_og = x.size(-1);

    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    const int dim = x.size(1);

  
    at::Tensor out = torch::empty_like(x);
    
    at::Tensor weights;
    bool norm_affine = false;
    if(weights_.has_value()){
        weights = weights_.value();
        norm_affine = true;
        TORCH_CHECK(weights.scalar_type() == at::ScalarType::Float && weights.is_cuda(), "rms norm: weights error");
    }

    RMSNormsParamsBase params;
    set_rms_norm_params(params, batch_size, dim, x, out, weights, norm_affine);

    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
   
    if (norm_affine){
        rms_norm_cuda<true, at::Float8_e4m3fn>(params, stream);
    } else {
        rms_norm_cuda<false, at::Float8_e4m3fn>(params, stream);
    }

    return out.reshape(shapes_og);
}