#include <torch/extension.h>
#include <torch/python.h>

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cute/tensor.hpp>

using namespace cute;


void matmul_fn(int8_t *A, int8_t *B, void *C, float* A_scale, float* B_scale, int B_A, int B_B, int M, int N, int K, bool fuse_gelu) {
    int B;
    TORCH_CHECK(B_A == B_B || (B_A == 1 || B_B == 1), "Batch size mismatch");
}



torch::Tensor q8_mm(torch::Tensor a, torch::Tensor a_scale, torch::Tensor b, torch::tensor b_scale, bool fuse_gelu) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    
    int m, n, k;

    int a_ndim = a.sizes().size();
    int b_ndim = b.sizes().size();

    int bs_a;
    if (a_ndim == 3) {
        bs_a = a.size(0);
        m = a.size(1);
    }
    else {
        bs_a = 1;
        m = a.size(0);
    }

    int bs_b;
    if (b_ndim == 3) {
        bs_b = b.size(0);
        n = b.size(1);
    }
    else {
        bs_b = 1;
        n = b.size(0);
    }

    k = a.size(a_ndim - 1);

    TORCH_CHECK(bs_a == bs_b || bs_a == 1 || bs_b == 1, "Batch missmatch");

    int B;
    if (a_ndim == 1 || b_ndim == 1) {
        B = bs_a * bs_b;
    }
    else {
        B = bs_a;
    }
    auto opts = a.options();
    auto out = torch::empty({B, m, n}, opts.dtype(torch::kFloat8_e4m3fn));

    matmul_fn(
        a.data_ptr<int8_t>(), 
        b.data_ptr<int8_t>(), 
        out.data_ptr(), 
        a_scale.data_ptr<float>(), 
        b_scale.data_ptr<float>(), 
        bs_a, bs_b, m, n, k, fuse_gelu
    );
    cudaDeviceSynchronize();
    CUDA_ERROR_CHECK(cudaGetLastError());

    return out
}