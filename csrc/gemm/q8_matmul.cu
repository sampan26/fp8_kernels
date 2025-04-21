#include <torch/extension.h>
#include <torch/python.h>
#include "q8_gemm_api.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cute/tensor.hpp>

using namespace cute;

void matmul_fn(int8_t *A, int8_t *B, void *C, float* A_scale, float* B_scale, int B_A, int B_B, int M, int N, int K, bool fuse_gelu) {
    int Bs;
    TORCH_CHECK(B_A == B_B || (B_A == 1 || B_B == 1), "Batch size mismatch");

    if (B_A == 1 || B_B == 1) {
        Bs = B_A * B_B;
    }
    else if (B_A == B_B) {
        Bs = B_A;
    }
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<64>{};
    auto bP = Int<2>{};

    using SmemLayoutAtom = decltype(composition(
        Swizzle<2, 4, 3>{},
        cute::make_layout(cute::make_shape(Int<8>{}, Int<bK>{}),
                    cute::make_stride(Int<bK>{}, Int<1>{}))));

    using SmemLayoutA = decltype(
        cute::tile_to_shape(SmemLayoutAtom{}, cute::make_shape(bM, bK, bP))
    );
    
    using SmemLayoutB = decltype(
        cute::tile_to_shape(SmemLayoutAtom{}, cute::make_shape(bN, bK, bP))
    );

    using mma_op = SM80_16x8x32_S32S8S8S32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    static constexpr int WARP_ROWS = 2;
    static constexpr int WARP_COLS = 2;
    
    using mma_atom_shape = mma_traits::Shape_MNK;

    static constexpr int MMA_WARP_M = WARP_ROWS * get<0>(mma_atom_shape{});
    static constexpr int MMA_WARP_N = 1 * WARP_COLS * get<1>(mma_atom_shape{});
    static constexpr int MMA_WARP_K = 1 * get<2>(mma_atom_shape{});

    using MMA_WARP_Tile = decltype(
        make_layout(make_shape(Int<WARP_ROWS>{}, Int<WARP_COLS>{}, Int<1>{}))
    );
    using MMA_PARTITION_TILE = Tile<Int<MMA_WARP_M>, Int<MMA_WARP_N>, Int<MMA_WARP_K>>;
    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_WARP_Tile{}, MMA_PARTITION_TILE{}));

}



torch::Tensor q8_mm(torch::Tensor a, torch::Tensor a_scale, torch::Tensor b, torch::Tensor b_scale, bool fuse_gelu) {
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

    return out;
}