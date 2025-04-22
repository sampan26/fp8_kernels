#include <torch/extension.h>
#include <torch/python.h>
#include "q8_gemm_api.cuh"
#include "static_switch.h"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cute/tensor.hpp>

using namespace cute;

template<bool Is_Even, bool fuse_gelu_activation, int BM, int BN, int BK, bool BATCH_A, bool BATCH_B, int KStages,
        typename TiledMMA, typename G2SCopyA, typename G2SCopyB,
        typename SmemLayoutA, typename SmemLayoutB, typename SmemLayoutC,

        typename SmemLayoutScaleA, typename SmemLayoutScaleB,

        typename LayoutScaleAT, typename LayoutScaleBT,
        typename LayoutScaleAV_S, typename LayoutScaleBV_S,
        typename LayoutScaleAV_D, typename LayoutScaleBV_D,
        typename LayoutScaleAV_DView, typename LayoutScaleBV_DView,
        typename G2SScaleCopyA, typename G2SScaleCopyB, 

        typename AScaleTV, typename I2TVAScale,
        typename BScaleTV, typename I2TVBScale,
        

        typename S2RCopyAtomA, typename S2RCopyAtomB,
        typename R2SCopyAtomC, typename S2GCopyAtomC, typename S2GCopyC>
__global__ void q8_gemm_kernel(const int8_t * Aptr, const int8_t * Bptr, 
                                 const float* A_scales, const float* B_scales,
                                 float_e4m3_t*  Cptr, 
                                 int M,
                                 int N, int K, 
                                 int BATCH) {
    extern __shared__ float shm_data[];

    float *Ascale_shm = shm_data;
    float *BScale_shm = shm_data + cosize(SmemLayoutScaleA{});
    int8_t *Ashm = (int8_t*)(shm_data + cosize(SmemLayoutScaleA{}) + cosize(SmemLayoutScaleA{}));
    float_e4m3_t* Cshm = (float_e4m3_t*)(shm_data + cosize(SmemLayoutScaleA{}) + cosize(SmemLayoutScaleA{}));
    int8_t *Bshm = (int8_t*)(Ashm + cosize(SmemLayoutA{}));

    int idx = threadIdx.x;
    int ix = blockIdx.x;
    int iy = blockIdx.y;
    int iz = blockIdx.z;

    int batch_a_offset = BATCH_A ? iz * M * K : 0;
    int batch_b_offset = BATCH_B ? iz * N * K : 0;
    int batch_c_offset = iz * M * N;

    int batch_ascales_offset = BATCH_A ? iz * M : 0;
    int batch_bscales_offset = BATCH_B ? iz * N : 0;

    Tensor A = cute::make_tensor(cute::make_gmem_ptr(Aptr + batch_a_offset), cute::make_shape(M, K), cute::make_stride(K, Int<1>{}));
    Tensor B = cute::make_tensor(cute::make_gmem_ptr(Bptr + batch_b_offset), cute::make_shape(N, K), cute::make_stride(K, Int<1>{}));
    Tensor D = cute::make_tensor(cute::make_gmem_ptr(Cptr + batch_c_offset), cute::make_shape(M, N), cute::make_stride(K, Int<1>{}));

    Tensor AScales = cute::make_tensor(cute::make_gmem_ptr(A_scales + batch_ascales_offset), cute::make_shape(_1{}, M), cute::make_stride(M, Int<1>{}));
    Tensor BScales = cute::make_tensor(cute::make_gmem_ptr(B_scales + batch_bscales_offset), cute::make_shape(_1{}, N), cute::make_stride(N, Int<1>{}));

    Tensor gA = cute::local_tile(A, cute::make_tile(Int<BM>{}, Int<BK>{}), cute::make_coord(iy, _));
    Tensor gB = cute::local_tile(B, cute::make_tile(Int<BN>{}, Int<BK>{}), cute::make_coord(iy, _));
    Tensor gC = cute::local_tile(C, cute::make_tile(Int<BM>{}, Int<BN>{}), cute::make_coord(iy, _));

    Tensor gAscales = cute::local_tile(Ascales, cute::make_tile(Int<1>{}, Int<BM>{}), cute::make_coord(_, iy));
    Tensor gBscales = cute::local_tile(Bscales, cute::make_tile(Int<1>{}, Int<BN>{}), cute::make_coord(_, ix));

    auto sA = cute::make_tensor(cute::make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = cute::make_tensor(cute::make_smem_ptr(Bshm), SmemLayoutB{});

    auto sAscale = cute::make_tensor(cute::make_smem_ptr(Ascale_shm), SmemLayoutA{});
    auto sBscale = cute::make_tensor(cute::make_smem_ptr(Bscale_shm), SmemLayoutB{});    

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_A(gD);
    clear(tCrD);

    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA); 

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); 
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB); 

    auto s2r_tiled_copy_a = cute::make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA = s2r_thr_copy_a.partition_S(sA);     
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); 

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB = s2r_thr_copy_b.partition_S(sB);    
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

    G2SScaleCopyA g2s_scale_copy_a;
    G2SScaleCopyB g2s_scale_copy_b;

    auto g2s_scale_thr_copy_a = g2s_scale_copy_a.get_slice(idx);
    auto g2s_scale_thr_copy_b = g2s_scale_copy_b.get_slice(idx);

    auto tAScalegAScale = g2s_scale_thr_copy_a.partition_S(gAscales);
    auto tAScalesAScale = g2s_scale_thr_copy_a.partition_D(sAscales);

    auto tBScalegBScale = g2s_scale_thr_copy_b.partition_S(gBscales);
    auto tBScalesBScale = g2s_scale_thr_copy_b.partition_D(sBscales);
    
    auto cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));
    auto tAcA = g2s_thr_copy_a.partition_S(cA);
    
    int itile_to_read = 0;
    int ismem_read = 0;
    int ismem_write = 0;
    
    int residual = M - iy*BM;

    auto tAscalepAscale = make_tensor<bool>(_1{});
    tAscalepAscale(0) = idx < residual;

}

void matmul_fn(int8_t *A, int8_t *B, void *C, float* A_scale, float* B_scale, int B_A, int B_B, int M, int N, int K, bool fuse_gelu) {
    int Bs;
    TORCH_CHECK(B_A == B_B || (B_A == 1 || B_B == 1), "Batch size mismatch");

    if (B_A == 1 || B_B == 1) {
        Bs = B_A * B_B;
    }
    else if (B_A == B_B) {
        Bs = B_A;
    }
    auto BM = Int<128>{};
    auto BN = Int<128>{};
    auto bK = Int<64>{};
    auto bP = Int<2>{};

    using SmemLayoutAtom = decltype(composition(
        Swizzle<2, 4, 3>{},
        cute::make_layout(cute::make_shape(Int<8>{}, Int<bK>{}),
                    cute::make_stride(Int<bK>{}, Int<1>{}))));

    using SmemLayoutA = decltype(
        cute::tile_to_shape(SmemLayoutAtom{}, cute::make_shape(BM, bK, bP))
    );
    
    using SmemLayoutB = decltype(
        cute::tile_to_shape(SmemLayoutAtom{}, cute::make_shape(BN, bK, bP))
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

    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_atom>;

    using G2SCopyA = decltype(
        make_tiled_copy(g2s_copy_atom{}, 
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<16>{}))));

    using G2SCopyB = G2SCopyA;

    using s2r_copy_op_a = SM75_U32x4_LDSM_N;
    using s2r_copy_traits_a = Copy_Traits<s2r_copy_op_a>;
    using s2r_copy_atom_a = Copy_Atom<s2r_copy_traits_a, int8_t>;

    using s2r_copy_op_b = SM75_U32x2_LDSM_N;
    using s2r_copy_traits_b = Copy_Traits<s2r_copy_op_b>;
    using s2r_copy_atom_b = Copy_Atom<s2r_copy_traits_b, int8_t>;

    using SmemLayoutC = decltype(
        compostiion(
            Swizzle<2, 4, 3>{},
            make_layout(make_shape(Int<MMA_WARP_M>{}, Int<MMA_WARP_N*Int<4>{}>{})),
            make_stride(Int<MMA_WARP_N*Int<4>{}>{}, Int<1>{}))
    );

    using R2SCopyAtomC = Copy_Atom<UniversalCopy<cute::uint16_t>, float_e4m3_t>;
    using S2RCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, float_e4m3_t>;

    using S2GCopyC =
        decltype(make_tiled_copy(S2GCopyAtomC{},
                                make_layout(make_shape(Int<32>{}, Int<4>{}),
                                            make_stride(Int<4>{}, Int<1>{})),
                                make_layout(make_shape(Int<1>{}, Int<16>{}))));

    int bX = (N + BN - 1) / BN;
    int bY = (M + BM - 1) / BM;
    int bZ = Bs;

    dim3 block(size(MMA{}));
    dim3 grid(bX, bY, bZ);

    static constexpr int shm_size_AB = cute::cosize(SmemLayoutA) + cute::cosize(SmemLayoutB);
    static constexpr int shm_size_C = cute::cosize(SmemLayoutC);
    static constexpr int kShmSize = cute::max(shm_size_AB, shm_size_C) * sizeof(cute::float_e4m3_t);

    int shm_size = kShmSize;

    using G2SScales_copy_op = SM80_CP_ASYNC_CACHEALWAYS<float>;
    using G2SScales_copy_traits = Copy_Traits<G2SScales_copy_op>;
    using G2SScales_copy_atom = Copy_Atom<G2SScales_copy_traits>;


    using G2SScalesCopyA = decltype(make_tiled_copy(G2SScales_copy_atom{}, make_layout(
                                                                        make_shape(Int<1>{},Int<BM>{}), make_stride(Int<BM>{}, Int<1>{})),
                                                                        make_layout(make_shape(Int<1>{},Int<1>{}), make_stride(Int<1>{}, Int<1>{}))));
    
    using G2SScalesCopyB = decltype(make_tiled_copy(G2SScales_copy_atom{}, make_layout(
                                                                        make_shape(Int<1>{},Int<BN>{}), make_stride(Int<BN>{}, Int<1>{})),
                                                                        make_layout(make_shape(Int<1>{},Int<1>{}), make_stride(Int<1>{}, Int<1>{}))));

    using SmemLayoutScaleA = Layout<Shape<Int<1>, Int<BM>>, Stride<Int<BM>, Int<1>>>;
    using SmemLayoutScaleB = Layout<Shape<Int<1>, Int<BN>>, Stride<Int<BM>, Int<1>>>;

    using AScale_TLayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2>>, Stride<Stride<_0, _8>, Stride<_64, _0>>>;
    using AScale_VLayoutS = Layout<Shape<_4, _2, _1, decltype(KStages)>, Stride<_1, _4, _128, _256>>;
    using AScale_VLayoutD = Layout<Shape<_4, _2, _1>, Stride<_1, _4, _8>>;
    using AScale_VLayoutD_View = Layout<Shape<Shape<_2, _2>, _4, _1>, Stride<Stride<_0, _1>, _2, _8>>;
    
    using AScaleTV = Layout<Shape<
                                Shape<
                                    Shape<_4, _8>, Shape<_2, _2>,
                                >, 
                                Shape<_2, _4>
                            >, 
                            Stride<
                                Stride<Stride<_0, _1>, Stride<_16, _0>>, 
                                Stride<_8, _32>
                            >>;
    using I2TVAScale = decltype(right_inverse(AScaleTV{}));

    auto BN_elems =  _2{}*BN / _16{};
    auto MMA_N = BN / _16{};
    auto CPY_N = MMA_N / _2{};

    using BScale_TLayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2>>, Stride<Stride<decltype(BN_elems), _0>, Stride<_0, _32>>>;
    using BScale_VLayoutS = Layout<Shape<_4, decltype(CPY_N), _1, decltype(KStages)>, Stride<_1, _4, _96, _96>>;
    using BScale_VLayoutD = Layout<Shape<_4, decltype(CPY_N), _1>, Stride<_1, _4, _16>>;
    using BScale_VLayoutD_view = Layout<Shape<Shape<_2, _2>, decltype(MMA_N), _1>, Stride<Stride<_1, _0>, _2, _8>>;

    using BScaleTV = Layout<Shape<
                                Shape<
                                    Shape<_4, _8>, Shape<_2, _2>,
                                >, 
                                Shape<_2, _8>
                            >, 
                            Stride<
                                Stride<Stride<_2, _0>, Stride<_0, _8>>, 
                                Stride<_1, _16>
                            >>;
    using I2TVBScale = decltype(right_inverse(BScaleTV{}));

    bool is_even = M % BM == 0;
    BOOL_SWITCH(fuse_gelu, fuse_gelu_, [&]{
        BOOL_SWITCH(is_even, Is_Even, [&]{
            BATCH_SWITCH(BA, BB, [&]{
                auto kernel = &q8_gemm_kernel<Is_Even, fuse_gelu_, BM, BN, BK, B_A, B_B, KStages, MMA,
                                                G2SCopyA, G2SCopyB, 
                                                SmemLayoutA, SmemLayoutB, SmemLayoutC, 
                                                SmemLayoutScaleA, SmemLayoutScaleA,  
                                                AScale_TLayout, BScale_TLayout,
                                                AScale_VLayoutS, BScale_VLayoutS,
                                                AScale_VLayoutD, BScale_VLayoutD, 
                                                AScale_VLayoutD_View, BScale_VLayoutD_view,
                                                G2SScalesCopyA,G2SScalesCopyB, 
                                                AScaleTV, I2TVAScale,
                                                BScaleTV, I2TVBScale,
                                                s2r_copy_atom_a, s2r_copy_atom_b, 
                                                R2SCopyAtomC, S2GCopyAtomC, S2GCopyC>;
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
                kernel<<<grid, block, shm_size>>>((int8_t*)A, (int8_t*)B, A_scale, B_scale, (float_e4m3_t*)C, M, N, K, Bs);
            });
        });
    });



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