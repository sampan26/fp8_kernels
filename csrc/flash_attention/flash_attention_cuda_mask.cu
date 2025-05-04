


void flash_attention_cuda_mask(void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, int* BatchMask,
                                size_t bs, size_t nh, size_t N, size_t dim, float softmax_scale, cudaStream_t stream) {
    const int BLOCK_SIZE = 128;
    const int Br = 64, Bc = 64;

    dim3 grid(bs * nh, (N + Br - 1) / Br);
    if (N % Br == 0 && N % Bc == 0) {
        auto kernel = &flash
    }
}

void flash_attention_cuda_mask(void* Q_ptr, void* K_ptr, void* V_ptr, void* O_ptr, int* BatchMask,
                                size_t bs, size_t nh, size_t N, size_t dim, float softmax_scale, cudaStream_t stream);
