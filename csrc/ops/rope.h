#pragma once

struct RopeParamsBase {
    using index_t = int64_t;

    int batch, dim;
    int eps;

    index_t x_batch_stride;
    index_t out_batch_stride;
    
    index_t cos_freq_batch_stride;
    index_t sin_freq_batch_stride;

    void *__restrict__ x_ptr;
    void *__restrict__ cos_freq;
    void *__restrict__ sin_freq;

    void *__restrict__ out_ptr;
    
};
