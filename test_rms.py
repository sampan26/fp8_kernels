import torch
import math
import torch.utils.cpp_extension
from triton.testing import do_bench
from q8_matmul.ops._C import rms_norm

batch_size = 2
dim = 512

x = torch.randn(batch_size, dim, dtype=torch.float32)
x_fp8 = x.to(torch.float8_e4m3fn).cuda()

weights = torch.ones(dim, dtype=torch.float32)
weights_fp8 = weights.to(torch.float8_e4m3fn).cuda()

custom_out = rms_norm(x_fp8, weights_fp8).cpu()

def benchmark(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median")


print("Q8 Kernel RMSNorm:", benchmark(rms_norm, x_fp8, weights_fp8))
