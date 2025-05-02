import os
import torch
import math

from q8_matmul.gemm._C import q8_mm

def diff_max(a, b):
    return (a.float() - b.float()).abs().max()


def quant(x):
    x_abs_max = x.float().abs().max(-1, False).values
    x_scale = x_abs_max/127.0
    x_q8 =  (x.float() / x_scale[..., None]).round().to(torch.int8)
    return x_q8, x_scale


x = torch.randn(2, 3795, 2048, device='cuda')  # Activation tensor
w = torch.randn(8192, 2048, device='cuda')     # FFN projection weight

x_quant, x_scales = quant(x)
w_quant, w_scales = quant(w)

o_q8 = q8_mm(x_quant, w_quant, x_scales, w_scales, False)

o_q8_torch = ((x_scales[..., None] * w_scales[None, None, :]) * torch.matmul(x_quant.float(), w_quant.float().t())).to(torch.float8_e4m3fn)
o_orig = torch.matmul(x.to(torch.float8_e4m3fn).half(), w.to(torch.float8_e4m3fn).half().t())# nn.functional.linear(x, w, bias=None)

diff_q8 = diff_max(o_q8, o_orig)
diff_q8_torch = diff_max(o_q8_torch, o_orig)

o_fp8_orig = torch._scaled_mm(x[1].to(torch.float8_e4m3fn),  w.to(torch.float8_e4m3fn).contiguous().t(), scale_a=torch.tensor([1.0]).cuda(), scale_b=torch.tensor([1.0]).cuda())

print("DIFF no Hadamard: ", diff_q8)
print("DIFF no Hadamard Torch: ", diff_q8_torch)

# print("torch fp8 vs fp16: ", diff_max(o_fp8_orig, o_orig[1]))
# print("diff torch fp8 vs q8: ", diff_max(o_fp8_orig, o_q8[1]))
