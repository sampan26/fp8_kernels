import os
import torch
import math
from fast_hadamard_transform import hadamard_transform

from q8_matmul.gemm._C import q8_mm

def hadamard_quant(x):
    k = x.shape[-1]
    x_hadamard = hadamard_transform(x, scale=1/math.sqrt(k))
    x_abs_max_hadamard = x_hadamard.float().abs().max(-1, False).values
    x_scale_hadamard = x_abs_max_hadamard/127.0
    x_q8_hadamard = (x_hadamard.float() / x_scale_hadamard[..., None]).round().to(torch.int8)
    return x_q8_hadamard, x_scale_hadamard

def quant(x):
    x_abs_max = x.float().abs().max(-1, False).values
    x_scale = x_abs_max/127.0
    x_q8 =  (x.float() / x_scale[..., None]).round().to(torch.int8)
    return x_q8, x_scale


# x = torch.rand(2, 3795, 2048).cuda()
# w = torch.rand(2048, 1024).cuda()
x = torch.rand(1, 24, 16).cuda()
w = torch.rand(16, 8).cuda()


k = x.shape[-1]
x_hadamard = hadamard_transform(x, scale=1/math.sqrt(k))
w_hadamard = hadamard_transform(w, scale=1/math.sqrt(k))

x_quant_h, x_scales_h = hadamard_quant(x)
w_quant_h, w_scales_h = hadamard_quant(w)

x_quant, x_scales = quant(x)
w_quant, w_scales = quant(w)

o_q8 = q8_mm(x_quant, x_scales, w_quant, w_scales, False)

o_q8_torch = ((x_scales[..., None] * w_scales[None, None, :]) * torch.matmul(x_quant.float(), w_quant.float().t())).to(torch.float8_e4m3fn)
o_orig = torch.matmul(x.to(torch.float8_e4m3fn).half(), w.to(torch.float8_e4m3fn).half().t())# nn.functional.linear(x, w, bias=None)

diff_q8 = diff_max(o_q8, o_orig)
diff_q8_torch = diff_max(o_q8_torch, o_orig)