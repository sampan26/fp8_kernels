import os
import torch
import math

from q8_matmul.gemm._C import q8_mm

def hadamard_torch(n: int,
                   *,
                   dtype: torch.dtype = torch.int8,
                   device: torch.device | str | None = None) -> torch.Tensor:
    """
    Construct an n‑by‑n Hadamard matrix with Sylvester’s recursion.
    """
    if n <= 0 or (n & (n - 1)) != 0:        
        raise ValueError("`n` must be a positive power of two")

    H = torch.ones((1, 1), dtype=dtype, device=device)

    # Sylvester’s construction
    stages = int(math.log2(n))
    for _ in range(stages):
        H = torch.cat((
                torch.cat((H,  H), dim=1),
                torch.cat((H, -1 * H), dim=1)
             ), dim=0)

    return H


def hadamard_transform(x: torch.Tensor, *, scale=1.0):
    dim   = x.size(-1)
    H_f32 = hadamard_torch(dim, dtype=torch.float32, device=x.device)
    y     = torch.nn.functional.linear(x, H_f32) * scale
    return y

def hadamard_quant(x):
    k = x.shape[-1]
    x_hadamard = hadamard_transform(x, scale=1/math.sqrt(k))
    x_abs_max_hadamard = x_hadamard.abs().max(-1, False).values
    x_scale_hadamard = x_abs_max_hadamard/127.0
    x_q8_hadamard = (x_hadamard / x_scale_hadamard[..., None]).round().to(torch.int8)
    return x_q8_hadamard, x_scale_hadamard

def quant(x):
    x_abs_max = x.abs().max(-1, False).values
    x_scale = x_abs_max/127.0
    x_q8 =  (x / x_scale[..., None]).round().to(torch.int8)
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

o_q8_h = q8_mm(x_quant_h[0].contiguous(), w_quant_h, x_scales_h, w_scales_h, False)
o_q8_h = q8_mm(x_quant_h, w_quant_h, x_scales_h, w_scales_h, False)
o_q8 = q8_mm(x_quant, w_quant, x_scales, w_scales, False)