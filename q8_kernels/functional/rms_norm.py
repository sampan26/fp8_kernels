from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from q8_kernels_cuda.ops._C import rms_norm

class RMSNorm8Bit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weights: Optional[torch.Tensor], out_type: Optional[torch.Tensor]) -> torch.Tensor:
        assert x.dtype == torch.float8_e4m3fn and weights.dtype == torch.float, "RMSNorm8bit: dtype mismatch"
        return rms_norm(x, weights, out_type)

def rms_norm_8bit(x: torch.Tensor, weights: Optional[torch.Tensor] = None, out_type: Optional[torch.Tensor]=None) -> torch.Tensor:
    return RMSNorm8bit.apply(x, weights, out_type)