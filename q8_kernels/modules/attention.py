import torch
import torch.nn as nn

import q8_kernels.functional as Q8F

from typing import *

from .rms_norm import RMSNorm
from .linear import Q8Linear

def is_16bit(x) -> bool:
    return x.dtype == torch.float16 or x.dtype == torch.bfloat16


class Attention(nn.Module):
    def __init__(self, 
                 query_dim: int, num_heads: int, head_dim: int, 
                 bias: bool=False, out_bias: bool=False,
                 out_dim: Optional[int]=None,
                 qk_rms_norm: bool = False, 
                 kv_dim: Optional[int] = None,
                 use_rope: bool = False
                ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bias = bias
        self.out_bias = out_bias
        self.inner_dim = self.num_heads * self.head_dim
        self.qk_rms_norm = qk_rms_norm
        if qk_rms_norm:
            self.q_norm = RMSNorm(self.inner_dim)
            self.k_norm = RMSNorm(self.inner_dim)
        self.use_rope = use_rope
        kv_dim = query_dim if kv_dim is None else kv_dim
        out_dim = query_dim if out_dim is None else out_dim

        self.c_q = Q8Linear(query_dim, self.inner_dim, bias=bias)
        self.c_k = Q8Linear(kv_dim, self.inner_dim, bias=bias)
        self.c_v = Q8Linear(kv_dim, self.inner_dim, bias=bias)
        
        self.to_out = Q8Linear(self.inner_dim, out_dim, out_bias)
    
    def forward(self, hidden_states, 
                freqs_cis=None, encoder_hidden_states=None, attention_mask=None, 
                non_mm_precision=torch.bfloat16, apply_qk_hadamard=True):
        B, T, C = hidden_states.size()
        if attention_mask is not None and attention_mask.ndim > 1:
            attention_mask = attention_mask.argmin(-1).squeeze().int()

        q = self.c_q(hidden_states)
        if self.qk_rms_norm:
            q = self.q_norm(q, non_mm_precision)
        
        if encoder_hidden_states:
            k = self.c_k(encoder_hidden_states)
            if self.qk_rms_norm:
                k = self.qk_rms_norm(k, non_mm_precision)
            v = self.c_v(encoder_hidden_states)
        else:
            k = self.c_k(hidden_states)
            if self.qk_rms_norm:
                k = self.qk_rms_norm(k, non_mm_precision)
            v = self.c_v(hidden_states)
            if self.use_rope:
                q = Q8F.rope.apply_rope(q, freqs_cis[0], freqs_cis[1], out_type=non_mm_precision)
                k = Q8F.rope.apply_rope(k, freqs_cis[0], freqs_cis[1], out_type=non_mm_precision)
        
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1,2)

        hidden_states = Q8F.flash_attention.flash_attn_func(q, k, v, batch_mask=attention_mask, apply_qk_hadamard=apply_qk_hadamard)
        hidden_states = hidden_states.transpose(1,2).reshape(B, T, C)

        hidden_states = hidden_states.to_out(hidden_states)
        return hidden_states.to(non_mm_precision)