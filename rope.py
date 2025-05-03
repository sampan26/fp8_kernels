import torch
import math
from typing import *

from einops import rearrange

from q8_matmul.ops._C import rope

frame_rate = 25

video_scale_factor = 8
vae_scale_factor = 32

height = 480
width = 720
num_frames = 81 
num_frames = ((num_frames - 2) // 8 + 1) * 8 + 1

height = ((height - 1) // 32 + 1) * 32
width = ((width - 1) // 32 + 1) * 32

latent_frame_rate = frame_rate / video_scale_factor


latent_frame_rates = (
                    torch.ones(
                        2, 1, device="cuda:0"
                    )
                    * latent_frame_rate
                )

latent_height = height // vae_scale_factor
latent_width = width // vae_scale_factor
latent_num_frames = num_frames // video_scale_factor + 1
num_latent_patches = latent_height * latent_width * latent_num_frames

def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    elif dims_to_append == 0:
        return x
    return x[(...,) + (None,) * dims_to_append]


def apply_rotary_emb(
    input_tensor: torch.Tensor,
    freqs_cis: Tuple[torch.FloatTensor, torch.FloatTensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos_freqs = freqs_cis[0]
    sin_freqs = freqs_cis[1]

    t_dup = rearrange(input_tensor, "... (d r) -> ... d r", r=2)
    t1, t2 = t_dup.unbind(dim=-1)
    t_dup = torch.stack((-t2, t1), dim=-1)
    input_tensor_rot = rearrange(t_dup, "... d r -> ... (d r)")
    
    out = input_tensor * cos_freqs + input_tensor_rot * sin_freqs

    return out
def get_grid(
        orig_num_frames, orig_height, orig_width, batch_size, scale_grid, device
    ):
        _patch_size = [1, 1, 1]
        f = orig_num_frames // _patch_size[0]
        h = orig_height // _patch_size[1]
        w = orig_width // _patch_size[2]
        grid_h = torch.arange(h, dtype=torch.float32, device=device)
        grid_w = torch.arange(w, dtype=torch.float32, device=device)
        grid_f = torch.arange(f, dtype=torch.float32, device=device)
        grid = torch.meshgrid(grid_f, grid_h, grid_w)
        grid = torch.stack(grid, dim=0)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

        if scale_grid is not None:
            for i in range(3):
                if isinstance(scale_grid[i], torch.Tensor):
                    scale = append_dims(scale_grid[i], grid.ndim - 1)
                else:
                    scale = scale_grid[i]
                grid[:, i, ...] = grid[:, i, ...] * scale * _patch_size[i]

        grid = rearrange(grid, "b c f h w -> b c (f h w)", b=batch_size)
        return grid

scale_grid = ((
    1 / latent_frame_rates,
    vae_scale_factor,
    vae_scale_factor,
))

indices_grid = get_grid(
                    orig_num_frames=latent_num_frames,
                    orig_height=latent_height,
                    orig_width=latent_width,
                    batch_size=2,
                    scale_grid=scale_grid,
                    device="cuda",
)

def get_fractional_positions(indices_grid):
    fractional_positions = torch.stack(
        [
            indices_grid[:, i] / [20, 2048, 2048][i]
            for i in range(3)
        ],
        dim=-1,
    )
    return fractional_positions

def precompute_freqs_cis( indices_grid, spacing="exp"):
    dtype = torch.float32  # We need full precision in the freqs_cis computation.
    dim = 2048
    theta = 10000.0

    fractional_positions = get_fractional_positions(indices_grid)

    start = 1
    end = theta
    device = fractional_positions.device
    if spacing == "exp":
        indices = theta ** (
            torch.linspace(
                math.log(start, theta),
                math.log(end, theta),
                dim // 6,
                device=device,
                dtype=dtype,
            )
        )
        indices = indices.to(dtype=dtype)
    elif spacing == "exp_2":
        indices = 1.0 / theta ** (torch.arange(0, dim, 6, device=device) / dim)
        indices = indices.to(dtype=dtype)
    elif spacing == "linear":
        indices = torch.linspace(start, end, dim // 6, device=device, dtype=dtype)
    elif spacing == "sqrt":
        indices = torch.linspace(
            start**2, end**2, dim // 6, device=device, dtype=dtype
        ).sqrt()

    indices = indices * math.pi / 2

    if spacing == "exp_2":
        freqs = (
            (indices * fractional_positions.unsqueeze(-1))
            .transpose(-1, -2)
            .flatten(2)
        )
    else:
        freqs = (
            (indices * (fractional_positions.unsqueeze(-1) * 2 - 1))
            .transpose(-1, -2)
            .flatten(2)
        )

    cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
    sin_freq = freqs.sin().repeat_interleave(2, dim=-1)
    if dim % 6 != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, : dim % 6])
        sin_padding = torch.zeros_like(cos_freq[:, :, : dim % 6])
        cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
        sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)
    return cos_freq.to(dtype), sin_freq.to(dtype)



cos_freq, sin_freq = precompute_freqs_cis(indices_grid)
x = torch.randn(2, 3795, 2048, device='cuda', dtype=torch.bfloat16) 
x_fp8 = x.to(torch.float8_e4m3fn)
s = torch.cuda.Event(True)
e = torch.cuda.Event(True)
s.record()
o_rope = rope(x_fp8, cos_freq, sin_freq)
e.record()
torch.cuda.synchronize()
print(s.elapsed_time(e))


cos_freq_fp16 = cos_freq.to(torch.bfloat16)
sin_freq_fp16 = sin_freq.to(torch.bfloat16)
freqs = [cos_freq_fp16, sin_freq_fp16]
torch.cuda.synchronize()
s = torch.cuda.Event(True)
e = torch.cuda.Event(True)
s.record()
o_torch_rope = apply_rotary_emb(x, freqs)
e.record()
torch.cuda.synchronize()
print(s.elapsed_time(e))