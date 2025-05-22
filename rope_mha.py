import torch
import torch.nn as nn
import math

def rotate_half(x):
    # Split last dim into even and odd, then apply rotation
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, sin, cos):
    # RoPE: x * cos + rotate(x) * sin
    return (x * cos) + (rotate_half(x) * sin)

class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim % 2 == 0, "Head dimension must be even for RoPE"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _get_rotary_emb(self, seq_len, device):
        dim = self.head_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        pos = torch.arange(seq_len, dtype=torch.float, device=device)
        freqs = torch.einsum("i,j->ij", pos, inv_freq)  # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        sin = emb.sin()[None, :, None, :]
        cos = emb.cos()[None, :, None, :]
        return sin, cos

    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)  # (B, T, 3 * C)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, D)

        sin, cos = self._get_rotary_emb(T, x.device)
        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_probs = torch.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_probs, v)  # (B, H, T, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        return self.out_proj(attn_output)
