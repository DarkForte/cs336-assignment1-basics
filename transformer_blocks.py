import torch
import einops
import math
from typing import Optional
from utils import softmax

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2.0 / (in_features + out_features))
        self.weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(
            torch.empty(out_features, in_features, device=device, dtype=dtype),
            mean = 0.0,
            std = std,
            a = -3 * std,
            b = 3 * std
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(x, self.weight, '... in_feature, out_feature in_feature -> ... out_feature')


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0
        ))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids.long()]  # token_ids should be of type LongTensor
    

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.weight
        return result.to(in_dtype)
    

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1.forward(x)  # (..., d_ff)
        silu_x = self.silu(w1_x)
        w3_x = self.w3.forward(x)  # (..., d_ff)

        elemental = einops.einsum(silu_x, w3_x, '... d_ff, ... d_ff -> ... d_ff')
        return self.w2.forward(elemental)  # (..., d_model)
    

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.r = torch.ones(max_seq_len, d_k//2, 2, 2)
        for i in range(max_seq_len):
            for k in range(d_k//2):
                now_theta = i / (theta ** (2*k / d_k))
                self.r[i, k, 0, 0] = math.cos(now_theta)
                self.r[i, k, 0, 1] = math.sin(now_theta)
                self.r[i, k, 1, 0] = -math.sin(now_theta)
                self.r[i, k, 1, 1] = math.cos(now_theta)
        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        r = self.r[token_positions] # r: (..., seq_len, d_k, 2, 2)
        unrolled_x = einops.rearrange(x, '... seq_len (d_k p)-> ... seq_len d_k p', p=2)
        rotated_x = einops.einsum(unrolled_x, r, '... seq_len d_k p1, ... seq_len d_k p1 p2 -> ... seq_len d_k p2')
        return einops.rearrange(rotated_x, '... seq_len d_k p -> ... seq_len (d_k p)', p=2)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    d_k = q.size(-1)
    scores = einops.einsum(q, k, '... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k') / math.sqrt(d_k)

    if mask is not None:
        score_mask = torch.zeros_like(mask, dtype=torch.float32)
        score_mask[mask == False] = float('-inf')
        scores += score_mask

    attn_weights = softmax(scores, dim=-1)
    output = einops.einsum(attn_weights, v, '... seq_len_q seq_len_k , ... seq_len_k d_v -> ... seq_len_q d_v')
    
    return output

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len = None, theta = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device=device)
        else:
            print("No Rotary Positional Embedding used.")
            self.rope = None

        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)
        self.device = device

    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.size(-2)
        q = self.q_proj.forward(x)  # (..., seq_len, d_model)
        k = self.k_proj.forward(x)  # (..., seq_len, d_model)
        v = self.v_proj.forward(x)  # (..., seq_len, d_model)

        q = einops.rearrange(q, '... seq_len (head d_k) -> ... head seq_len d_k', head=self.num_heads)
        k = einops.rearrange(k, '... seq_len (head d_k) -> ... head seq_len d_k', head=self.num_heads)
        v = einops.rearrange(v, '... seq_len (head d_v) -> ... head seq_len d_v', head=self.num_heads)
        if self.rope is not None and token_positions is not None:
            q = self.rope.forward(q, token_positions) 
            k = self.rope.forward(k, token_positions)

        mask = einops.rearrange(torch.triu(torch.ones(seq_len, seq_len, device=self.device)).bool(), "r c -> c r")  # Lower triangular mask
        multihead = scaled_dot_product_attention(q, k, v, mask)  # ... x seq_len x d_v
        multihead = einops.rearrange(multihead, '... head seq_len d_v -> ... seq_len (head d_v)', head=self.num_heads)

        ret = self.output_proj.forward(multihead)
        return ret
    
class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype)
        
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_model)
        normed = self.ln1.forward(x)
        attn_output = self.attn.forward(normed, token_positions=torch.arange(0, x.size(-2), dtype=torch.int32))
        x = x + attn_output
        normed = self.ln2.forward(x)
        ff_output = self.ffn.forward(normed)
        x = x + ff_output
        return x

class TransformerBlockNoNorm(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_model)
        attn_output = self.attn.forward(x, token_positions=torch.arange(0, x.size(-2), dtype=torch.int32))
        x = x + attn_output
        ff_output = self.ffn.forward(x)
        x = x + ff_output
        return x
    
class Transformer(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, n_heads: int, d_ff: int, rope_theta=None, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len=context_length, theta=rope_theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings.forward(x)  # (..., seq_len, d_model)
        for layer in self.layers:
            x = layer.forward(x)
        x = self.ln_final.forward(x)
        x = self.lm_head.forward(x)  # (..., seq_len, vocab_size)
        return x

class TransformerBlockPostNorm(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_model)
        # Post-norm: Apply normalization after residual connection
        attn_output = self.attn.forward(x, token_positions=torch.arange(0, x.size(-2), dtype=torch.int32))
        x = self.ln1.forward(x + attn_output)
        
        ff_output = self.ffn.forward(x)
        x = self.ln2.forward(x + ff_output)
        return x

class TransformerPostNorm(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, n_heads: int, d_ff: int, rope_theta=None, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([
            TransformerBlockPostNorm(d_model, n_heads, d_ff, max_seq_len=context_length, theta=rope_theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings.forward(x)  # (..., seq_len, d_model)
        for layer in self.layers:
            x = layer.forward(x)
        x = self.ln_final.forward(x)
        x = self.lm_head.forward(x)  # (..., seq_len, vocab_size)
        return x

class TransformerNoNorm(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, n_heads: int, d_ff: int, rope_theta=None, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([
            TransformerBlockNoNorm(d_model, n_heads, d_ff, max_seq_len=context_length, theta=rope_theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings.forward(x)  # (..., seq_len, d_model)
        for layer in self.layers:
            x = layer.forward(x)
        x = self.lm_head.forward(x)  # (..., seq_len, vocab_size)
        return x
    
class TransformerBlockNoPE(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        # Initialize attention without any positional embedding (no max_seq_len, no theta)
        self.attn = MultiheadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_model)
        normed = self.ln1.forward(x)
        # Don't pass token_positions to attention
        attn_output = self.attn.forward(normed)
        x = x + attn_output
        normed = self.ln2.forward(x)
        ff_output = self.ffn.forward(normed)
        x = x + ff_output
        return x

class TransformerNoPE(torch.nn.Module):
    """Transformer variant without any positional embeddings."""
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, n_heads: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([
            TransformerBlockNoPE(d_model, n_heads, d_ff, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings.forward(x)  # (..., seq_len, d_model)
        for layer in self.layers:
            x = layer.forward(x)
        x = self.ln_final.forward(x)
        x = self.lm_head.forward(x)  # (..., seq_len, vocab_size)
        return x