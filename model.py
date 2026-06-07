import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from src.config import NeuraCoderConfig

# ------------------------------------------------------------
# 1. RMSNorm با cast به fp32 برای پایداری
# ------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor):
        # برای پایداری عددی، در fp32 محاسبه کنیم
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).to(dtype)

# ------------------------------------------------------------
# 2. RoPE پیشرفته با پشتیبانی از مقیاس‌دهی طولانی
# ------------------------------------------------------------
class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self._init_rope()
    
    def _init_rope(self):
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin(self.max_seq_len)
    
    def _set_cos_sin(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.cos_cached.shape[0]:
            self._set_cos_sin(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)

# ------------------------------------------------------------
# 3. Grouped-Query Attention با QK Norm و FlashAttention و پایداری fp32
# ------------------------------------------------------------
class GroupedQueryAttention(nn.Module):
    def __init__(self, config: NeuraCoderConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)
        
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.use_qk_norm else nn.Identity()
        
        self.dropout = config.dropout
        self.use_flash = config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
    
    def forward(self, x, cos, sin, mask=None):
        batch, seq_len, _ = x.shape
        
        # Projections
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        
        # RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # QK Norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # GQA: repeat KV heads
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # Attention با قابلیت fp32 برای پایداری
        if self.config.use_fp32_in_attention:
            q, k, v = q.float(), k.float(), v.float()
        
        if self.use_flash and mask is None:
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_out = torch.matmul(attn_weights, v)
        
        if self.config.use_fp32_in_attention:
            attn_out = attn_out.to(x.dtype)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn_out)

# ------------------------------------------------------------
# 4. SwiGLU با residual در fp32
# ------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, config: NeuraCoderConfig):
        super().__init__()
        hidden = config.hidden_size
        self.w1 = nn.Linear(hidden, hidden * 4, bias=config.bias)
        self.w2 = nn.Linear(hidden, hidden * 4, bias=config.bias)
        self.w3 = nn.Linear(hidden * 4, hidden, bias=config.bias)
        self.residual_fp32 = config.residual_in_fp32
    
    def forward(self, x):
        if self.residual_fp32:
            x_orig = x
            x = x.float()
            gate = self.w1(x)
            up = self.w2(x)
            out = self.w3(F.silu(gate) * up)
            return (out + x_orig).to(x_orig.dtype)  # residual external? نه، اینجا خود FFN است
        else:
            return self.w3(F.silu(self.w1(x)) * self.w2(x))

# ------------------------------------------------------------
# 5. MoE با z-loss و load balancing پیشرفته (Qwen3 style)
# ------------------------------------------------------------
class MoELayer(nn.Module):
    def __init__(self, config: NeuraCoderConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(config) for _ in range(config.num_experts)])
        self.router_z_loss_weight = config.moe_router_z_loss_weight
        self.aux_loss_coef = config.moe_aux_loss_coef
        
    def forward(self, x):
        batch, seq, d = x.shape
        logits = self.router(x)  # [b,s,E]
        
        # z-loss: برای جلوگیری از logits بیش از حد بزرگ (Qwen3 trick)
        z_loss = torch.logsumexp(logits, dim=-1).square().mean() * self.router_z_loss_weight
        
        # top-k selection
        top_logits, top_indices = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(top_logits, dim=-1)  # [b,s,k]
        
        # auxiliary loss: load balancing
        # fraction of tokens dispatched to each expert
        expert_mask = F.one_hot(top_indices, num_classes=self.num_experts).float()
        fraction_per_expert = expert_mask.sum(dim=(0,1,2)) / (batch * seq * self.top_k)
        # average router probability per expert
        router_prob_per_expert = logits.softmax(dim=-1).mean(dim=(0,1))
        aux_loss = (fraction_per_expert * router_prob_per_expert).sum() * self.num_experts
        aux_loss = aux_loss * self.aux_loss_coef
        
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            w = weights[..., k]  # [b,s]
            idx = top_indices[..., k]  # [b,s]
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    expert_out = self.experts[e](x[mask])
                    output[mask] += expert_out * w[mask].unsqueeze(-1)
        return output, z_loss + aux_loss

# ------------------------------------------------------------
# 6. Transformer Block با DeepNorm initialization
# ------------------------------------------------------------
class NeuraCoderBlock(nn.Module):
    def __init__(self, config: NeuraCoderConfig, layer_idx: int):
        super().__init__()
        self.attention = GroupedQueryAttention(config, layer_idx)
        if config.use_moe:
            self.ffn = MoELayer(config)
        else:
            self.ffn = SwiGLU(config)
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.use_moe = config.use_moe
        self.layer_idx = layer_idx
        self.alpha = config.deepnorm_alpha if config.init_method == "deepnorm" else 1.0
    
    def forward(self, x, cos, sin, mask=None):
        # Attention with residual
        attn_out = self.attention(self.norm1(x), cos, sin, mask)
        x = x + self.dropout(attn_out) * self.alpha
        
        # FFN with residual
        if self.use_moe:
            ffn_out, moe_loss = self.ffn(self.norm2(x))
        else:
            ffn_out = self.ffn(self.norm2(x))
            moe_loss = torch.tensor(0.0, device=x.device)
        x = x + self.dropout(ffn_out) * self.alpha
        return x, moe_loss

# ------------------------------------------------------------
# 7. مدل اصلی NeuraCoder با تمام ترفندهای پایداری
# ------------------------------------------------------------
class NeuraCoder(nn.Module):
    def __init__(self, config: NeuraCoderConfig):
        super().__init__()
        self.config = config
        
        # Stable embedding (نرمال‌سازی قبل از embedding)
        if config.use_stable_embedding:
            self.embed_scale = math.sqrt(config.hidden_size)
        else:
            self.embed_scale = 1.0
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.rope = RoPE(config.head_dim, config.max_seq_len, config.rope_theta)
        self.layers = nn.ModuleList([NeuraCoderBlock(config, i) for i in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        
        self._init_weights(config)
    
    def _init_weights(self, config):
        # Initialization با رعایت عمق (DeepNorm style)
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if config.init_method == "deepnorm":
                    # DeepNorm: variance scaling با عمق
                    depth = config.num_layers
                    scale = (2 * depth) ** 0.5
                    nn.init.normal_(p, mean=0.0, std=config.init_std / scale)
                elif config.init_method == "small_init":
                    nn.init.normal_(p, mean=0.0, std=0.01)
                else:  # normal
                    nn.init.normal_(p, mean=0.0, std=config.init_std)
            elif 'bias' in name and p is not None:
                nn.init.zeros_(p)
        
        # Special initialization for LM head (optional)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=config.init_std)
    
    def forward(self, input_ids, mask=None):
        x = self.token_embedding(input_ids)
        if self.config.use_stable_embedding:
            x = x * self.embed_scale
        
        cos, sin = self.rope(x, input_ids.size(1))
        total_moe_loss = 0.0
        for layer in self.layers:
            x, moe_loss = layer(x, cos, sin, mask)
            total_moe_loss = total_moe_loss + moe_loss
        
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, total_moe_loss
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.9):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self.forward(input_ids[:, -self.config.max_seq_len:])
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = -float('Inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits = next_logits.masked_fill(indices_to_remove, -float('Inf'))
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
    
    def save_pretrained(self, path: str):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        self.config.save_pretrained(path)
    
    @classmethod
    def from_pretrained(cls, path: str):
        config = NeuraCoderConfig.from_pretrained(path)
        model = cls(config)
        model.load_state_dict(torch.load(os.path.join(path, "model.pt")))
        return model