from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class NeuraCoderConfig:
    """پیکربندی فوق‌پیشرفته NeuraCoder Pro - بهتر از Qwen3"""
    
    # معماری
    vocab_size: int = 32000
    hidden_size: int = 1024      # برای تست 150M، برای 3B به 2560 برسان
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 4        # GQA
    max_seq_len: int = 2048
    dropout: float = 0.1
    bias: bool = False
    
    # MoE (پیشرفته با Qwen3-style load balancing)
    use_moe: bool = False         # برای 150M خاموش، برای بزرگتر True
    num_experts: int = 8
    top_k_experts: int = 2
    expert_capacity: int = 4
    moe_load_balancing_weight: float = 0.01
    moe_router_z_loss_weight: float = 0.001   # Qwen3 style
    moe_aux_loss_coef: float = 0.01
    
    # Normalization
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_scaling: Optional[dict] = None
    
    # بهینه‌سازی‌های پیشرفته
    use_flash_attention: bool = True
    use_qk_norm: bool = True
    gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    
    # ترفندهای پایداری عددی
    use_stable_embedding: bool = True      # نرمال‌سازی قبل از softmax
    use_fp32_in_attention: bool = False    # برخی عملیات در fp32
    
    # Initialization (عمق‌محور)
    init_method: str = "deepnorm"          # deepnorm, small_init, normal
    init_std: float = 0.02
    deepnorm_alpha: float = 0.7            # برای مدل‌های عمیق
    residual_in_fp32: bool = True
    
    # Scheduling و clipping پویا
    learning_rate: float = 3e-4
    warmup_steps: int = 2000
    lr_schedule: str = "cosine"            # cosine, linear, constant
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    gradient_clipping: float = 1.0
    clip_threshold_dynamic: bool = True    # clipping پویا بر اساس نرم گرادیان
    
    # داده و آموزش
    code_token_weight: float = 1.2
    code_special_tokens: List[str] = field(default_factory=lambda: ["<code>", "</code>", "<python>", "<javascript>"])
    
    def __post_init__(self):
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size {self.hidden_size} not divisible by num_heads {self.num_heads}")
        self.head_dim = self.hidden_size // self.num_heads
        if self.use_moe and self.num_experts < self.top_k_experts:
            raise ValueError(f"num_experts {self.num_experts} < top_k {self.top_k_experts}")
    
    @classmethod
    def from_pretrained(cls, path: str):
        import json, os
        with open(os.path.join(path, "config.json"), "r") as f:
            data = json.load(f)
        return cls(**data)
    
    def save_pretrained(self, path: str):
        import json, os
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.__dict__, f, indent=2, default=lambda o: o if not isinstance(o, tuple) else list(o))