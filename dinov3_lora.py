"""
LoRA for DINOv3's SelfAttention (qkv, proj) and SwiGLUFFN (w1, w2, w3) linears.
Verified against the actual dinov3 layer classes in attention.py / ffn_layers.py.

Usage: see the patch snippet for ssl_meta_arch.py below this file.
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear (or subclass, e.g. LinearKMaskedBias) with a
    trainable low-rank delta. Base weights are frozen; only lora_A/lora_B
    are trainable."""

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # delta = 0 at init -> identical to base

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        delta = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * delta


# DINOv3 backbone target linears, confirmed against source:
#   SelfAttention:  self.qkv (may be LinearKMaskedBias, still an nn.Linear subclass), self.proj
#   SwiGLUFFN:      self.w1, self.w2, self.w3
#   Mlp (fallback): self.fc1, self.fc2  (include in case some configs use plain MLP)
DEFAULT_TARGETS = ("qkv", "proj", "w1", "w2", "w3", "fc1", "fc2")


def inject_lora(backbone: nn.Module, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0, targets=DEFAULT_TARGETS, verbose=True):
    replaced = []
    for name, module in backbone.named_modules():
        for attr_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and attr_name in targets:
                setattr(module, attr_name,
                        LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced.append(f"{name}.{attr_name}")
    if verbose:
        print(f"[LoRA] injected into {len(replaced)} layers of {type(backbone).__name__}")
        for r_ in replaced[:6]:
            print(f"  - {r_}")
        if len(replaced) > 6:
            print(f"  ... and {len(replaced) - 6} more")
    if len(replaced) == 0:
        raise RuntimeError(
            "inject_lora found 0 target layers -- check that `targets` match "
            "your backbone's actual attribute names (they may differ if the "
            "config uses a different attention/mlp variant)."
        )
    return backbone


def sync_lora_init(student_backbone: nn.Module, teacher_backbone: nn.Module):
    """Copy student's freshly-initialized LoRA A/B into teacher so both start
    identical (matters because update_ema will EMA teacher toward student --
    starting from mismatched random A wastes early iterations)."""
    s_params = {n: p for n, p in student_backbone.named_parameters()
                if "lora_A" in n or "lora_B" in n}
    t_params = {n: p for n, p in teacher_backbone.named_parameters()
                if "lora_A" in n or "lora_B" in n}
    assert s_params.keys() == t_params.keys(), (
        "Student/teacher LoRA param names differ -- inject_lora must be "
        "called with identical r/targets on both backbones."
    )
    with torch.no_grad():
        for name, s_p in s_params.items():
            t_params[name].copy_(s_p)


def build_flat_lora_param_group(backbone_module: nn.Module, lr_multiplier=1.0,
                                 wd_multiplier=1.0):
    """Bypasses get_params_groups_with_decay_fsdp (which parses param names
    for layerwise decay and will choke on/misparse 'lora_A'/'lora_B'/'base.*'
    names). Returns a single flat param group of only the trainable LoRA
    params, matching the dict shape ssl_meta_arch.py expects."""
    trainable = [p for p in backbone_module.parameters() if p.requires_grad]
    return [{
        "params": trainable,
        "lr_multiplier": lr_multiplier,
        "wd_multiplier": wd_multiplier,
        "is_last_layer": False,
    }]
