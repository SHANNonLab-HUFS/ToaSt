"""Re-pack a pruned model into dense, physically smaller matmuls.

SCWP and TCS both produce *masked* models: the tensors keep their original shapes and the
pruned entries are zero. That is the right form for training and for accuracy numbers, but a
zero still costs a multiply, so it says nothing about speed.

Because SCWP keeps the same number of dimensions in every head, the surviving weights of a
block re-pack into contiguous smaller matrices -- no gather, no sparse kernel, no custom CUDA.
This module performs that re-packing so latency can be measured on the compressed shapes:

    masked model  ->  extract_dense_heads()  ->  DenseToastBlock / DenseCoupledAttention

**Accuracy is not bit-identical to the masked model.** The FFN's `norm2` is the reason: the
masked path normalises over all `C` channels (the rejected ones contributing zeros), the dense
path normalises over the kept channels only, and the two have different mean and variance.
Report accuracy from the masked path (`toast.patch`) and latency from this one; the tests
check that the two agree to a loose tolerance rather than exactly.
"""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .token_channel import DEFAULT_CLS_WEIGHT, DEFAULT_SAMPLE_RATIO, channel_importance

__all__ = ["extract_dense_heads", "DenseCoupledAttention", "DenseToastBlock", "densify"]


def extract_dense_heads(model: nn.Module, skip_first_block: bool = True) -> Dict[int, Dict]:
    """Collect each block's surviving attention weights, grouped per head.

    Reads the structure off the *weights* rather than off a mask list, so it works on a
    checkpoint whose masks were not saved. Requires the per-head uniformity that SCWP
    guarantees.

    Returns ``{block_idx: {...}}`` with the dense per-head weight blocks and the index sets
    needed to slice biases.
    """
    dense: Dict[int, Dict] = {}

    for block_idx, block in enumerate(model.blocks):
        if skip_first_block and block_idx == 0:
            continue

        qkv_weight = block.attn.qkv.weight
        o_weight = block.attn.proj.weight
        num_heads = block.attn.num_heads

        embed_dim = qkv_weight.shape[1]
        head_dim = embed_dim // num_heads
        q_weight, k_weight, v_weight = qkv_weight.split(embed_dim, dim=0)

        qk_active = (q_weight != 0).any(dim=1)
        v_active = (v_weight != 0).any(dim=1)
        o_active = (o_weight != 0).any(dim=0)

        # SCWP ties V's rows to O's columns; if this fails the model was not pruned by SCWP.
        if not torch.equal(v_active, o_active):
            raise ValueError(
                f"block {block_idx}: V rows and output-projection columns disagree, so the "
                "value path is not coupled -- dense re-packing needs SCWP-shaped sparsity"
            )

        q_heads, k_heads, v_heads, o_heads = [], [], [], []
        qk_dims, v_dims = [], []

        for h in range(num_heads):
            lo, hi = h * head_dim, (h + 1) * head_dim
            qk_mask = qk_active[lo:hi]
            v_mask = v_active[lo:hi]

            if qk_mask.any():
                q_heads.append(q_weight[lo:hi][qk_mask].contiguous())
                k_heads.append(k_weight[lo:hi][qk_mask].contiguous())
            else:
                q_heads.append(None)
                k_heads.append(None)
            qk_dims.append(int(qk_mask.sum()))

            if v_mask.any():
                v_heads.append(v_weight[lo:hi][v_mask].contiguous())
                o_heads.append(o_weight[:, lo:hi][:, v_mask].contiguous())
            else:
                v_heads.append(None)
                o_heads.append(None)
            v_dims.append(int(v_mask.sum()))

        dense[block_idx] = {
            "q_heads": q_heads,
            "k_heads": k_heads,
            "v_heads": v_heads,
            "o_heads": o_heads,
            "qk_head_dims": qk_dims,
            "v_head_dims": v_dims,
            "qk_indices": torch.where(qk_active)[0],
            "v_indices": torch.where(v_active)[0],
            "num_heads": num_heads,
            "head_dim": head_dim,
            "embed_dim": embed_dim,
        }

    return dense


class DenseCoupledAttention(nn.Module):
    """Attention on re-packed weights, with a smaller per-head dimension.

    Requires every head to have kept the same number of dimensions -- otherwise the heads
    cannot share one reshape, which is the whole point of SCWP's uniform budget.
    """

    def __init__(self, attn: nn.Module, info: Dict):
        super().__init__()
        self.num_heads = info["num_heads"]
        self.embed_dim = info["embed_dim"]
        self.scale = attn.scale
        self.attn_drop = attn.attn_drop
        self.proj_drop = attn.proj_drop

        qk_dims = {d for d in info["qk_head_dims"] if d > 0}
        v_dims = {d for d in info["v_head_dims"] if d > 0}
        if len(qk_dims) != 1 or len(v_dims) != 1:
            raise ValueError(
                "dense attention needs a uniform per-head budget; got "
                f"qk={sorted(info['qk_head_dims'])} v={sorted(info['v_head_dims'])}"
            )
        self.qk_head_dim = qk_dims.pop()
        self.vo_head_dim = v_dims.pop()
        self.total_qk = self.num_heads * self.qk_head_dim
        self.total_vo = self.num_heads * self.vo_head_dim

        q = torch.cat([h for h in info["q_heads"] if h is not None], dim=0)
        k = torch.cat([h for h in info["k_heads"] if h is not None], dim=0)
        v = torch.cat([h for h in info["v_heads"] if h is not None], dim=0)
        self.register_buffer("qkv_weight", torch.cat([q, k, v], dim=0))
        self.register_buffer("o_weight", torch.cat([h for h in info["o_heads"] if h is not None], dim=1))

        if attn.qkv.bias is not None:
            qb, kb, vb = attn.qkv.bias.split(self.embed_dim)
            qk_idx, v_idx = info["qk_indices"], info["v_indices"]
            self.register_buffer(
                "qkv_bias", torch.cat([qb[qk_idx], kb[qk_idx], vb[v_idx]], dim=0)
            )
        else:
            self.qkv_bias = None

        self.proj_bias = attn.proj.bias

    def forward(self, x: torch.Tensor):
        B, N, _ = x.shape
        qkv = F.linear(x, self.qkv_weight, self.qkv_bias)

        q = qkv[:, :, : self.total_qk].reshape(B, N, self.num_heads, self.qk_head_dim).transpose(1, 2)
        k = (
            qkv[:, :, self.total_qk : 2 * self.total_qk]
            .reshape(B, N, self.num_heads, self.qk_head_dim)
            .transpose(1, 2)
        )
        v = qkv[:, :, 2 * self.total_qk :].reshape(B, N, self.num_heads, self.vo_head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.total_vo)
        x = F.linear(x, self.o_weight, self.proj_bias)
        return self.proj_drop(x), attn


class DenseToastBlock(nn.Module):
    """Block combining dense attention with dense (narrowing) Token Channel Selection.

    Mirrors :class:`toast.patch.ToastBlock`, except each selection slices the weight matrix
    instead of zeroing an input, so the matmul actually shrinks.
    """

    def __init__(
        self,
        block: nn.Module,
        attn_info: Dict,
        block_index: int,
        fc1_prune_ratios: Sequence[float],
        fc2_prune_ratios: Sequence[float],
        cls_weight: float = DEFAULT_CLS_WEIGHT,
        sample_ratio: float = DEFAULT_SAMPLE_RATIO,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.attn = DenseCoupledAttention(block.attn, attn_info)

        self.norm1 = block.norm1
        self.norm2 = block.norm2
        self.fc1 = block.mlp.fc1
        self.fc2 = block.mlp.fc2
        self.act = block.mlp.act
        self.drop1 = getattr(block.mlp, "drop1", None) or getattr(block.mlp, "drop", nn.Identity())
        self.drop2 = getattr(block.mlp, "drop2", None) or getattr(block.mlp, "drop", nn.Identity())
        self.drop_path = (
            getattr(block, "drop_path1", None) or getattr(block, "drop_path", None) or nn.Identity()
        )

        self.block_index = block_index
        self.fc1_ratio = float(fc1_prune_ratios[block_index])
        self.fc2_ratio = float(fc2_prune_ratios[block_index])
        self.cls_weight = cls_weight
        self.sample_ratio = sample_ratio
        self.generator = generator

    def _kept(self, tokens: torch.Tensor, ratio: float, cls_attn) -> torch.Tensor:
        importance = channel_importance(
            tokens, cls_attn, self.cls_weight, self.sample_ratio, self.generator
        )
        C = tokens.shape[-1]
        num_keep = min(max(1, int(C * (1.0 - ratio))), C)
        return torch.topk(importance, num_keep).indices.sort()[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_attn, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(x_attn)

        needs_cls_attn = self.fc1_ratio > 0.0 or self.fc2_ratio > 0.0
        cls_attn = (
            attn.mean(dim=1)[:, 0:1, :]
            if needs_cls_attn and attn is not None and attn.dim() == 4
            else None
        )

        residual = x
        kept: Optional[torch.Tensor] = None

        if self.fc1_ratio == 0.0:
            h = self.fc1(self.norm2(x))
        else:
            kept = self._kept(x, self.fc1_ratio, cls_attn)
            # LayerNorm over the kept channels only -- see the module docstring.
            h = F.layer_norm(
                x[:, :, kept],
                (kept.numel(),),
                weight=self.norm2.weight.index_select(0, kept),
                bias=self.norm2.bias.index_select(0, kept),
                eps=self.norm2.eps,
            )
            h = F.linear(h, self.fc1.weight.index_select(1, kept), self.fc1.bias)

        h = self.drop1(self.act(h))

        if self.fc2_ratio == 0.0:
            h = self.fc2(h)
        else:
            kept_fc2 = self._kept(h, self.fc2_ratio, cls_attn)
            h = F.linear(h[:, :, kept_fc2], self.fc2.weight.index_select(1, kept_fc2), self.fc2.bias)

        h = self.drop2(h)

        if kept is None:
            return residual + self.drop_path(h)

        # Match ToastBlock: channels dropped before fc1 are dropped from the residual too.
        gated = torch.zeros_like(residual)
        gated.index_copy_(2, kept, residual.index_select(2, kept))
        return gated + self.drop_path(h)


def densify(
    model: nn.Module,
    fc1_prune_ratios: Optional[Sequence[float]] = None,
    fc2_prune_ratios: Optional[Sequence[float]] = None,
    skip_first_block: bool = True,
    **tcs_kwargs,
) -> nn.Module:
    """Return `model` with its pruned blocks replaced by dense equivalents, in place.

    Deep-copy first if you still need the masked model:
    ``densify(copy.deepcopy(model), ...)``.
    """
    n = len(model.blocks)
    fc1 = list(fc1_prune_ratios) if fc1_prune_ratios is not None else [0.0] * n
    fc2 = list(fc2_prune_ratios) if fc2_prune_ratios is not None else [0.0] * n

    dense_info = extract_dense_heads(model, skip_first_block)
    for block_idx, info in dense_info.items():
        model.blocks[block_idx] = DenseToastBlock(
            model.blocks[block_idx], info, block_idx, fc1, fc2, **tcs_kwargs
        )
    return model
