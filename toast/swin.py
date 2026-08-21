"""Token Channel Selection for Swin backbones.

:func:`toast.patch.apply_toast` dispatches here when it is handed a timm `SwinTransformer`.
The mechanism is the one :mod:`toast.token_channel` describes -- score the feed-forward
channels from the activations, compute only the top fraction, per forward pass -- with two
differences that come from the architecture.

**Block indexing.** Swin nests its blocks in stages, `model.layers[i].blocks[j]`. The ratio
vectors are indexed by the *global* block index, counting straight through the stages
(:mod:`toast.arch`), so Swin-T takes a twelve-element vector exactly as DeiT does, and block 0
is the first block of the first stage.

**Scoring.** There is no class token, so neither the class-token term of the ViT score nor the
attention weighting it applies to patches has a direct counterpart. A ViT weights each patch by
the class token's attention *to* it, which is one row of the map; a window map has no
distinguished row. Two substitutes are implemented:

* ``attn_weighting=False`` (default) -- the score is the token magnitude alone. This is what
  the schedules in `configs/tcs.json` were measured with, so it is the default.
* ``attn_weighting=True`` -- each token is weighted by the attention it *receives*, averaged
  over the queries of its window and over heads (:func:`window_attention_received`). This is
  the closest analogue of the ViT weighting, and needs the attention map, so it also swaps in
  :class:`SwinToastWindowAttention` to keep the map instead of letting a fused kernel discard
  it.

Note that the *other* natural substitute -- each token's average attention over its window,
i.e. averaging the map along the key axis -- is not a substitute at all: softmax normalises
along exactly that axis, so the average is ``1 / window_area`` for every token alike, and a
constant cannot reorder anything. Weighting by it selects precisely the channels magnitude
alone selects. That is why the two options above are the ones offered.

Everything else follows :mod:`toast.patch`: the masked formulation used here zeroes rejected
channels and keeps shapes static, and :mod:`toast.dense` re-packs the same selection into
physically smaller matmuls for latency.
"""

from typing import Optional, Sequence, Tuple

import torch
from timm.models.swin_transformer import (
    SwinTransformer,
    SwinTransformerBlock,
    WindowAttention,
    window_reverse,
)

from .arch import iter_blocks, num_blocks, part

__all__ = [
    "DEFAULT_SWIN_SAMPLE_RATIO",
    "SwinToastBlock",
    "SwinToastWindowAttention",
    "apply_toast_swin",
    "swin_channel_importance",
    "swin_kept_channels",
    "window_attention_received",
]

# Ten times the ViT ratio, because Swin's later stages have far fewer tokens to average over --
# 49 in the last stage of a 224px model, where 2 % would be a single token. This is the ratio
# the paper's Swin rows were measured with.
DEFAULT_SWIN_SAMPLE_RATIO = 0.2


class SwinToastWindowAttention(WindowAttention):
    """Window attention that keeps its attention map on ``last_attn``.

    Only installed when ``attn_weighting`` is on, and deliberately does not use fused/SDPA
    attention: the map is then an output, not an implementation detail we can let the kernel
    discard. The map is stored head-averaged and detached -- it is read to rank channels, and
    ranking is not differentiable, so nothing needs the graph.
    """

    last_attn: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + self._get_rel_pos_bias()
        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        self.last_attn = attn.detach().mean(dim=1)  # (num_windows*B, N, N)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        return self.proj_drop(x)

    def pop_attn(self) -> Optional[torch.Tensor]:
        """Return the stored map and clear it, so a stale one can never be reused."""
        attn, self.last_attn = self.last_attn, None
        return attn


def window_attention_received(
    attn: torch.Tensor,
    window_size: Tuple[int, int],
    shift_size: Tuple[int, int],
    H: int,
    W: int,
) -> torch.Tensor:
    """Per-token attention received, mapped back to image order.

    Args:
        attn: head-averaged window map, ``(num_windows*B, N, N)``, as
            :class:`SwinToastWindowAttention` stores it.
        window_size, shift_size: the block's, both ``(h, w)``.
        H, W: the block's token grid, *before* padding.

    Returns ``(B, H * W)``, averaging to 1 so the weights are comparable to the unweighted
    score. Averaging the map over its queries gives what each token receives; averaging over
    its keys would give ``1 / window_area`` for every token, since softmax already normalises
    that axis.

    The inverse of timm's `_attn` -- window reverse, crop the padding, undo the cyclic shift --
    is applied in that order, because a token's weight has to land back on the token it came
    from. Skipping this step is not a rounding error: window order is not image order.
    """
    weights = attn.mean(dim=-2)  # (num_windows*B, N) -- mean over queries
    weights = weights * float(attn.shape[-1])  # mean 1 rather than 1 / window_area

    wh, ww = window_size
    pad_h = (wh - H % wh) % wh
    pad_w = (ww - W % ww) % ww
    weights = weights.view(-1, wh, ww, 1)
    weights = window_reverse(weights, window_size, H + pad_h, W + pad_w)  # (B, Hp, Wp, 1)
    weights = weights[:, :H, :W, :]
    if any(shift_size):
        weights = torch.roll(weights, shifts=shift_size, dims=(1, 2))
    return weights.reshape(weights.shape[0], H * W)


def swin_channel_importance(
    tokens: torch.Tensor,
    token_weights: Optional[torch.Tensor] = None,
    sample_ratio: float = DEFAULT_SWIN_SAMPLE_RATIO,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Per-channel importance of a ``(B, N, C)`` activation, larger means more important.

    Mean ``|x|`` over the batch and over a random subsample of the tokens, optionally weighted
    per token by `token_weights` ``(B, N)`` -- see :func:`window_attention_received`.
    """
    N = tokens.shape[1]
    num_samples = max(1, int(N * sample_ratio))
    if num_samples < N:
        indices = torch.randint(0, N, (num_samples,), device=tokens.device, generator=generator)
        tokens = tokens[:, indices, :]
        if token_weights is not None:
            token_weights = token_weights[:, indices]

    if token_weights is None:
        return tokens.abs().mean(dim=(0, 1))
    return (tokens.abs() * token_weights.unsqueeze(-1).to(tokens.dtype)).mean(dim=(0, 1))


def swin_kept_channels(importance: torch.Tensor, prune_ratio: float) -> torch.Tensor:
    """Indices of the channels to keep, sorted ascending."""
    C = importance.numel()
    num_keep = min(max(1, int(C * (1.0 - prune_ratio))), C)
    return torch.topk(importance, num_keep).indices.sort()[0]


class SwinToastBlock(SwinTransformerBlock):
    """Swin block whose FFN applies Token Channel Selection.

    Set up by :func:`apply_toast_swin`. Attention is left to timm's own `_attn`, so window
    shifting, padding and the attention mask keep working across timm versions.
    """

    block_index: int = 0
    fc1_prune_ratios: Sequence[float] = ()
    fc2_prune_ratios: Sequence[float] = ()
    sample_ratio: float = DEFAULT_SWIN_SAMPLE_RATIO
    attn_weighting: bool = False
    tcs_generator: Optional[torch.Generator] = None

    def configure_tcs(
        self,
        block_index: int,
        fc1_prune_ratios: Sequence[float],
        fc2_prune_ratios: Sequence[float],
        sample_ratio: float = DEFAULT_SWIN_SAMPLE_RATIO,
        generator: Optional[torch.Generator] = None,
        attn_weighting: bool = False,
    ) -> None:
        self.block_index = block_index
        self.fc1_prune_ratios = fc1_prune_ratios
        self.fc2_prune_ratios = fc2_prune_ratios
        self.sample_ratio = sample_ratio
        self.tcs_generator = generator
        self.attn_weighting = attn_weighting

    @property
    def fc1_ratio(self) -> float:
        return 0.0 if self.block_index == 0 else float(self.fc1_prune_ratios[self.block_index])

    @property
    def fc2_ratio(self) -> float:
        return 0.0 if self.block_index == 0 else float(self.fc2_prune_ratios[self.block_index])

    @property
    def selects_channels(self) -> bool:
        return self.fc1_ratio != 0.0 or self.fc2_ratio != 0.0

    def _token_weights(self, H: int, W: int) -> Optional[torch.Tensor]:
        """Attention received per token, or ``None`` when the score does not use it."""
        if not self.attn_weighting:
            return None
        attn = self.attn.pop_attn() if hasattr(self.attn, "pop_attn") else None
        if attn is None:
            return None
        return window_attention_received(attn, self.window_size, self.shift_size, H, W)

    def _select(self, tokens, ratio: float, token_weights) -> torch.Tensor:
        importance = swin_channel_importance(
            tokens, token_weights, self.sample_ratio, self.tcs_generator
        )
        kept = swin_kept_channels(importance, ratio)
        mask = torch.zeros(tokens.shape[-1], dtype=tokens.dtype, device=tokens.device)
        mask[kept] = 1.0
        return tokens * mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x + self.drop_path1(self._attn(self.norm1(x)))
        x = x.reshape(B, -1, C)

        token_weights = self._token_weights(H, W)
        mlp = self.mlp
        fc1_ratio = self.fc1_ratio

        # The residual carries the selected activations too, so a channel dropped before fc1
        # is dropped from this block's contribution to the residual stream as well.
        residual = x if fc1_ratio == 0.0 else self._select(x, fc1_ratio, token_weights)

        h = mlp.fc1(self.norm2(residual))
        h = mlp.act(h)
        h = part(mlp, "drop1", "drop")(h)
        h = part(mlp, "norm")(h)

        fc2_ratio = self.fc2_ratio
        if fc2_ratio != 0.0:
            h = self._select(h, fc2_ratio, token_weights)

        h = mlp.fc2(h)
        h = part(mlp, "drop2", "drop")(h)

        x = residual + self.drop_path2(h.to(residual.dtype))
        return x.reshape(B, H, W, C)


def apply_toast_swin(
    model: SwinTransformer,
    fc1_prune_ratios: Optional[Sequence[float]] = None,
    fc2_prune_ratios: Optional[Sequence[float]] = None,
    sample_ratio: float = DEFAULT_SWIN_SAMPLE_RATIO,
    generator: Optional[torch.Generator] = None,
    attn_weighting: bool = False,
) -> SwinTransformer:
    """Enable Token Channel Selection on a Swin `model`, in place.

    Args:
        model: a timm SwinTransformer.
        fc1_prune_ratios: per-block fraction of `fc1` *input* channels to drop, indexed by the
            global block index. ``None`` means no pruning anywhere.
        fc2_prune_ratios: per-block fraction of `fc2` input channels (i.e. hidden units) to
            drop.
        sample_ratio, generator: forwarded to :func:`swin_channel_importance`.
        attn_weighting: weight each token by the attention it receives, instead of scoring on
            magnitude alone. Off by default -- the recorded schedules were measured without it.
            Turning it on installs :class:`SwinToastWindowAttention`, which costs the fused
            attention kernel, on the blocks that actually select channels.

    Returns the same model object.
    """
    n = num_blocks(model)
    fc1 = list(fc1_prune_ratios) if fc1_prune_ratios is not None else [0.0] * n
    fc2 = list(fc2_prune_ratios) if fc2_prune_ratios is not None else [0.0] * n

    for label, ratios in (("fc1_prune_ratios", fc1), ("fc2_prune_ratios", fc2)):
        if len(ratios) != n:
            raise ValueError(f"{label} has {len(ratios)} entries but the model has {n} blocks")
        if any(not 0.0 <= r < 1.0 for r in ratios):
            raise ValueError(f"{label} entries must lie in [0, 1); got {ratios}")

    for index, block in iter_blocks(model):
        if not isinstance(block, SwinTransformerBlock):
            raise TypeError(
                f"block {index} is a {type(block).__name__}, not a SwinTransformerBlock; "
                "ToaST's Swin path targets timm's SwinTransformer"
            )
        block.__class__ = SwinToastBlock
        block.configure_tcs(index, fc1, fc2, sample_ratio, generator, attn_weighting)
        if attn_weighting and block.selects_channels:
            block.attn.__class__ = SwinToastWindowAttention

    return model
