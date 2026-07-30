"""Invariant checks for ToaST. Self-contained: no dataset, no pretrained weights, CPU only.

    pytest tests/ -v

These cover the properties the method depends on, not accuracy numbers:

* SCWP keeps Q/K and V/O masks tied, and every head keeps exactly the same count;
* the realised sparsity matches the requested one;
* the dense re-packing agrees with the masked model within the tolerance its LayerNorm
  difference allows, and really does shrink the matrices;
* TCS scoring is reproducible under an explicit generator, and importing `toast` leaves the
  global RNG alone.
"""

import copy
import os
import sys

import pytest
import timm
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toast import (  # noqa: E402
    COUPLINGS,
    SCORES,
    StructuredCoupledPruner,
    apply_toast,
    attention_layers,
    channel_importance,
    densify,
    extract_dense_heads,
    reapply_masks,
    select_channels_dense,
    select_channels_masked,
)

MODEL = "deit_tiny_patch16_224"  # 12 blocks, 3 heads, head_dim 64 -- smallest useful case


def build(model_name=MODEL, seed=0, **toast_kwargs):
    torch.manual_seed(seed)
    model = timm.create_model(model_name, pretrained=False)
    apply_toast(model, **toast_kwargs)
    return model


# ---------------------------------------------------------------------------------- SCWP


@pytest.mark.parametrize("sparsity", [50.0, 75.0, 90.0])
@pytest.mark.parametrize("coupling", sorted(COUPLINGS))
def test_qk_and_vo_masks_are_tied(sparsity, coupling):
    model = build()
    pruner = StructuredCoupledPruner(
        model, head_sparsity=sparsity, coupling=coupling, verbose=False
    )
    for i in range(0, len(pruner.masks), 2):
        qkv_mask, proj_mask = pruner.masks[i], pruner.masks[i + 1]
        embed_dim = qkv_mask.shape[0] // 3
        q_mask, k_mask, v_mask = qkv_mask.split(embed_dim, dim=0)

        assert torch.equal(q_mask, k_mask), "Q and K must share the attention-pair mask"
        # V prunes rows, the output projection prunes the matching columns.
        assert torch.equal(v_mask[:, 0], proj_mask[0, :]), "V rows and O columns must agree"


@pytest.mark.parametrize("sparsity", [25.0, 50.0, 90.0])
def test_every_head_keeps_the_same_count(sparsity):
    """The uniform per-head budget is what makes dense re-packing possible."""
    model = build()
    pruner = StructuredCoupledPruner(model, head_sparsity=sparsity, verbose=False)
    num_heads = model.blocks[1].attn.num_heads

    for i in range(0, len(pruner.masks), 2):
        qkv_mask = pruner.masks[i]
        embed_dim = qkv_mask.shape[0] // 3
        head_dim = embed_dim // num_heads
        expected = int(head_dim * (100.0 - sparsity) / 100.0)

        for third in qkv_mask.split(embed_dim, dim=0):
            keep = ~third[:, 0]
            per_head = {int(keep[h * head_dim : (h + 1) * head_dim].sum()) for h in range(num_heads)}
            assert per_head == {expected}, f"per-head counts {per_head}, expected {expected}"


@pytest.mark.parametrize("sparsity", [50.0, 90.0])
def test_realised_sparsity_matches_request(sparsity):
    model = build()
    pruner = StructuredCoupledPruner(model, head_sparsity=sparsity, verbose=False)
    head_dim = model.blocks[1].attn.head_dim if hasattr(model.blocks[1].attn, "head_dim") else 64
    # Integer truncation means the realised value can exceed the request by <1 dimension.
    expected = 1.0 - int(head_dim * (100.0 - sparsity) / 100.0) / head_dim
    assert pruner.sparsity == pytest.approx(expected, abs=1e-9)


def test_pruned_weights_are_zero_and_stay_zero():
    model = build()
    pruner = StructuredCoupledPruner(model, head_sparsity=90.0, verbose=False)
    layers = [lin for _, qkv, proj in attention_layers(model) for lin in (qkv, proj)]

    for layer, mask in zip(layers, pruner.masks):
        assert torch.count_nonzero(layer.weight.data[mask]) == 0

    # An optimiser step pushes them off zero; reapply_masks must restore it.
    for layer in layers:
        layer.weight.data.add_(1.0)
    assert torch.count_nonzero(layers[0].weight.data[pruner.masks[0]]) > 0
    reapply_masks(model, pruner.masks)
    for layer, mask in zip(layers, pruner.masks):
        assert torch.count_nonzero(layer.weight.data[mask]) == 0


def test_first_block_is_left_dense_by_default():
    model = build()
    pruner = StructuredCoupledPruner(model, head_sparsity=90.0, verbose=False)
    assert len(pruner.masks) == 2 * (len(model.blocks) - 1)
    assert torch.count_nonzero(model.blocks[0].attn.qkv.weight) == model.blocks[0].attn.qkv.weight.numel()


def test_sparsity_leaving_no_dimensions_is_rejected():
    with pytest.raises(ValueError, match="at least one must survive"):
        StructuredCoupledPruner(build(), head_sparsity=99.9, verbose=False)


def test_per_block_sparsity_list():
    model = build()
    schedule = [0.0] + [50.0] * 5 + [90.0] * 6
    pruner = StructuredCoupledPruner(model, head_sparsity=schedule, verbose=False)
    # Mask index 0 is block 1 (block 0 skipped); blocks 1-5 at 50%, 6-11 at 90%.
    assert pruner.masks[0].float().mean().item() == pytest.approx(0.5, abs=0.02)
    assert pruner.masks[-2].float().mean().item() == pytest.approx(0.90625, abs=0.02)


def test_each_block_is_scored_from_its_own_weights():
    """Selections are per block: two blocks with different weights must not agree by default."""
    pruner = StructuredCoupledPruner(build(), head_sparsity=90.0, verbose=False)
    assert not torch.equal(pruner.masks[0], pruner.masks[2]), "blocks must be scored separately"
    assert not torch.equal(pruner.masks[1], pruner.masks[3])


@pytest.mark.parametrize("score", sorted(SCORES))
def test_all_scores_produce_valid_masks(score):
    pruner = StructuredCoupledPruner(build(), head_sparsity=80.0, score=score, verbose=False)
    assert pruner.sparsity == pytest.approx(0.8125, abs=1e-9)


def test_pruner_rejects_a_model_without_blocks():
    with pytest.raises(AttributeError, match="no `.blocks`"):
        StructuredCoupledPruner(torch.nn.Linear(4, 4), head_sparsity=50.0, verbose=False)


# ----------------------------------------------------------------------------------- TCS


def test_channel_importance_is_reproducible_with_a_generator():
    x = torch.randn(4, 197, 192)
    attn = torch.rand(4, 1, 197)

    def once(seed):
        g = torch.Generator().manual_seed(seed)
        return channel_importance(x, attn, generator=g)

    assert torch.equal(once(0), once(0))
    assert not torch.equal(once(0), once(1)), "different seeds must sample different patches"


def test_channel_importance_uses_all_patches_when_sample_ratio_is_one():
    x = torch.randn(2, 50, 96)
    a = channel_importance(x, None, sample_ratio=1.0)
    b = channel_importance(x, None, sample_ratio=1.0)
    assert torch.equal(a, b), "no sampling means no randomness"


def test_channel_importance_handles_a_lone_class_token():
    x = torch.randn(3, 1, 96)
    imp = channel_importance(x, None, cls_weight=2.0)
    assert torch.equal(imp, x[:, 0, :].abs().mean(dim=0) * 2.0)


def test_importing_toast_does_not_disturb_the_global_rng():
    torch.manual_seed(1234)
    expected = torch.randn(5)
    torch.manual_seed(1234)
    import importlib

    import toast

    importlib.reload(toast)
    assert torch.equal(torch.randn(5), expected)


@pytest.mark.parametrize("ratio", [0.25, 0.5, 0.9])
def test_masked_and_dense_selection_agree(ratio):
    x = torch.randn(2, 197, 192)
    attn = torch.rand(2, 1, 197)
    g1 = torch.Generator().manual_seed(3)
    g2 = torch.Generator().manual_seed(3)

    masked, info_m = select_channels_masked(x, ratio, attn, generator=g1)
    narrow, info_d = select_channels_dense(x, ratio, attn, generator=g2)

    assert torch.equal(info_m["kept_indices"], info_d["kept_indices"])
    assert narrow.shape[-1] == info_m["kept_channels"]
    assert torch.equal(narrow, masked[:, :, info_m["kept_indices"]])
    # Everything outside the selection is gone from the masked tensor.
    dropped = torch.ones(x.shape[-1], dtype=torch.bool)
    dropped[info_m["kept_indices"]] = False
    assert torch.count_nonzero(masked[:, :, dropped]) == 0


def test_tcs_ratio_of_zero_is_a_no_op():
    model = build(fc1_prune_ratios=[0.0] * 12, fc2_prune_ratios=[0.0] * 12)
    reference = build()
    model.eval()
    reference.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.equal(model(x), reference(x))


def test_apply_toast_rejects_bad_ratio_lists():
    with pytest.raises(ValueError, match="12 blocks"):
        apply_toast(timm.create_model(MODEL, pretrained=False), fc1_prune_ratios=[0.5] * 6)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        apply_toast(timm.create_model(MODEL, pretrained=False), fc1_prune_ratios=[1.0] * 12)


def test_toast_forward_shape_and_finiteness():
    ratios = [0.0] * 8 + [0.2, 0.3, 0.5, 0.5]
    model = build(fc1_prune_ratios=ratios, fc2_prune_ratios=[r * 1.5 for r in ratios])
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(2, 3, 224, 224))
    assert y.shape == (2, 1000)
    assert torch.isfinite(y).all()


# --------------------------------------------------------------------------------- dense


def _compressed(fc1, fc2, sparsity=90.0):
    model = build(fc1_prune_ratios=fc1, fc2_prune_ratios=fc2)
    masks = StructuredCoupledPruner(model, head_sparsity=sparsity, verbose=False).masks
    return model.eval(), masks


def test_dense_repacking_shrinks_the_weights():
    model, _ = _compressed([0.0] * 12, [0.0] * 12, sparsity=75.0)
    dense = densify(copy.deepcopy(model))
    attn = dense.blocks[6].attn

    head_dim = model.blocks[6].attn.qkv.weight.shape[1] // model.blocks[6].attn.num_heads
    assert attn.qk_head_dim == int(head_dim * 0.25)
    assert attn.qkv_weight.shape[0] == 3 * attn.num_heads * attn.qk_head_dim
    assert attn.qkv_weight.shape[0] < model.blocks[6].attn.qkv.weight.shape[0]


def test_dense_matches_masked_without_fc1_pruning():
    """With fc1 untouched both paths normalise over the same channels, so they agree tightly."""
    fc2 = [0.0] * 11 + [0.7]
    model, masks = _compressed([0.0] * 12, fc2)
    dense = densify(copy.deepcopy(model), fc2_prune_ratios=fc2).eval()

    x = torch.randn(2, 3, 224, 224)
    torch.manual_seed(5)
    with torch.no_grad():
        y_masked = model(x)
    torch.manual_seed(5)
    with torch.no_grad():
        y_dense = dense(x)
    assert torch.allclose(y_masked, y_dense, rtol=1e-4, atol=1e-4)


def test_dense_tracks_masked_when_fc1_is_pruned():
    """fc1 pruning re-normalises over kept channels only, so only the ranking survives."""
    fc1 = [0.0] * 10 + [0.3, 0.3]
    fc2 = [0.0] * 10 + [0.5, 0.7]
    model, masks = _compressed(fc1, fc2)
    dense = densify(copy.deepcopy(model), fc1_prune_ratios=fc1, fc2_prune_ratios=fc2).eval()

    x = torch.randn(4, 3, 224, 224)
    torch.manual_seed(5)
    with torch.no_grad():
        y_masked = model(x)
    torch.manual_seed(5)
    with torch.no_grad():
        y_dense = dense(x)
    assert torch.isfinite(y_dense).all()
    assert (y_masked.argmax(-1) == y_dense.argmax(-1)).float().mean() >= 0.5


def test_extract_dense_heads_rejects_uncoupled_sparsity():
    model, _ = _compressed([0.0] * 12, [0.0] * 12)
    # Break the V/O coupling by reviving one output-projection column.
    model.blocks[3].attn.proj.weight.data[:, 0] = 1.0
    with pytest.raises(ValueError, match="not coupled"):
        extract_dense_heads(model)


def test_extract_dense_heads_skips_the_first_block():
    model, _ = _compressed([0.0] * 12, [0.0] * 12)
    assert 0 not in extract_dense_heads(model)
    assert sorted(extract_dense_heads(model)) == list(range(1, len(model.blocks)))


# --------------------------------------------------------------------------------- flops


def test_baseline_flops_match_the_published_budgets():
    """The FLOPs convention must agree with the numbers the comparison tables are quoted in."""
    from toast import spec_from_model, vit_flops

    for model_name, expected in [
        ("deit_tiny_patch16_224", 1.3),
        ("deit_small_patch16_224", 4.6),
        ("deit_base_patch16_224", 17.6),
    ]:
        spec = spec_from_model(timm.create_model(model_name, pretrained=False), num_classes=1000)
        # Published budgets are quoted to one decimal, so compare at that resolution.
        assert vit_flops(spec).gflops == pytest.approx(expected, abs=0.05)


def test_scwp_scales_mhsa_and_leaves_the_ffn_alone():
    from toast import spec_from_model, vit_flops

    spec = spec_from_model(timm.create_model(MODEL, pretrained=False), num_classes=1000)
    dense = vit_flops(spec)
    pruned = vit_flops(spec, head_sparsity=75.0)

    keep = int(spec.head_dim * 0.25) / spec.head_dim
    assert pruned.mhsa[0] == dense.mhsa[0]  # block 0 stays dense
    assert pruned.mhsa[1] == pytest.approx(dense.mhsa[1] * keep)
    assert pruned.ffn == dense.ffn


def test_tcs_scales_the_ffn_and_leaves_mhsa_alone():
    from toast import spec_from_model, vit_flops

    spec = spec_from_model(timm.create_model(MODEL, pretrained=False), num_classes=1000)
    dense = vit_flops(spec)
    # fc1 and fc2 are one half of the FFN each, so 0.5 on both halves the block.
    pruned = vit_flops(spec, fc1_prune_ratios=0.5, fc2_prune_ratios=0.5)
    assert pruned.ffn[3] == pytest.approx(dense.ffn[3] * 0.5)
    assert pruned.mhsa == dense.mhsa


def test_flops_reject_a_wrong_length_ratio_list():
    from toast import spec_from_model, vit_flops

    spec = spec_from_model(timm.create_model(MODEL, pretrained=False), num_classes=1000)
    with pytest.raises(ValueError, match="12 blocks"):
        vit_flops(spec, fc1_prune_ratios=[0.5] * 7)


# -------------------------------------------------------------------------------- config


def test_every_config_entry_matches_its_recorded_flops():
    """Keeps each schedule and its recorded FLOPs consistent."""
    from toast import load_tcs_config, spec_from_model, vit_flops

    config = load_tcs_config()
    for model_name, entry in config["models"].items():
        if not entry.get("supported", True):
            continue
        spec = spec_from_model(
            timm.create_model(model_name, pretrained=False), num_classes=1000
        )
        for target, schedule in entry["configs"].items():
            computed = vit_flops(
                spec,
                head_sparsity=schedule["head_sparsity"],
                fc1_prune_ratios=schedule["fc1"],
                fc2_prune_ratios=schedule["fc2"],
            ).gflops
            assert computed == pytest.approx(schedule["computed_gflops"], abs=1e-3), (
                f"{model_name} @ {target}G: config records "
                f"{schedule['computed_gflops']}, toast.flops gives {computed:.3f}"
            )


def test_config_ratio_vectors_have_one_entry_per_block():
    from toast import load_tcs_config

    config = load_tcs_config()
    for model_name, entry in config["models"].items():
        for target, schedule in entry["configs"].items():
            for key in ("fc1", "fc2"):
                assert len(schedule[key]) == entry["num_blocks"], (
                    f"{model_name} @ {target}G: {key} has {len(schedule[key])} entries, "
                    f"expected {entry['num_blocks']}"
                )
                assert all(0.0 <= r < 1.0 for r in schedule[key])


def test_resolve_config_by_name_and_alias():
    from toast import resolve_config

    by_name = resolve_config("deit_small_patch16_224", 2.9)
    by_alias = resolve_config("deit_small", "2.9")
    assert by_name.fc1_prune_ratios == by_alias.fc1_prune_ratios
    assert by_name.head_sparsity == 90.0
    assert by_name.fc2_prune_ratios[-2:] == [0.9, 0.9]
    assert "Acc@1" in by_name.summary()


def test_resolve_config_reports_available_budgets():
    from toast import available_targets, resolve_config

    assert available_targets("deit_base_patch16_224") == ["10.27", "10.4", "10.7", "11.5"]
    with pytest.raises(KeyError, match="available budgets"):
        resolve_config("deit_base_patch16_224", 99.0)
    with pytest.raises(KeyError, match="knows about"):
        resolve_config("resnet50", 2.9)


def test_swin_configs_are_recorded_but_refuse_to_resolve():
    """Swin schedules are recorded; patching SwinTransformerBlock is not in this release."""
    from toast import load_tcs_config, resolve_config

    config = load_tcs_config()
    swin = config["models"]["swin_small_patch4_window7_224"]
    assert swin["supported"] is False
    assert sum(len(stage) for stage in swin["configs"]["5.4"]["fc1_by_stage"]) == swin["num_blocks"]
    with pytest.raises(NotImplementedError, match="not runnable"):
        resolve_config("swin_small", 5.4)


# ------------------------------------------------------------------- training/eval contract


def test_gradients_reach_pruned_weights_when_qkv_bias_is_live():
    """Why `reapply_masks` exists.

    A zeroed row of `W_Q` still has its bias, so the attention logit it feeds is
    `b_q[i] * b_k[i]` rather than identically zero, and a gradient reaches the zeroed row.
    Pretrained checkpoints have nonzero qkv biases, so this is the case that matters; timm's
    random init zeroes them, which makes the gradient vanish.
    """
    model = build()
    for block in model.blocks:
        torch.nn.init.normal_(block.attn.qkv.bias, std=0.1)
    pruner = StructuredCoupledPruner(model, head_sparsity=90.0, verbose=False)
    layers = [lin for _, qkv, proj in attention_layers(model) for lin in (qkv, proj)]

    model.train()
    model(torch.randn(2, 3, 224, 224)).sum().backward()

    reached = sum(
        int(torch.count_nonzero(layer.weight.grad[mask]))
        for layer, mask in zip(layers, pruner.masks)
    )
    assert reached > 0, "if this ever becomes 0, reapply_masks is no longer needed"


class _OneBatchLoader:
    """Smallest thing `MetricLogger.log_every` accepts: an iterable with a length."""

    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        return iter([self.batch])

    def __len__(self):
        return 1


def test_evaluate_reprojects_before_measuring():
    """The contract main.py relies on: the measured model is the sparse one."""
    import engine

    model = build()
    pruner = StructuredCoupledPruner(model, head_sparsity=90.0, verbose=False)
    layers = [lin for _, qkv, proj in attention_layers(model) for lin in (qkv, proj)]

    # Stand in for an epoch of updates.
    for layer in layers:
        layer.weight.data.add_(0.01)
    assert torch.count_nonzero(layers[0].weight.data[pruner.masks[0]]) > 0

    loader = _OneBatchLoader((torch.randn(2, 3, 224, 224), torch.randint(0, 1000, (2,))))
    engine.evaluate(loader, model, torch.device("cpu"), logger=None, masks=pruner.masks)

    for layer, mask in zip(layers, pruner.masks):
        assert torch.count_nonzero(layer.weight.data[mask]) == 0, (
            "evaluate must project onto the mask, or reported accuracy is not the sparse model's"
        )


def test_state_dict_snapshot_after_reprojection_is_sparse():
    """model_best.pth is snapshotted after evaluate, so it must be exactly sparse."""
    model = build()
    pruner = StructuredCoupledPruner(model, head_sparsity=90.0, verbose=False)
    layers = [lin for _, qkv, proj in attention_layers(model) for lin in (qkv, proj)]
    names = [n for n, m in model.named_modules() if m in set(layers)]

    for layer in layers:
        layer.weight.data.add_(0.01)
    reapply_masks(model, pruner.masks)

    state = model.state_dict()
    for name, mask in zip(names, pruner.masks):
        assert torch.count_nonzero(state[f"{name}.weight"][mask]) == 0
