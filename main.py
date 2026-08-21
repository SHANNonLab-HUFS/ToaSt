"""Train and evaluate ToaST-compressed vision transformers.

Handles both the ImageNet runs and the CIFAR-100 transfer runs; the difference is entirely in
the flags, see `scripts/`.

    # evaluate a compressed DeiT-S off the shelf
    python main.py --eval --model deit_small_patch16_224 --data-path $IMAGENET \
        --weight-pruning --head-sparsity 90 --fc2-prune-ratio 0 0 0 0 0 0 0 0 0 0 0 0.7

    # fine-tune after compression, 4 GPUs
    torchrun --nproc_per_node=4 main.py --model deit_small_patch16_224 --data-path $IMAGENET \
        --weight-pruning --head-sparsity 90 --epochs 300 --output_dir ./log/deit_s_90

    # Swin takes the same flags; its ratio vectors are indexed by global block
    python main.py --eval --model swin_small_patch4_window7_224 --target-flops 5.4 \
        --data-path $IMAGENET
"""

import argparse
import datetime
import inspect
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler

import utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from samplers import RASampler
from toast import (
    COUPLINGS,
    DEFAULT_CONFIG_PATH,
    SCORES,
    StructuredCoupledPruner,
    apply_toast,
    num_blocks as count_blocks,
    resolve_config,
    spec_for_model,
    toast_flops,
)
from utils import MultiEpochsDataLoader

warnings.filterwarnings("ignore")

NUM_BLOCKS_HINT = 12  # only used to size the default per-block ratio lists


def get_args_parser():
    parser = argparse.ArgumentParser("ToaST training and evaluation", add_help=False)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--eval", action="store_true", help="evaluate only, no training")

    # Model
    parser.add_argument("--model", default="deit_small_patch16_224", type=str, metavar="MODEL")
    parser.add_argument("--input-size", default=224, type=int)
    parser.add_argument("--drop", type=float, default=0.0, metavar="PCT", help="dropout rate")
    parser.add_argument("--drop-path", type=float, default=0.1, metavar="PCT", help="stochastic depth")
    parser.add_argument(
        "--pretrained",
        default="",
        type=str,
        help="path to a checkpoint to start from; omit to download timm's pretrained weights",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_false",
        dest="pretrained_timm",
        help="start from random init instead of timm's pretrained weights",
    )
    parser.set_defaults(pretrained_timm=True)

    # ------------------------------------------------------------------ ToaST: config
    cfg = parser.add_argument_group("Compression schedule")
    cfg.add_argument(
        "--target-flops", default=None,
        help="load this model's schedule for the given GFLOPs budget from --tcs-config, "
        "instead of spelling out --head-sparsity and the ratio vectors. Explicit flags "
        "still win over the config.",
    )
    cfg.add_argument(
        "--tcs-config", default=DEFAULT_CONFIG_PATH,
        help="path to the schedule config (default: configs/tcs.json)",
    )
    cfg.add_argument(
        "--print-flops", action="store_true", default=True,
        help="log the analytic FLOPs breakdown at startup (default)",
    )
    cfg.add_argument("--no-print-flops", action="store_false", dest="print_flops")

    # ------------------------------------------------------------------ ToaST: SCWP
    scwp = parser.add_argument_group("Structured Coupled Weight Pruning (attention)")
    scwp.add_argument("--weight-pruning", action="store_true", help="enable SCWP")
    scwp.add_argument(
        "--head-sparsity",
        type=float,
        nargs="+",
        default=[90.0],
        metavar="PCT",
        help="percentage of each head's dimensions to drop; one value for all blocks, "
        "or one per block (index 0 = block 0, which is left dense)",
    )
    scwp.add_argument(
        "--importance", type=str, default="gm", choices=sorted(SCORES),
        help="row score: gm = geometric-median distance (paper), l1/l2 = magnitude baselines",
    )
    scwp.add_argument(
        "--coupling", type=str, default="coupled", choices=sorted(COUPLINGS),
        help="how Q/K and V/O are scored together; anything but 'coupled' is an ablation",
    )
    scwp.add_argument(
        "--prune-first-block", action="store_false", dest="skip_first_block",
        help="also prune block 0 (left dense by default)",
    )
    scwp.set_defaults(skip_first_block=True)
    scwp.add_argument(
        "--mask-every-step", action="store_true",
        help="project pruned weights back onto their masks after every optimiser step, "
        "in addition to before each evaluation",
    )

    # ------------------------------------------------------------------ ToaST: TCS
    tcs = parser.add_argument_group("Token Channel Selection (feed-forward)")
    tcs.add_argument(
        "--fc1-prune-ratio", type=float, nargs="+", default=None, metavar="R",
        help=f"per-block fraction of fc1 input channels to drop ({NUM_BLOCKS_HINT} values)",
    )
    tcs.add_argument(
        "--fc2-prune-ratio", type=float, nargs="+", default=None, metavar="R",
        help="per-block fraction of fc2 input channels (hidden units) to drop",
    )
    tcs.add_argument(
        "--cls-weight", type=float, default=2.0,
        help="class-token weight in TCS scoring; ignored on Swin, which has no class token",
    )
    tcs.add_argument(
        "--sample-ratio", type=float, default=None,
        help="fraction of tokens TCS estimates its scores from "
        "(default: 0.02 for ViT, 0.2 for Swin)",
    )
    tcs.add_argument(
        "--swin-attn-weighting", action="store_true",
        help="Swin only: weight each token by the attention it receives, instead of scoring "
        "on magnitude alone. Off by default -- configs/tcs.json was measured without it",
    )

    # Optimiser
    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR")
    parser.add_argument("--min-lr", type=float, default=1e-5, metavar="LR")
    parser.add_argument("--warmup-lr", type=float, default=1e-6, metavar="LR")
    parser.add_argument("--warmup-epochs", type=int, default=0, metavar="N")
    parser.add_argument("--decay-rate", "--dr", type=float, default=0.1, metavar="RATE")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=None, metavar="NORM")
    parser.add_argument(
        "--scale-lr", action="store_true",
        help="scale --lr linearly by batch_size * world_size / 512",
    )

    # Augmentation
    parser.add_argument("--color-jitter", type=float, default=0.4, metavar="PCT")
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1", metavar="NAME")
    parser.add_argument("--smoothing", type=float, default=0.1, help="label smoothing")
    parser.add_argument("--train-interpolation", type=str, default="bicubic")
    parser.add_argument("--repeated-aug", action="store_true")
    parser.add_argument("--no-repeated-aug", action="store_false", dest="repeated_aug")
    parser.set_defaults(repeated_aug=True)
    parser.add_argument("--reprob", type=float, default=0.25, metavar="PCT", help="random erase prob")
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)
    parser.add_argument("--resplit", action="store_true", default=False)
    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--cutmix-minmax", type=float, nargs="+", default=None)
    parser.add_argument("--mixup-prob", type=float, default=1.0)
    parser.add_argument("--mixup-switch-prob", type=float, default=0.5)
    parser.add_argument("--mixup-mode", type=str, default="batch")

    # Data
    parser.add_argument("--data-path", default="./data/imagenet", type=str)
    parser.add_argument(
        "--data-set", default="IMNET", choices=["CIFAR", "IMNET", "INAT", "INAT19"], type=str
    )
    parser.add_argument(
        "--inat-category", default="name",
        choices=["kingdom", "phylum", "class", "order", "supercategory", "family", "genus", "name"],
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin-mem", action="store_true")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Bookkeeping
    parser.add_argument("--output_dir", default="./log/temp")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume training from checkpoint")
    parser.add_argument("--autoresume", action="store_true", help="resume from <output_dir>/checkpoint.pth")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")
    parser.add_argument(
        "--dist-eval", action="store_true", default=True,
        help="shard the validation set across ranks (default)",
    )
    parser.add_argument(
        "--no-dist-eval", action="store_false", dest="dist_eval",
        help="evaluate the full validation set sequentially; use this for final numbers, "
        "since sharding pads the set with duplicates when it does not divide evenly",
    )
    parser.add_argument("--print-freq", default=10, type=int)

    # Distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--port", default="15662", type=str, help="master port for SLURM launches")
    parser.add_argument("--dist_url", default="env://")

    return parser


def load_pretrained(model, path, logger):
    """Load `path` into `model`, dropping a mismatched head and interpolating pos_embed."""
    checkpoint = utils.load_checkpoint(path)
    state_dict = checkpoint.get("model", checkpoint)

    target = model.state_dict()
    head_keys = tuple(
        f"{prefix}.{suffix}"
        for prefix in ("head", "head.fc", "head_dist")  # Swin wraps its head in ClassifierHead
        for suffix in ("weight", "bias")
    )
    for key in head_keys:
        if key in state_dict and (key not in target or state_dict[key].shape != target[key].shape):
            logger.info(f"dropping {key} (shape mismatch with the target head)")
            del state_dict[key]

    # Swin has no positional embedding to resize -- its relative position bias is per window.
    if (
        "pos_embed" in state_dict
        and "pos_embed" in target
        and state_dict["pos_embed"].shape != target["pos_embed"].shape
    ):
        state_dict["pos_embed"] = _interpolate_pos_embed(model, state_dict["pos_embed"])
        logger.info(f"interpolated pos_embed to {tuple(state_dict['pos_embed'].shape)}")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(f"loaded {path}")
    if msg.missing_keys:
        logger.info(f"  missing keys: {msg.missing_keys}")
    if msg.unexpected_keys:
        logger.info(f"  unexpected keys: {msg.unexpected_keys}")

    if any(k.startswith("head.") for k in msg.missing_keys):
        classifier = model.get_classifier() if hasattr(model, "get_classifier") else model.head
        torch.nn.init.trunc_normal_(classifier.weight, std=0.02)
        torch.nn.init.zeros_(classifier.bias)
        logger.info(f"re-initialised head for {classifier.out_features} classes")

    return checkpoint


def _interpolate_pos_embed(model, pos_embed):
    """Bicubically resize a positional embedding to this model's patch grid."""
    embed_dim = pos_embed.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra = model.pos_embed.shape[-2] - num_patches
    orig_size = int((pos_embed.shape[-2] - num_extra) ** 0.5)
    new_size = int(num_patches**0.5)

    extra = pos_embed[:, :num_extra]
    tokens = pos_embed[:, num_extra:]
    tokens = tokens.reshape(-1, orig_size, orig_size, embed_dim).permute(0, 3, 1, 2)
    tokens = torch.nn.functional.interpolate(
        tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
    )
    tokens = tokens.permute(0, 2, 3, 1).flatten(1, 2)
    return torch.cat((extra, tokens), dim=1)


def build_scheduler(optimizer, args):
    """Cosine schedule, tolerating timm's rename of `decay_rate` to `cycle_decay`."""
    kwargs = dict(
        t_initial=args.epochs,
        lr_min=args.min_lr,
        warmup_t=args.warmup_epochs,
        warmup_lr_init=args.warmup_lr,
    )
    accepted = inspect.signature(CosineLRScheduler.__init__).parameters
    for name in ("cycle_decay", "decay_rate"):
        if name in accepted:
            kwargs[name] = args.decay_rate
            break
    return CosineLRScheduler(optimizer, **kwargs)


def apply_schedule_config(args, num_blocks, logger):
    """Fill unset compression flags from `--tcs-config` at `--target-flops`.

    Explicit flags win, so a config entry can be used as a starting point and overridden one
    field at a time.
    """
    if args.target_flops is None:
        return

    config = resolve_config(args.model, args.target_flops, path=args.tcs_config)
    logger.info("Loaded schedule from config:\n" + config.summary())

    if len(config.fc1_prune_ratios) != num_blocks:
        raise ValueError(
            f"config for {config.model} @ {config.target_flops}G has "
            f"{len(config.fc1_prune_ratios)} ratios but the model has {num_blocks} blocks"
        )

    if args.fc1_prune_ratio is None:
        args.fc1_prune_ratio = config.fc1_prune_ratios
    if args.fc2_prune_ratio is None:
        args.fc2_prune_ratio = config.fc2_prune_ratios

    defaults = get_args_parser().parse_args([])
    if args.head_sparsity == defaults.head_sparsity:
        args.head_sparsity = [config.head_sparsity]
    if args.importance == defaults.importance:
        args.importance = config.importance
    if args.coupling == defaults.coupling:
        args.coupling = config.coupling
    if not args.weight_pruning:
        # A schedule implies its weight pruning; without it the FLOPs budget is not met.
        logger.info("enabling --weight-pruning, required by the loaded schedule")
        args.weight_pruning = True


def resolve_ratios(values, num_blocks, name):
    """Expand a CLI ratio list to one entry per block."""
    if values is None:
        return [0.0] * num_blocks
    if len(values) == 1:
        return [float(values[0])] * num_blocks
    if len(values) != num_blocks:
        raise ValueError(f"--{name} needs 1 or {num_blocks} values, got {len(values)}")
    return [float(v) for v in values]


def main(args):
    utils.init_distributed_mode(args)

    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir, dist_rank=utils.get_rank())
    logger.info(args)

    device = torch.device(args.device)
    torch.manual_seed(args.seed + utils.get_rank())
    np.random.seed(args.seed + utils.get_rank())
    cudnn.benchmark = True

    # ---------------------------------------------------------------------------- data
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    if args.repeated_aug:
        sampler_train = RASampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            logger.info(
                "Validation set is not divisible by the process count; duplicate entries will "
                "be added, so numbers shift slightly. Use one process for final results."
            )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = MultiEpochsDataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
    )
    data_loader_val = MultiEpochsDataLoader(
        dataset_val, sampler=sampler_val, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
    )

    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    mixup_fn = (
        Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes,
        )
        if mixup_active
        else None
    )

    # --------------------------------------------------------------------------- model
    logger.info(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained_timm and not args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
    )
    if args.pretrained:
        load_pretrained(model, args.pretrained, logger)

    # For Swin this counts through the stages, so a schedule is one flat per-block vector
    # whichever backbone it is for.
    num_blocks = count_blocks(model)
    apply_schedule_config(args, num_blocks, logger)

    fc1_ratios = resolve_ratios(args.fc1_prune_ratio, num_blocks, "fc1-prune-ratio")
    fc2_ratios = resolve_ratios(args.fc2_prune_ratio, num_blocks, "fc2-prune-ratio")
    head_sparsity = (
        args.head_sparsity[0] if len(args.head_sparsity) == 1
        else resolve_ratios(args.head_sparsity, num_blocks, "head-sparsity")
    )

    if args.print_flops:
        spec = spec_for_model(model, num_classes=args.nb_classes)
        baseline = toast_flops(spec)
        compressed = toast_flops(
            spec,
            head_sparsity=head_sparsity if args.weight_pruning else None,
            fc1_prune_ratios=fc1_ratios,
            fc2_prune_ratios=fc2_ratios,
            skip_first_block=args.skip_first_block,
        )
        logger.info("FLOPs (analytic, for the re-packed model)\n" + compressed.summary(baseline))

    # Token Channel Selection is always installed; all-zero ratios make it a no-op, which
    # keeps the model class identical between baseline and compressed runs.
    apply_toast(
        model,
        fc1_prune_ratios=fc1_ratios,
        fc2_prune_ratios=fc2_ratios,
        cls_weight=args.cls_weight,
        sample_ratio=args.sample_ratio,
        swin_attn_weighting=args.swin_attn_weighting,
    )
    if any(fc1_ratios) or any(fc2_ratios):
        logger.info(f"TCS fc1 ratios: {fc1_ratios}")
        logger.info(f"TCS fc2 ratios: {fc2_ratios}")

    masks = None
    if args.weight_pruning:
        pruner = StructuredCoupledPruner(
            model,
            head_sparsity=head_sparsity,
            score=args.importance,
            coupling=args.coupling,
            skip_first_block=args.skip_first_block,
            verbose=False,
        )
        masks = pruner.masks
        logger.info(pruner.report())

    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters:,}")

    if args.eval:
        stats = evaluate(data_loader_val, model, device, logger, masks, args.print_freq)
        logger.info(f"Acc@1 {stats['acc1']:.3f} on {len(dataset_val)} test images")
        return

    # ------------------------------------------------------------------------ optimiser
    lr = args.lr
    if args.scale_lr:
        lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        logger.info(f"scaled lr: {lr:g} (base {args.lr:g}, batch {args.batch_size}, world {num_tasks})")

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(), lr=lr, weight_decay=args.weight_decay
    )
    loss_scaler = utils.NativeScalerWithGradNormCount()
    lr_scheduler = build_scheduler(optimizer, args)

    if mixup_active:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.autoresume and (output_dir / "checkpoint.pth").exists():
        args.resume = str(output_dir / "checkpoint.pth")
    if args.resume:
        checkpoint = utils.load_checkpoint(args.resume)
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        if {"optimizer", "lr_scheduler", "epoch"} <= checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
        logger.info(f"resumed from {args.resume} at epoch {args.start_epoch}")

    # ----------------------------------------------------------------------------- loop
    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler,
            max_norm=args.clip_grad, mixup_fn=mixup_fn, set_training_mode=True, logger=logger,
            masks=masks, mask_every_step=args.mask_every_step, print_freq=args.print_freq,
        )
        lr_scheduler.step(epoch)

        def training_state():
            return {
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "scaler": loss_scaler.state_dict(),
                # A plain dict, not the Namespace: keeps the checkpoint readable under
                # torch.load's safe weights_only=True default.
                "args": vars(args),
                **({"masks": masks} if masks is not None else {}),
            }

        # Mid-training resume point, written before the projection below so that resuming
        # continues the same trajectory.
        utils.save_on_master(
            {"model": model_without_ddp.state_dict(), **training_state()},
            output_dir / "checkpoint.pth",
        )

        # Projects the attention weights onto their masks first, so the accuracy below is
        # the sparse model's.
        test_stats = evaluate(data_loader_val, model, device, logger, masks, args.print_freq)
        logger.info(f"Acc@1 {test_stats['acc1']:.3f} on {len(dataset_val)} test images")

        if utils.is_main_process() and test_stats["acc1"] > max_accuracy:
            # Snapshotted after that projection, so the saved weights are exactly sparse and
            # `toast.dense` can re-pack them. Keep this below `evaluate`.
            utils.save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    **training_state(),
                    "accuracy": test_stats["acc1"],
                    "test_stats": test_stats,
                },
                output_dir / "model_best.pth",
            )
            logger.info(f"new best: {test_stats['acc1']:.3f}% (was {max_accuracy:.3f}%)")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        logger.info(f"Max accuracy: {max_accuracy:.2f}%")

        if utils.is_main_process():
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total = datetime.timedelta(seconds=int(time.time() - start_time))
    logger.info(f"Training time {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ToaST", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
