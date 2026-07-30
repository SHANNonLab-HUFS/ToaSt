"""Train and evaluate loops used by `main.py`."""

import math
import sys
from typing import Iterable, Optional, Sequence

import torch
from timm.data import Mixup
from timm.utils import accuracy

import utils
from toast import reapply_masks


def train_one_epoch(
    model: torch.nn.Module,
    criterion,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    set_training_mode: bool = True,
    logger=None,
    masks: Optional[Sequence[torch.Tensor]] = None,
    mask_every_step: bool = False,
    print_freq: int = 10,
):
    """One epoch of fine-tuning.

    Args:
        masks: SCWP masks. Only used when `mask_every_step` is set.
        mask_every_step: project pruned weights back onto their masks after every optimiser
            step, so the training forward pass is sparse too. Off by default, matching the
            published setup, where the projection happens before each evaluation (see
            `evaluate`).
    """
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    if mask_every_step and masks is None:
        raise ValueError("mask_every_step requires the SCWP masks")

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            loss = criterion(model(samples), targets)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.info(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        is_second_order = getattr(optimizer, "is_second_order", False)
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=utils.unwrap_model(model).parameters(),
            create_graph=is_second_order,
        )

        if mask_every_step:
            reapply_masks(utils.unwrap_model(model), masks)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, logger=None, masks=None, print_freq: int = 10):
    """Top-1/top-5 on `data_loader`.

    If `masks` is given, the attention weights are projected onto them first, so the measured
    model is the sparse one.
    """
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")

    model.eval()
    if masks is not None:
        reapply_masks(utils.unwrap_model(model), masks)

    for images, target in metric_logger.log_every(data_loader, print_freq, "Test:", logger):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.meters["loss"].update(loss.item(), n=batch_size)
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    if logger is not None:
        logger.info(
            f"* Acc@1 {metric_logger.acc1.global_avg:.3f} "
            f"Acc@5 {metric_logger.acc5.global_avg:.3f} "
            f"loss {metric_logger.loss.global_avg:.3f}"
        )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
