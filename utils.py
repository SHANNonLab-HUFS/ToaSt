"""Distributed helpers, logging and metric tracking.

Trimmed from the DeiT reference implementation; see NOTICE.
"""

import datetime
import logging
import os
import sys
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist
from termcolor import colored

inf = float("inf")


# ---------------------------------------------------------------------------------------
# Distributed
# ---------------------------------------------------------------------------------------


def _slurm_init(port="15662"):
    """Derive rank/world size/address from SLURM environment variables."""
    import multiprocessing

    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    rank = int(os.environ["SLURM_PROCID"])
    world_size = os.environ["SLURM_NTASKS"]
    node_list = os.environ["SLURM_NODELIST"]
    gpu_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(gpu_id)

    if "[" in node_list:
        beg = node_list.find("[")
        pos1 = node_list.find("-", beg)
        pos1 = 1000 if pos1 < 0 else pos1
        pos2 = node_list.find(",", beg)
        pos2 = 1000 if pos2 < 0 else pos2
        node_list = node_list[: min(pos1, pos2)].replace("[", "")
    addr = node_list[8:].replace("-", ".")

    os.environ["MASTER_PORT"] = str(port)
    os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = world_size
    os.environ["RANK"] = str(rank)

    torch.distributed.init_process_group(backend="nccl")
    return rank, int(world_size), gpu_id


def init_distributed_mode(args):
    """Set `args.distributed`, `args.rank`, `args.gpu` from the launcher's environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.distributed = True

        torch.cuda.set_device(args.gpu)
        args.dist_backend = "nccl"
        print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    elif "SLURM_PROCID" in os.environ:
        args.distributed = True
        args.rank, args.world_size, args.local_rank = _slurm_init(port=args.port)
        args.gpu = args.rank % torch.cuda.device_count()
        args.device = f"cuda:{args.local_rank}"
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """Silence `print` on non-master ranks (``force=True`` overrides)."""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        if is_master or kwargs.pop("force", False):
            builtin_print(*args, **kwargs)
        else:
            kwargs.pop("force", None)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def load_checkpoint(path, map_location="cpu"):
    """Load a checkpoint, tolerating both our format and older pickled ones.

    We store run configuration as a plain dict so the safe `weights_only=True` path works.
    Checkpoints written before that stored an `argparse.Namespace`, which the safe loader
    rejects; those fall back to a full unpickle, so only load them from a source you trust.
    """
    if path.startswith("https"):
        return torch.hub.load_state_dict_from_url(path, map_location=map_location, check_hash=True)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        print(f"{path}: not loadable with weights_only=True, falling back to a full unpickle")
        return torch.load(path, map_location=map_location, weights_only=False)


def unwrap_model(model):
    """The underlying module, whether or not `model` is DDP-wrapped."""
    return model.module if hasattr(model, "module") else model


# ---------------------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------------------


def create_logger(output_dir, dist_rank=0, name=""):
    """Logger writing to stdout (master only) and to ``<output_dir>/log_rank<n>.txt``."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
    color_fmt = (
        colored("[%(asctime)s %(name)s]", "green")
        + colored("(%(filename)s %(lineno)d)", "yellow")
        + ": %(levelname)s %(message)s"
    )

    if dist_rank == 0:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.DEBUG)
        console.setFormatter(logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(console)

    file_handler = logging.FileHandler(os.path.join(output_dir, f"log_rank{dist_rank}.txt"), mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------


class SmoothedValue:
    """A value tracked over a sliding window, plus its global average."""

    def __init__(self, window_size=20, fmt=None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """All-reduce `count` and `total`. Does not synchronise the window."""
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        self.count = int(t[0].item())
        self.total = t[1].item()

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    """Named `SmoothedValue` meters with progress logging."""

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        return self.delimiter.join(f"{name}: {meter}" for name, meter in self.meters.items())

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header="", logger=None):
        log = logger.info if logger is not None else print
        i = 0
        start_time = end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")

        space_fmt = f":{len(str(len(iterable)))}d"
        parts = [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
        if torch.cuda.is_available():
            parts.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(parts)
        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta = str(datetime.timedelta(seconds=int(iter_time.global_avg * (len(iterable) - i))))
                fields = dict(eta=eta, meters=str(self), time=str(iter_time), data=str(data_time))
                if torch.cuda.is_available():
                    fields["memory"] = torch.cuda.max_memory_allocated() / MB
                log(log_msg.format(i, len(iterable), **fields))
            i += 1
            end = time.time()

        total = time.time() - start_time
        log(
            f"{header} Total time: {datetime.timedelta(seconds=int(total))} "
            f"({total / len(iterable):.4f} s / it)"
        )


# ---------------------------------------------------------------------------------------
# AMP
# ---------------------------------------------------------------------------------------


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return torch.tensor(0.0)
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    if norm_type == inf:
        return max(p.grad.detach().abs().max().to(device) for p in parameters)
    return torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
        norm_type,
    )


class NativeScalerWithGradNormCount:
    """AMP grad scaler that also reports the gradient norm."""

    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if not update_grad:
            return None
        self._scaler.unscale_(optimizer)
        if clip_grad is not None:
            assert parameters is not None
            norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        else:
            norm = ampscaler_get_grad_norm(parameters)
        self._scaler.step(optimizer)
        self._scaler.update()
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


# ---------------------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------------------


class _RepeatSampler:
    """Sampler that never stops yielding, so worker processes survive across epochs."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """DataLoader that keeps its workers alive between epochs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)
