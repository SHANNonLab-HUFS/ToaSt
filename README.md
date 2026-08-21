# ToaST

Official implementation of **ToaST**, ICML 2026.

> **TODO** — paper title, author list, and arXiv/OpenReview links.

ToaST compresses a vision transformer by attacking the two halves of a block with two
different mechanisms, because they have different structure to exploit.

**Structured Coupled Weight Pruning (SCWP)** — attention. Head dimensions are removed in
coupled pairs: a Q/K dimension leaves `W_Q` and `W_K` together, a V/O dimension leaves `W_V`'s
rows and `W_O`'s columns together. Pruning one side without the other would leave the other
side's parameters computing against zeros. Importance is the distance of each row of the
concatenated pair from that pair's column-wise median, so a dimension survives only when it
matters to both halves. Every head keeps the *same number* of dimensions, which is what lets
the surviving weights re-pack into smaller dense matrices — no gather, no sparse kernel.

**Token Channel Selection (TCS)** — feed-forward. Which FFN channels matter depends on the
image, so they are chosen per forward pass rather than once offline. Channels are scored from
activations: the class token's magnitude, up-weighted because it is what reaches the
classifier, plus the patch tokens' magnitude weighted by how much attention the class token
pays them. Patch statistics come from a 2 % subsample, which keeps the selection cost
negligible against the matmul it shrinks.

The two compose: SCWP fixes a sparse attention structure offline, TCS narrows the FFN at run
time, and `toast.dense` re-packs the result so the savings show up in wall-clock time.

Both stages run on plain ViTs (DeiT, MAE) and on Swin. Nothing in SCWP cares whether attention
is global or windowed, and the feed-forward half is the same either way; what differs is that
Swin nests its blocks in stages and has no class token. [`toast/arch.py`](toast/arch.py)
flattens the nesting, so a schedule is one per-block vector for either backbone, and
[`toast/swin.py`](toast/swin.py) holds the rest — see [Swin](#swin) below.

## Install

```bash
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.x and timm 0.9. `toast/patch.py` resolves timm's
`drop_path`/`drop_path1` and `Mlp.drop`/`drop1` naming differences, so both older and newer
timm releases work.

## Data

ImageNet in the usual `ImageFolder` layout:

```
imagenet/
├── train/<class>/*.JPEG
└── val/<class>/*.JPEG
```

CIFAR-100 is downloaded by torchvision into `--data-path` on first use.

## Quick start

Compression schedules live in [`configs/tcs.json`](configs/tcs.json), keyed by model and FLOPs
budget, so a run is named rather than spelled out:

```bash
python main.py --eval --model deit_small_patch16_224 --target-flops 2.9 \
    --data-path /path/to/imagenet
```

That loads the head sparsity and the per-block fc1/fc2 ratios for that budget, and logs the
FLOPs breakdown. Explicit flags override the config, so you can start from an entry and change
one field:

```bash
python main.py --eval --model deit_small_patch16_224 --target-flops 2.9 \
    --coupling q_only --data-path /path/to/imagenet
```

Or set everything by hand and skip the config:

```bash
python main.py --eval --model deit_small_patch16_224 --data-path /path/to/imagenet \
    --weight-pruning --head-sparsity 90 \
    --fc2-prune-ratio 0 0 0 0 0 0 0 0 0 0 0.9 0.9
```

Swin takes the same flags:

```bash
python main.py --eval --model swin_small_patch4_window7_224 --target-flops 5.4 \
    --data-path /path/to/imagenet

# the same schedule, scoring channels by attention received rather than magnitude
python main.py --eval --model swin_small_patch4_window7_224 --target-flops 5.4 \
    --data-path /path/to/imagenet --swin-attn-weighting
```

Compress and fine-tune on ImageNet:

```bash
IMAGENET=/path/to/imagenet TARGET_FLOPS=2.9 bash scripts/train_imagenet.sh
```

Transfer a compressed checkpoint to CIFAR-100:

```bash
CIFAR100=/path/to/cifar100 PRETRAINED=./log/imagenet_deit_small_patch16_224_sp90/model_best.pth \
    bash scripts/finetune_cifar100.sh
```

Every script takes its settings from the environment (`MODEL`, `GPUS`, `EPOCHS`,
`HEAD_SPARSITY`, `FC1_RATIOS`, `FC2_RATIOS`, …) and forwards extra arguments straight to
`main.py`.

## Results

ImageNet-1k, 224x224, measured on an H100 at batch size 128. Every schedule below lives in
[`configs/tcs.json`](configs/tcs.json) next to the numbers it produced, so a table row and the
command that reproduces it stay together. Print all models:

```bash
python -m experiments.flops_table            # or --markdown
```

**DeiT-T**

| method | GFLOPs | Acc@1 | Acc@5 | img/s | speedup |
|---|---|---|---|---|---|
| baseline | 1.3 | 72.20 | 91.10 | 2091 | 1.00x |
| ToMe | 0.7 | 71.25 | 90.74 | 2485 | 1.19x |
| DiffRate | 0.9 | 71.78 | 90.87 | 2423 | 1.16x |
| DiffRate | 0.8 | 71.67 | 90.78 | 2256 | 1.08x |
| **ToaST** | 0.9 | 69.93 | 89.67 | 3469 | 1.66x |
| **ToaST** | 0.8 | **74.30** | **92.26** | **4365** | **2.09x** |
| **ToaST** | 0.76 | 74.25 | 92.65 | 4250 | 2.03x |

**DeiT-S**

| method | GFLOPs | Acc@1 | Acc@5 | img/s | speedup |
|---|---|---|---|---|---|
| baseline | 4.6 | 79.82 | 94.95 | 2313 | 1.00x |
| ToMe | 2.7 | 79.35 | 94.65 | 2737 | 1.18x |
| DiffRate | 2.9 | 79.56 | 94.80 | 2808 | 1.21x |
| DiffRate | 2.7 | 79.38 | 94.65 | 3260 | 1.41x |
| DiffRate | 2.5 | 79.09 | 94.50 | 3257 | 1.41x |
| **ToaST** | 2.9 | 80.77 | 95.29 | 4519 | 1.95x |
| **ToaST** | 2.7 | **83.89** | **97.13** | 4659 | 2.01x |
| **ToaST** | 2.5 | 83.40 | 96.97 | **4783** | **2.07x** |

**DeiT-B**

| method | GFLOPs | Acc@1 | Acc@5 | img/s | speedup |
|---|---|---|---|---|---|
| baseline | 17.6 | 81.80 | 95.60 | 1123 | 1.00x |
| ToMe | 11.5 | 80.59 | 94.83 | 1628 | 1.45x |
| DiffRate | 11.5 | 81.51 | 95.40 | 1554 | 1.38x |
| DiffRate | 10.4 | 81.01 | 95.02 | 1660 | 1.48x |
| **ToaST** | 11.5 | 82.25 | 96.07 | 1621 | 1.44x |
| **ToaST** | 10.7 | **84.82** | **97.10** | 1691 | 1.51x |
| **ToaST** | 10.4 | 82.87 | 96.29 | 1708 | **1.52x** |
| **ToaST** | 10.27 | 82.02 | 95.57 | **1711** | **1.52x** |

**ViT-B / ViT-L / ViT-H (MAE)**

| model | method | GFLOPs | Acc@1 | Acc@5 | img/s | speedup |
|---|---|---|---|---|---|---|
| ViT-B | baseline | 17.6 | 83.75 | 96.54 | 1140 | 1.00x |
| ViT-B | ToMe (r=13) | 10.4 | 81.87 | 96.02 | 1783 | 1.56x |
| ViT-B | DiffRate | 11.5 | 82.90 | 96.14 | 1553 | 1.36x |
| ViT-B | **ToaST** | 11.5 | 83.44 | 95.49 | 1624 | 1.42x |
| ViT-B | **ToaST** | 11.0 | **84.13** | **96.39** | **1693** | **1.48x** |
| ViT-L | baseline | 61.6 | 85.96 | 97.55 | 349 | 1.00x |
| ViT-L | ToMe (r=6) | 38.5 | 84.58 | 97.12 | 523 | 1.50x |
| ViT-L | DiffRate | 38.5 | 85.38 | 97.39 | 513 | 1.47x |
| ViT-L | **ToaST** | 42.3 | 81.86 | 95.84 | 487 | 1.40x |
| ViT-L | **ToaST** | 38.5 | **88.94** | **97.95** | **527** | **1.51x** |
| ViT-H | baseline | 167.4 | 86.88 | 98.07 | 130 | 1.00x |
| ViT-H | ToMe (r=5) | 113.9 | 86.28 | 97.88 | 186 | 1.43x |
| ViT-H | DiffRate | 103.4 | 86.65 | 97.88 | 203 | 1.56x |
| ViT-H | **ToaST** | 103.4 | **90.03** | **98.77** | 203 | 1.56x |
| ViT-H | **ToaST** | 101.4 | 88.52 | 98.29 | 206 | **1.59x** |
| ViT-H | **ToaST** | 100.4 | 86.92 | 97.68 | 206 | **1.59x** |

Larger backbones reach their compressed accuracy in fewer fine-tuning epochs (297 for ViT-B,
139 for ViT-L, 15 for ViT-H); the config records the epoch count per entry.

**Swin (classification)**

| model | method | GFLOPs | Acc@1 | Acc@5 | img/s | speedup |
|---|---|---|---|---|---|---|
| Swin-T | baseline | 4.5 | 81.20 | 95.50 | 2611 | 1.00x |
| Swin-T | **ToaST** | 3.1 | **81.76** | **95.70** | **2706** | **1.04x** |
| Swin-S | baseline | 8.7 | 83.20 | 96.20 | 1534 | 1.00x |
| Swin-S | STViT-R | 5.8 | 82.60 | 96.07 | 1647 | 1.07x |
| Swin-S | **ToaST** | 5.4 | **84.65** | **96.80** | **1909** | **1.24x** |
| Swin-B | baseline | 15.4 | 83.50 | 96.50 | 1100 | 1.00x |
| Swin-B | STViT-R | 10.3 | 83.20 | 96.40 | 1206 | 1.10x |
| Swin-B | **ToaST** | 8.8 | **85.21** | 96.50 | **1409** | **1.30x** |

**COCO detection and segmentation** (Cascade Mask R-CNN, Swin backbones)

| backbone | method | box mAP | mask mAP |
|---|---|---|---|
| Swin-S | baseline | 51.9 | 45.0 |
| Swin-S | **ToaST** | **52.2** | 44.7 |
| Swin-B | baseline | 51.9 | 45.0 |
| Swin-B | **ToaST** | **52.2** | 44.7 |

The Swin schedules are in the config as flat per-block vectors, alongside `fc1_by_stage` /
`fc2_by_stage`, which are the same numbers grouped by stage for reading. Run them exactly like
the DeiT ones — `--target-flops 5.4 --model swin_small_patch4_window7_224` — or through
`scripts/train_imagenet.sh` with `MODEL=` set.

The COCO rows above used the same schedules on a detection backbone; that pipeline is a
separate codebase and is not part of this release.

## FLOPs

[`toast/flops.py`](toast/flops.py) counts multiply-accumulates, the same convention the
token-compression literature uses, so budgets here are directly comparable. SCWP enters as a
multiplier on the MHSA term and TCS as one on the FFN term; the module docstring derives both.

```bash
python -m toast.flops --model deit_base_patch16_224 --head-sparsity 90 \
    --fc2-prune-ratio 0 0 0 0 0 0 0 0 0 0 0.7 0.9
python -m toast.flops --model swin_small_patch4_window7_224 --head-sparsity 90
```

`main.py` logs the same breakdown at startup. Swin is counted per stage, since each has its own
token count and width, with windowed attention costing `2*N*window_area*C` instead of
`2*N^2*C`, and the patch-merging linears counted separately — `swin_flops` reproduces the
published 4.5 / 8.7 / 15.4 GFLOPs baselines to the decimal they are quoted at.

## Implementation notes

**Masked and dense forms.** [`toast/patch.py`](toast/patch.py) keeps tensor shapes and zeroes
the channels a selection rejects; this is the form used for training and accuracy.
[`toast/dense.py`](toast/dense.py) re-packs the same model into physically smaller matrices for
latency, which is possible because SCWP gives every head the same budget. The FFN's `norm2`
normalises over the kept channels in the dense form and over all of them in the masked form, so
take accuracy from the masked path and latency from the dense one --
[`experiments/latency.py`](experiments/latency.py) reports both.

**Weight masks during fine-tuning.** `evaluate` projects the attention weights back onto their
masks before measuring, and `model_best.pth` is written after that projection, so released
weights are exactly sparse and re-pack cleanly. `--mask-every-step` projects after every
optimiser step as well.

<a id="swin"></a>
**Swin.** Blocks are indexed globally, counting through the stages, so Swin-T takes a
twelve-element ratio vector exactly as DeiT-T does and `--head-sparsity`'s per-block form lines
up the same way. Block 0 — the first block of the first stage — is the one left dense.
[`toast/swin.py`](toast/swin.py) subclasses timm's `SwinTransformerBlock` and overrides only the
feed-forward half, so window shifting, padding and the attention mask stay timm's code and keep
working across its versions; `toast.dense` swaps in a re-packed window attention the same way.

The score is where Swin genuinely differs. A ViT weights each patch by the class token's
attention to it — one row of the map. A window map has no distinguished row, so there are two
candidate substitutes, and only one of them carries information:

- **each token's mean attention over its window** — i.e. averaging the map along its key axis.
  Softmax normalises exactly that axis, so this is `1 / window_area` for every token alike. A
  constant cannot reorder anything: weighting by it selects precisely the channels magnitude
  alone selects.
- **the attention each token receives** — averaging along the query axis instead. This does
  vary between tokens, and is the real analogue of the ViT weighting.

The default is magnitude alone, which is what the schedules in `configs/tcs.json` were measured
with. `--swin-attn-weighting` switches to attention-received; it needs the map, so it installs a
non-fused `SwinToastWindowAttention` on the blocks that select channels. Mapping the weights
back from window order to image order (window reverse, crop, un-shift) is part of that path and
is not optional — window order is not image order, and a weight that lands on the wrong token is
worse than no weight at all.

Consequently `--cls-weight` has no effect on Swin, and `--sample-ratio` defaults to 0.2 there
against 0.02 for ViT.

## Citation

> **TODO** — replace with the camera-ready entry.

```bibtex
@inproceedings{toast2026,
  title     = {TODO},
  author    = {TODO},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## Acknowledgements

This codebase started as a fork of [DiffRate](https://github.com/OpenGVLab/DiffRate) and
retains training scaffolding from [DeiT](https://github.com/facebookresearch/deit) and
[timm](https://github.com/huggingface/pytorch-image-models). DiffRate's own compression
machinery is not part of ToaST and has been removed; use the upstream repository to reproduce
it as a baseline. See [NOTICE](NOTICE) for the full attribution.

Licensed under Apache-2.0. See [LICENSE](LICENSE).
