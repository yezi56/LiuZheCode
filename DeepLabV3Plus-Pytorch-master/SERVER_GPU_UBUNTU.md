# Ubuntu GPU Server Guide

This guide is for the current `black_rot` dataset layout and the bundled `server_run.sh` launcher.

## 1. Expected layout on the server

```text
DeepLabV3Plus-Pytorch-master/
|- server_run.sh
|- main.py
|- predict.py
|- requirements.txt
`- datasets/
   `- data/
      `- black_rot/
         |- images/
         `- runs/
            `- 20260408_151834/
               `- lesion_mask/
```

Defaults used by `server_run.sh`:

- images: `datasets/data/black_rot/images`
- masks: latest `datasets/data/black_rot/runs/*/lesion_mask`
- virtualenv: `.venv`

## 2. One-command environment setup

Recommended:

```bash
bash server_run.sh setup --torch-channel cu124
```

If the machine already has `python3`, `python3-venv`, and `pip`, and you do not want `apt`:

```bash
bash server_run.sh setup --skip-apt --torch-channel cu124
```

If `cu124` is not suitable for your server, try one of:

- `--torch-channel cu130`
- `--torch-channel cu121`
- `--torch-channel cu118`

## 3. One-command training

Typical training command:

```bash
bash server_run.sh train \
  --ckpt ./checkpoints/your_model.pth \
  --epochs 20 \
  --batch-size 4 \
  --val-batch-size 2 \
  --crop-size 512 \
  --gpu-id 0
```

Notes:

- `--epochs` is a convenience option
- the script converts `--epochs` into `main.py --total_itrs`
- if you want direct control, use `--total-itrs`
- if `--ckpt` is omitted, training starts from random initialization
- add `--pretrained-backbone` to use an ImageNet pretrained backbone when no checkpoint is provided

More complete example:

```bash
bash server_run.sh train \
  --data-root ./datasets/data/black_rot \
  --model deeplabv3plus_mobilenet \
  --epochs 30 \
  --batch-size 8 \
  --val-batch-size 4 \
  --num-workers 8 \
  --val-num-workers 4 \
  --crop-size 512 \
  --val-percent 0.2 \
  --lr 0.001 \
  --gpu-id 0
```

Pass extra arguments directly to `main.py` after `--`:

```bash
bash server_run.sh train --epochs 20 -- --print_interval 5 --val_interval 50
```

## 4. One-command prediction

Run prediction for the whole image folder:

```bash
bash server_run.sh predict \
  --ckpt ./checkpoints/best_deeplabv3plus_mobilenet_black_rot_os16.pth \
  --input ./datasets/data/black_rot/images \
  --gpu-id 0
```

Default output directories:

- color predictions: `./outputs/color`
- raw class-index masks: `./outputs/mask`

Custom output directories:

```bash
bash server_run.sh predict \
  --ckpt ./checkpoints/best_deeplabv3plus_mobilenet_black_rot_os16.pth \
  --input ./datasets/data/black_rot/images \
  --save-color-dir ./outputs/infer_color \
  --save-mask-dir ./outputs/infer_mask
```

## 5. Common options

- `--torch-channel cu130|cu124|cu121|cu118|cpu`
- `--venv-dir PATH`
- `--skip-apt`
- `--data-root PATH`
- `--image-dir PATH`
- `--mask-dir PATH`
- `--ckpt PATH`
- `--epochs N`
- `--total-itrs N`
- `--batch-size N`
- `--val-batch-size N`
- `--crop-size N`
- `--gpu-id ID`
- `--num-workers N`
- `--val-num-workers N`
- `--lr FLOAT`
- `--pretrained-backbone`
- `--separable-conv`

Full help:

```bash
bash server_run.sh --help
```

## 6. Recommended sequence

First time on the server:

```bash
bash server_run.sh setup --torch-channel cu124
```

Then train:

```bash
bash server_run.sh train --epochs 20 --batch-size 4 --val-batch-size 2
```

Then predict:

```bash
bash server_run.sh predict --ckpt ./checkpoints/best_deeplabv3plus_mobilenet_black_rot_os16.pth
```

## 7. Notes

- headless server mode works without `visdom` unless you enable `--enable_vis`
- `black_rot` automatically uses the latest `runs/*/lesion_mask` if `--mask-dir` is not given
- if you later switch to a different pseudo-label directory, pass `--mask-dir`
- training still runs through `main.py`, prediction still runs through `predict.py`
