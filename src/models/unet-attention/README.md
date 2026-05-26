# U-Net Attention

This directory contains the reusable U-Net code from:

```text
D:\Code\seg\unet
```

Dataset-specific scripts, example datasets, generated predictions, cached files, and old task-specific training outputs were not imported.

## Role

`unet-attention` is a U-Net experiment branch for testing lightweight attention and backbone variants. The reusable attention modules are collected separately in:

```text
D:\Code\all\src\modules\attention_zoo
```

## Dataset Layout

The default generic dataset loader expects:

```text
dataset_root/
  train/
    images/
    masks/
  val/
    images/
    masks/
```

Masks should use integer class IDs. Background is `0`; foreground classes start from `1`.

## Example

```powershell
cd D:\Code\all\src\models\unet-attention
conda activate pytorch
python train.py `
  --data-path D:\SegData\new_dataset `
  --num-classes 1 `
  --model-name unet `
  --save-dir D:\SegRuns\outputs\unet-attention\new_dataset\exp01\weights `
  --log-dir D:\SegRuns\logs\unet-attention\new_dataset\exp01
```
