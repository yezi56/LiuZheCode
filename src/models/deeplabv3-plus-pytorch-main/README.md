# DeepLabV3+ Research Extensions

This directory contains the paper-experiment version of DeepLabV3+.

Current plug-in modules:

- `CBAM`
- `PPM after ASPP`
- `Focal Loss`
- `MixUp / CutMix`
- `VOC / VOC2 / VOC2_iter1` dataset switching
- `MobileNetV2 + Lite Swin Transformer` dual-backbone option

## Supported backbones

You can now switch backbone with:

- `mobilenet`
- `mobilenet_swin`
- `xception`

## Existing experimental modules

### 1. CBAM

Enable:

```powershell
python train.py --attention-type cbam
```

Disable:

```powershell
python train.py --attention-type none
```

### 2. PPM after ASPP

Enable:

```powershell
python train.py --use-ppm true
```

Custom pooling bins:

```powershell
python train.py --use-ppm true --ppm-bins 1 2 3 6
```

### 3. Focal Loss

```powershell
python train.py --focal-loss true --focal-alpha 0.5 --focal-gamma 2.0
```

### 4. MixUp / CutMix

MixUp:

```powershell
python train.py --mix-mode mixup --mix-prob 0.5 --mixup-alpha 0.4
```

CutMix:

```powershell
python train.py --mix-mode cutmix --mix-prob 0.5 --cutmix-alpha 1.0
```

## Dual Backbone: MobileNetV2 + Lite Swin Transformer

### Design goal

The dual-backbone version is designed to add Transformer representation ability without letting parameter count explode.

### How parameter growth is controlled

1. Shared low-level feature extraction

- The Swin branch does **not** start from the raw image.
- It reuses the low-level feature map already computed by MobileNetV2.
- This avoids duplicating the early CNN stem.

2. Small Swin branch

Current default settings are intentionally lightweight:

- `embed_dim = 192`
- `depth = 4`
- `heads = 4`
- `window_size = 7`
- `mlp_ratio = 2.0`

3. Window attention instead of global full attention

- The auxiliary branch uses local window attention.
- This is lighter and more suitable than full global attention for dense prediction.

4. Shifted window interaction

- Adjacent blocks alternate between regular windows and shifted windows.
- This improves cross-window information exchange without using full global attention.

5. Lightweight fusion

- MobileNetV2 high-level feature and Swin feature are fused with a single `1x1` conv block.
- No second heavy decoder is introduced.

### Implementation path

The segmentation-side lightweight implementation used here is:

- [nets/lite_swin.py](/D:/Code/all/src/models/deeplabv3-plus-pytorch-main/nets/lite_swin.py)
- [nets/deeplabv3_plus_dual.py](/D:/Code/all/src/models/deeplabv3-plus-pytorch-main/nets/deeplabv3_plus_dual.py)

### Training examples

Baseline MobileNetV2:

```powershell
python train.py --backbone mobilenet
```

Dual backbone:

```powershell
python train.py --backbone mobilenet_swin
```

Dual backbone + CBAM:

```powershell
python train.py --backbone mobilenet_swin --attention-type cbam
```

Dual backbone + CBAM + PPM:

```powershell
python train.py --backbone mobilenet_swin --attention-type cbam --use-ppm true
```

Dual backbone + CBAM + PPM + Focal:

```powershell
python train.py --backbone mobilenet_swin --attention-type cbam --use-ppm true --focal-loss true
```

### Initialization strategy

For `mobilenet_swin`, the current recommended initialization is:

- use `model_data/deeplab_mobilenetv2.pth`
- MobileNetV2-compatible parameters load into the shared CNN branch
- the newly added Swin branch is initialized randomly

Command example:

```powershell
python train.py --backbone mobilenet_swin --model-path model_data\deeplab_mobilenetv2.pth
```

## Recommended ablation route

1. `DeepLabV3+`
2. `DeepLabV3+ + CBAM`
3. `DeepLabV3+ + CBAM + PPM`
4. `DeepLabV3+ + CBAM + PPM + Focal`
5. `DeepLabV3+ + CBAM + PPM + Focal + MixUp`
6. `DeepLabV3+ + MobileNetV2 + Lite Swin Transformer`
7. `DeepLabV3+ + MobileNetV2 + Lite Swin Transformer + CBAM + PPM`

## VOC2 commands

First round:

```powershell
python train.py --dataset-name VOC2 --backbone mobilenet_swin --attention-type cbam --use-ppm true --focal-loss true --mix-mode mixup --mix-prob 0.3 --mixup-alpha 0.4 --model-path model_data\deeplab_mobilenetv2.pth --freeze-epoch 30 --unfreeze-epoch 240 --freeze-batch-size 4 --unfreeze-batch-size 2 --num-workers 0 --init-lr 0.0035 --dice-loss true --save-period 10 --eval-period 10 --save-dir outputs\voc2_seed_dual\weights --log-dir outputs\voc2_seed_dual\logs
```

Second round after refine:

```powershell
python train.py --dataset-name VOC2_iter1 --backbone mobilenet_swin --attention-type cbam --use-ppm true --focal-loss true --mix-mode mixup --mix-prob 0.3 --mixup-alpha 0.4 --model-path outputs\voc2_seed_dual\weights\best_epoch_weights.pth --freeze-epoch 20 --unfreeze-epoch 180 --freeze-batch-size 4 --unfreeze-batch-size 2 --num-workers 0 --init-lr 0.0025 --dice-loss true --save-period 10 --eval-period 10 --save-dir outputs\voc2_iter1_dual\weights --log-dir outputs\voc2_iter1_dual\logs
```
