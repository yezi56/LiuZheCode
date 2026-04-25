# DeepLabV3+ Research Extensions

这份工程已经被整理成面向论文实验的 `DeepLabV3+` 版本，当前支持的可插拔能力包括：

- `CBAM` 注意力
- `PPM after ASPP`
- `Focal Loss`
- `MixUp / CutMix` 批级增强
- 外部数据集路径与输出路径参数化

## 已接入能力

### 1. CBAM

当前训练默认已经启用：

```text
--attention-type cbam
```

注入位置：

1. low-level feature
2. high-level feature
3. ASPP output
4. decoder fusion output

关闭方式：

```powershell
python train.py --attention-type none
```

### 2. PPM after ASPP

现在已经支持把 `PPM` 接到 `ASPP` 之后：

```text
backbone -> ASPP -> CBAM(aspp) -> PPM -> decoder
```

默认关闭，手动打开：

```powershell
python train.py --use-ppm true
```

自定义金字塔池化尺度：

```powershell
python train.py --use-ppm true --ppm-bins 1 2 3 6
```

### 3. Focal Loss

工程原本已经带了 `Focal_Loss`，现在补成了完整参数化版本：

```powershell
python train.py --focal-loss true --focal-alpha 0.5 --focal-gamma 2.0
```

说明：

- `focal-alpha` 默认 `0.5`
- `focal-gamma` 默认 `2.0`

### 4. MixUp / CutMix

现在已经支持 batch 级别的图像与 mask 混合增强：

```powershell
python train.py --mix-mode mixup --mix-prob 0.5 --mixup-alpha 0.4
```

或：

```powershell
python train.py --mix-mode cutmix --mix-prob 0.5 --cutmix-alpha 1.0
```

说明：

- `mix-mode` 可选：`none` / `mixup` / `cutmix`
- `mix-prob` 控制 batch 触发概率
- `mixup-alpha` 控制 MixUp Beta 分布
- `cutmix-alpha` 控制 CutMix Beta 分布

## 推荐实验组合

### Baseline：DeepLabV3+ + CBAM

```powershell
python train.py --attention-type cbam
```

### DeepLabV3+ + CBAM + PPM

```powershell
python train.py --attention-type cbam --use-ppm true
```

### DeepLabV3+ + CBAM + PPM + Focal

```powershell
python train.py --attention-type cbam --use-ppm true --focal-loss true
```

### DeepLabV3+ + CBAM + PPM + Focal + MixUp

```powershell
python train.py --attention-type cbam --use-ppm true --focal-loss true --mix-mode mixup --mix-prob 0.3 --mixup-alpha 0.4
```

### DeepLabV3+ + CBAM + PPM + Focal + CutMix

```powershell
python train.py --attention-type cbam --use-ppm true --focal-loss true --mix-mode cutmix --mix-prob 0.3 --cutmix-alpha 1.0
```

## VOC2 / VOC2_iter1 推荐命令

### VOC2 第一轮

```powershell
python train.py --dataset-name VOC2 --attention-type cbam --use-ppm true --focal-loss true --mix-mode mixup --mix-prob 0.3 --mixup-alpha 0.4 --model-path model_data\deeplab_mobilenetv2.pth --freeze-epoch 30 --unfreeze-epoch 240 --freeze-batch-size 4 --unfreeze-batch-size 2 --num-workers 0 --init-lr 0.0035 --dice-loss true --save-period 10 --eval-period 10 --save-dir outputs\voc2_seed_cbam_ppm\weights --log-dir outputs\voc2_seed_cbam_ppm\logs
```

### VOC2_iter1 第二轮

```powershell
python train.py --dataset-name VOC2_iter1 --attention-type cbam --use-ppm true --focal-loss true --mix-mode mixup --mix-prob 0.3 --mixup-alpha 0.4 --model-path outputs\voc2_seed_cbam_ppm\weights\best_epoch_weights.pth --freeze-epoch 20 --unfreeze-epoch 180 --freeze-batch-size 4 --unfreeze-batch-size 2 --num-workers 0 --init-lr 0.0025 --dice-loss true --save-period 10 --eval-period 10 --save-dir outputs\voc2_iter1_cbam_ppm\weights --log-dir outputs\voc2_iter1_cbam_ppm\logs
```

## 关于你提到的 3 个仓库

### 1. Focal Loss 仓库

你当前工程本来就已经有 `Focal_Loss`，所以：

- 不一定需要额外克隆 `pytorch-multi-class-segmentation-focal-loss`
- 当前更重要的是参数化、训练接入和可切换

这部分已经完成。

### 2. CutMix-PyTorch

你需要的是“思想接入”而不是把整个分类项目生搬硬套进来。  
现在我已经按语义分割场景把 `CutMix` 逻辑做成了当前工程内可插拔的 batch 混合增强。

### 3. PSPNET_tutorial

这个仓库对你当前总仓库来说，**大概率不需要再引入**。

原因是你已经有：

- [pspnet-pytorch-master](/D:/Code/all/src/models/pspnet-pytorch-master)

而且其中已经包含标准 `PPM` 实现：

- [pspnet.py](/D:/Code/all/src/models/pspnet-pytorch-master/nets/pspnet.py:115)

所以从实验功能上看：

- `PSPNET_tutorial` 更偏教学参考
- 你现有 `pspnet-pytorch-master` 更偏可直接训练的工程实现

结论：

**现有 `pspnet-pytorch-master` 已经覆盖了你需要的 Pyramid Pooling 能力，没必要为了 PPM 再额外引一个几乎同功能仓库。**

## 推荐论文实验路线

建议按这个顺序做 ablation：

1. `DeepLabV3+`
2. `DeepLabV3+ + CBAM`
3. `DeepLabV3+ + CBAM + PPM`
4. `DeepLabV3+ + CBAM + PPM + Focal`
5. `DeepLabV3+ + CBAM + PPM + Focal + MixUp`
6. `DeepLabV3+ + CBAM + PPM + Focal + CutMix`

这样对比关系最清楚，论文里也最好讲。
