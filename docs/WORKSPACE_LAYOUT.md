# Workspace Layout

## 目标

这个总仓库不是单一项目，而是一个“论文实验工作台”。整理重点不是把所有代码强行揉成一个工程，而是做到：

- 模型来源清楚
- 模块复用清楚
- 数据路径可替换
- 输出路径可扩展
- 日志与结果更容易归档

## 建议逻辑结构

```text
all/
├─ src/
│  ├─ models/
│  │  ├─ deeplabv3-plus-pytorch-main/
│  │  ├─ DeepLabV3Plus-Pytorch-master/
│  │  ├─ hrnet-pytorch-main/
│  │  ├─ pspnet-pytorch-master/
│  │  ├─ unet-pytorch-main/
│  │  └─ CBAM.PyTorch-master/
│  └─ modules/
│     └─ shared_attention/
├─ configs/
├─ docs/
├─ data/
├─ logs/
└─ outputs/
```

## 实际使用建议

### 1. 模型源码

每个模型保持各自内部结构不变，避免为了“好看”而破坏原有训练脚本。

### 2. 数据集

数据集建议长期放在仓库外部，例如：

```text
D:\SegData\grape\VOC2devkit
D:\SegData\grape\VOC2_iter1devkit
```

### 3. 输出目录

建议统一实验输出结构：

```text
outputs/
└─ deeplabv3_plus/
   └─ grape/
      └─ exp_2026_04_25_iter1/
         ├─ weights/
         ├─ val_vis/
         └─ metrics/
```

### 4. 日志目录

建议统一日志结构：

```text
logs/
└─ deeplabv3_plus/
   └─ grape/
      └─ exp_2026_04_25_iter1/
```

## 适合论文实验的命名方式

- 模型名：`deeplabv3_plus` / `pspnet` / `hrnet` / `unet`
- 数据集名：`grape_voc2` / `grape_voc2_iter1`
- 实验名：`exp01_baseline` / `exp02_cbam` / `exp03_shared_attention`

推荐组合示例：

```text
outputs/deeplabv3_plus/grape_voc2/exp02_cbam
logs/deeplabv3_plus/grape_voc2/exp02_cbam
```
