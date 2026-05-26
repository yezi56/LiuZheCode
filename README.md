# Semantic Segmentation Lab

`D:\Code\all` 是语义分割论文实验总仓库。它的目标不是把所有模型硬合成一个工程，而是把不同模型、模块、训练脚本、实验输出规范地放在同一个实验台里，方便后续做对比实验和消融实验。

## 当前目录

```text
all/
├─ src/
│  ├─ models/
│  │  ├─ deeplabv3plus-grape/
│  │  ├─ hrnet/
│  │  ├─ pspnet/
│  │  ├─ unet-voc/
│  │  ├─ unet-attention/
│  │  ├─ unetpp/
│  │  ├─ segnext/
│  │  ├─ cbam/
│  │  ├─ efficientnet/
│  │  └─ efficientnetv2/
│  └─ modules/
│     ├─ attention_zoo/
│     ├─ shared_attention/
│     └─ third_party/
├─ configs/
├─ docs/
├─ data/
├─ logs/
└─ outputs/
```

## 模型定位

| 目录 | 定位 |
|---|---|
| `src/models/deeplabv3plus-grape` | 当前主实验工程，基于 DeepLabV3+，已接入 CBAM、PPM、Focal Loss、MixUp/CutMix、MobileNetV2 + Lite Swin 双骨干 |
| `src/models/hrnet` | HRNet 分割基线 |
| `src/models/pspnet` | PSPNet 分割基线，也可作为 PPM 对照来源 |
| `src/models/unet-voc` | U-Net VOC 风格分割基线 |
| `src/models/unet-attention` | 从 `D:\Code\seg\unet` 整理出的通用 U-Net 注意力实验分支，已移除旧数据集绑定逻辑 |
| `src/models/unetpp` | 从 `D:\Code\seg\U-Net++` 整理进来的 U-Net++ 参考实现 |
| `src/models/segnext` | SegNeXt 官方实现整理版，可作为现代卷积注意力分割模型对照 |
| `src/models/cbam` | CBAM 注意力机制参考实现 |
| `src/models/efficientnet` | EfficientNet 分类参考，可作为轻量 backbone 储备 |
| `src/models/efficientnetv2` | EfficientNetV2 分类参考，可作为轻量 backbone 储备 |

## 模块库

| 目录 | 定位 |
|---|---|
| `src/modules/attention_zoo` | 从 `D:\Code\seg\unet\module` 整理进来的注意力/轻量模块集合，用于后续消融和可插拔实验 |
| `src/modules/shared_attention` | 当前仓库已有的共享注意力模块 |
| `src/modules/third_party` | 第三方参考实现与论文模块来源 |

## 数据与结果

真实数据集不要放进仓库。建议长期放在仓库外部，例如：

```text
D:\SegData\dataset_name
D:\SegRuns\outputs
D:\SegRuns\logs
```

仓库内的 `data`、`logs`、`outputs` 只保留说明和轻量占位，不提交真实图片、mask、csv、训练日志、预测结果和大批量实验输出。

## 推荐实验命名

模型名直接使用目录名：

```text
deeplabv3plus-grape
hrnet
pspnet
unet-voc
unetpp
```

实验输出建议使用：

```text
outputs/<model_name>/<dataset_name>/<experiment_name>/
logs/<model_name>/<dataset_name>/<experiment_name>/
```

例如：

```text
outputs/deeplabv3plus-grape/grape_voc2_iter1/exp03_cbam_ppm_focal/
logs/deeplabv3plus-grape/grape_voc2_iter1/exp03_cbam_ppm_focal/
```

## 初始化权重状态

| 模型 | 默认期望权重 | 当前状态 |
|---|---|---|
| `deeplabv3plus-grape` | `model_data/deeplab_mobilenetv2.pth` | 已存在，可直接用于 MobileNetV2 初始化 |
| `pspnet` | `model_data/pspnet_mobilenetv2.pth` | 已存在，可直接用于 PSPNet MobileNetV2 初始化 |
| `hrnet` | `model_data/hrnetv2_w18_weights_voc.pth` | 代码期望该文件，当前本地未配齐 |
| `unet-voc` | `model_data/unet_vgg_voc.pth` | 代码期望该文件，当前本地未配齐 |
| `unet-attention` | 无固定默认权重 | 作为可插拔注意力 U-Net 实验分支，默认从头训练或手动指定 checkpoint |
| `segnext` | 上游 README 给出 Tsinghua Cloud 预训练链接 | 当前只保留源码和配置，权重不入库 |
| `cbam` | 无固定分割权重 | 作为注意力模块参考，不是独立分割训练主线 |
| `efficientnet` / `efficientnetv2` | 分类预训练权重 | 作为 backbone 参考，暂不作为当前分割主实验入口 |

## 当前主实验路线

建议论文实验按下面顺序组织：

1. `DeepLabV3+`
2. `DeepLabV3+ + CBAM`
3. `DeepLabV3+ + CBAM + PPM`
4. `DeepLabV3+ + CBAM + PPM + Focal Loss`
5. `DeepLabV3+ + CBAM + PPM + Focal Loss + MixUp`
6. `DeepLabV3+ + CBAM + PPM + Focal Loss + CutMix`
7. `DeepLabV3+ + MobileNetV2/Swin 双骨干` 与轻量化对比
8. `SegNeXt` 作为现代卷积注意力分割模型横向对照

细节文档：

- [工作区结构](D:/Code/all/docs/WORKSPACE_LAYOUT.md)
- [路径约定](D:/Code/all/configs/path_conventions.md)
- [Git/GitHub 命令](D:/Code/all/docs/GIT_GITHUB_COMMANDS.md)
