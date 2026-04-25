# Semantic Segmentation Lab

`D:\Code\all` 现在作为一个统一的语义分割论文实验总仓库使用，目标是尽可能集中管理：

- 多种语义分割模型
- 可复用注意力模块
- 统一的数据、日志、输出和文档规范

## 推荐打开方式

优先在 VS Code 中打开：

- [semantic-segmentation-lab.code-workspace](/D:/Code/all/semantic-segmentation-lab.code-workspace)

这样看到的是整理后的实验工作区视图，比直接打开根目录更适合做论文实验和多模型对比。

## 当前目录约定

- [src](/D:/Code/all/src)：统一源码入口
- [configs](/D:/Code/all/configs)：路径与实验配置说明
- [docs](/D:/Code/all/docs)：工作台文档
- [data](/D:/Code/all/data)：数据集入口说明，建议外置
- [logs](/D:/Code/all/logs)：统一日志归档
- [outputs](/D:/Code/all/outputs)：统一实验输出归档

## 当前实际保留的模型与模块

### 模型

- [deeplabv3-plus-pytorch-main](/D:/Code/all/src/models/deeplabv3-plus-pytorch-main)
- [hrnet-pytorch-main](/D:/Code/all/src/models/hrnet-pytorch-main)
- [pspnet-pytorch-master](/D:/Code/all/src/models/pspnet-pytorch-master)
- [unet-pytorch-main](/D:/Code/all/src/models/unet-pytorch-main)
- [CBAM.PyTorch-master](/D:/Code/all/src/models/CBAM.PyTorch-master)

### 共享模块

- [shared_attention](/D:/Code/all/src/modules/shared_attention)

说明：

- 之前提到过的 `DeepLabV3Plus-Pytorch-master` 当前已经不在现有工作区中，不作为当前总仓库的实际模型目录。

## 各模型初始化权重现状

下面这张表是按当前代码仓库里的真实文件状态加上各工程默认代码配置整理的。

| 模型 | 代码默认期望文件 | 当前是否实际存在 | 当前本地文件 | 说明 |
|---|---|---|---|---|
| `deeplabv3-plus-pytorch-main` | `model_data/deeplab_mobilenetv2.pth` | 有 | [model_data/deeplab_mobilenetv2.pth](/D:/Code/all/src/models/deeplabv3-plus-pytorch-main/model_data/deeplab_mobilenetv2.pth) | 当前 V3+ 默认加载的本地初始化权重，主干是 `mobilenetv2` |
| `hrnet-pytorch-main` | `model_data/hrnetv2_w18_weights_voc.pth` | 没有 | [model_data/README.md](/D:/Code/all/src/models/hrnet-pytorch-main/model_data/README.md) | 代码默认想加载 `hrnetv2_w18_weights_voc.pth`，但当前仓库里只有占位说明，没有实际 `.pth` |
| `pspnet-pytorch-master` | `model_data/pspnet_mobilenetv2.pth` | 有 | [model_data/pspnet_mobilenetv2.pth](/D:/Code/all/src/models/pspnet-pytorch-master/model_data/pspnet_mobilenetv2.pth) | PSPNet 当前可直接使用的本地初始化权重，主干是 `mobilenetv2` |
| `unet-pytorch-main` | `model_data/unet_vgg_voc.pth` | 没有 | [model_data/README.md](/D:/Code/all/src/models/unet-pytorch-main/model_data/README.md) | 代码默认想加载 `unet_vgg_voc.pth`，但当前仓库里只有占位说明，没有实际 `.pth` |
| `CBAM.PyTorch-master` | 无固定分割权重文件 | 没有 | 无 | 这是注意力机制实现仓库，不是完整分割工程；训练脚本里直接构建 `resnet50/resnet101` 或 `resnet50_cbam`，默认 `pretrained=False` |

### 目前可以直接用于初始化的模型权重

- `DeepLabV3+`：`deeplab_mobilenetv2.pth`
- `PSPNet`：`pspnet_mobilenetv2.pth`

### 目前代码里写了默认路径，但本地并没有文件的模型

- `HRNet`：默认写的是 `hrnetv2_w18_weights_voc.pth`
- `UNet`：默认写的是 `unet_vgg_voc.pth`

### 目前没有独立初始化权重仓库属性的模块

- `CBAM`：模块仓库，本身不提供分割模型初始化权重

### 当前结论

- 现在真正能开箱即用的初始化权重只有两份：`DeepLabV3+` 和 `PSPNet`
- `HRNet`、`UNet` 的代码里虽然已经写了默认权重文件名，但本地仓库目前没有对应文件
- `CBAM` 当前在总仓库里承担的是“注意力模块/插件”角色，不单独作为一个带现成初始化权重的分割工程使用

## 数据与输出路径建议

你的原始数据集已经迁到仓库外部了，这是对的。后续建议统一遵循：

- 数据集根目录放在仓库外部，例如 `D:\SegData\...`
- 训练输出统一放到仓库外部或 [outputs](/D:/Code/all/outputs) 下
- 训练日志统一放到仓库外部或 [logs](/D:/Code/all/logs) 下

更具体的命名约定见：

- [WORKSPACE_LAYOUT.md](/D:/Code/all/docs/WORKSPACE_LAYOUT.md)
- [path_conventions.md](/D:/Code/all/configs/path_conventions.md)

## logs / outputs 当前归档结构

我已经在 [logs](/D:/Code/all/logs) 和 [outputs](/D:/Code/all/outputs) 下给主要模型建好了统一目录：

- `deeplabv3_plus`
- `hrnet`
- `pspnet`
- `unet`
- `cbam`

每个模型下又预留了：

- `baseline`
- `grape_voc2`
- `grape_voc2_iter1`
- `ablation`

例如：

- [logs/deeplabv3_plus/grape_voc2](/D:/Code/all/logs/deeplabv3_plus/grape_voc2)
- [outputs/deeplabv3_plus/grape_voc2_iter1](/D:/Code/all/outputs/deeplabv3_plus/grape_voc2_iter1)

## 当前 DeepLabV3+ 已接入的实验扩展

当前 [deeplabv3-plus-pytorch-main](/D:/Code/all/src/models/deeplabv3-plus-pytorch-main) 已经接入：

- `CBAM`
- `PPM after ASPP`
- `Focal Loss`
- `MixUp / CutMix`

详细操作与训练命令见：

- [DeepLabV3+ README](/D:/Code/all/src/models/deeplabv3-plus-pytorch-main/README.md)
- [Windows.md](/D:/Code/all/src/models/deeplabv3-plus-pytorch-main/Windows.md)

## 当前最推荐的论文实验路线

建议按这个顺序做对比：

1. `DeepLabV3+`
2. `DeepLabV3+ + CBAM`
3. `DeepLabV3+ + CBAM + PPM`
4. `DeepLabV3+ + CBAM + PPM + Focal`
5. `DeepLabV3+ + CBAM + PPM + Focal + MixUp`
6. `DeepLabV3+ + CBAM + PPM + Focal + CutMix`

这样实验路线最清楚，后面写论文也最好讲。
