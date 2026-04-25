# Semantic Segmentation Lab

`D:\Code\all` 现在作为一个统一的语义分割实验总仓库使用，目标是集中管理：

- 多种分割模型
- 可复用注意力模块
- 统一的文档、配置、日志和输出约定

## 推荐使用方式

优先在 VS Code 中打开：

- [semantic-segmentation-lab.code-workspace](/D:/Code/all/semantic-segmentation-lab.code-workspace)

这样你会看到一个按“模型源码 / 模块 / 文档 / 配置 / 数据 / 输出”整理过的工作区视图，比直接打开根目录更适合论文实验。

## 目录约定

- [src](/D:/Code/all/src)：统一源码入口
- [configs](/D:/Code/all/configs)：路径与实验配置说明
- [docs](/D:/Code/all/docs)：工作台文档
- [data](/D:/Code/all/data)：数据集入口说明，建议外置
- [logs](/D:/Code/all/logs)：统一日志根目录
- [outputs](/D:/Code/all/outputs)：统一实验输出根目录

## 当前模型与模块

- `deeplabv3-plus-pytorch-main`
- `DeepLabV3Plus-Pytorch-master`
- `hrnet-pytorch-main`
- `pspnet-pytorch-master`
- `unet-pytorch-main`
- `CBAM.PyTorch-master`
- `shared_attention`

## 数据与输出建议

数据集你已经迁到其他路径了，后续建议统一遵循：

- 数据集根目录放在仓库外部，例如 `D:\SegData\...`
- 训练输出统一放到仓库外部或 [outputs](/D:/Code/all/outputs) 下
- 训练日志统一放到仓库外部或 [logs](/D:/Code/all/logs) 下

更具体的命名约定见：

- [WORKSPACE_LAYOUT.md](/D:/Code/all/docs/WORKSPACE_LAYOUT.md)
- [path_conventions.md](/D:/Code/all/configs/path_conventions.md)
