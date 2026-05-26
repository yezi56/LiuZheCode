# Path Conventions

## 原则

数据集、训练输出、日志、预测可视化都应当参数化，不要写死到模型源码里。这样换数据集时，只需要改命令或配置，不需要改代码。

## 推荐外部路径

```text
数据集根目录: D:\SegData
输出根目录:   D:\SegRuns\outputs
日志根目录:   D:\SegRuns\logs
```

## 推荐命令风格

以 `deeplabv3plus-grape` 为例：

```powershell
python train.py `
  --dataset-name VOC2_iter1 `
  --vocdevkit-path D:\SegData\grape\VOC2_iter1devkit `
  --save-dir D:\SegRuns\outputs\deeplabv3plus-grape\grape_voc2_iter1\exp01\weights `
  --log-dir D:\SegRuns\logs\deeplabv3plus-grape\grape_voc2_iter1\exp01
```

## 命名建议

- `dataset root`: 数据集总根目录
- `vocdevkit path`: 当前实验使用的 VOC 格式数据集
- `save dir`: 权重与检查点目录
- `log dir`: loss、mIoU、TensorBoard、训练日志目录
- `output dir`: 验证集可视化、预测结果、指标图表目录

## 换数据集时的建议

1. 数据先放到仓库外部，例如 `D:\SegData\new_dataset`。
2. 用独立脚本转换为训练脚本需要的格式。
3. 训练命令只改 `--vocdevkit-path`、`--save-dir`、`--log-dir`。
4. 新数据集稳定后，再把转换脚本整理进对应模型的 `utils` 目录。
