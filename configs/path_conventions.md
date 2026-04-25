# Path Conventions

## 总原则

以后尽量不要把真实数据集和大量训练结果直接塞回模型源码目录，而是通过“参数化路径”来引用。

## 推荐外部路径

```text
数据集根目录: D:\SegData
输出根目录:   D:\SegRuns\outputs
日志根目录:   D:\SegRuns\logs
```

如果你想全部放在当前仓库下，也建议统一到：

```text
D:\Code\all\data
D:\Code\all\outputs
D:\Code\all\logs
```

## 推荐参数写法

以你当前改过的 `deeplabv3-plus-pytorch-main` 为例，推荐命令风格：

```powershell
python train.py `
  --dataset-name VOC2_iter1 `
  --vocdevkit-path D:\SegData\grape\VOC2_iter1devkit `
  --save-dir D:\SegRuns\outputs\deeplabv3_plus\grape_voc2_iter1\exp01\weights `
  --log-dir D:\SegRuns\logs\deeplabv3_plus\grape_voc2_iter1\exp01
```

## 统一命名建议

- `dataset root`: 数据集总根目录
- `vocdevkit path`: 当前具体实验数据集
- `save dir`: 权重、检查点目录
- `log dir`: loss、miou、tensorboard、训练日志目录

## 复现实验时的建议

- 不同模型共用同一数据集根目录
- 不同实验只替换 `save_dir` 和 `log_dir`
- 不覆盖旧实验，保持一实验一目录
