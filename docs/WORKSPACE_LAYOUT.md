# Workspace Layout

## 目标

这个仓库是语义分割论文实验工作台。整理原则是：

- 模型目录按模型名字命名。
- 数据集放到仓库外部。
- 日志和输出按模型、数据集、实验名分层。
- 第三方模块保留来源说明，方便论文复现和引用。
- 删除样例图片、临时输出、缓存文件和旧数据集。

## 当前结构

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

## 模型目录规则

目录名尽量直接使用模型名：

- `hrnet`
- `pspnet`
- `unetpp`
- `unet-attention`
- `segnext`
- `efficientnet`
- `efficientnetv2`

如果某个工程已经结合了具体任务或改造方向，可以加后缀：

- `deeplabv3plus-grape`
- `unet-voc`

## 数据规则

仓库不保存真实数据集和样例数据。推荐外部路径：

```text
D:\SegData\<dataset_name>
```

每个训练脚本后续都应通过参数传入数据路径，例如：

```powershell
python train.py --vocdevkit-path D:\SegData\grape\VOC2_iter1devkit
```

## 输出规则

输出建议放到外部路径：

```text
D:\SegRuns\outputs\<model_name>\<dataset_name>\<experiment_name>
D:\SegRuns\logs\<model_name>\<dataset_name>\<experiment_name>
```

如果暂时放在仓库内，也按同样结构放在：

```text
outputs/<model_name>/<dataset_name>/<experiment_name>
logs/<model_name>/<dataset_name>/<experiment_name>
```

## 清理规则

这些内容不应进入 Git：

- `__pycache__`
- `.pyc`
- 训练输出目录
- 真实数据集
- 样例图片
- 大批量 mask、预测图、叠加图
- 临时 CSV 指标表

如果某张图确实用于论文或 README 说明，应放到 `docs/assets` 并在提交前确认来源和必要性。
