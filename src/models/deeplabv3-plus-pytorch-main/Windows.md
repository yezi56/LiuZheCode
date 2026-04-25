# GRAPE Windows 11 本地训练命令版

本文面向当前本地工程：

```text
D:\Code\all\deeplabv3-plus-pytorch-main
```

并假设：

1. 你的系统是 Windows 11
2. 你已经安装了 `conda`
3. 你已经有一个可用环境，名字叫：

```text
pytorch
```

4. 你的数据已经整理完成，当前工程里已经有：
   - `VOCdevkit/VOC2007/JPEGImages`
   - `VOCdevkit/VOC2007/SegmentationClass`
   - `VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt`
   - `VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt`

## 1. 当前训练默认配置

现在这套工程已经把主要默认参数写进 [train.py](/D:/Code/all/deeplabv3-plus-pytorch-main/train.py) 了，所以本地训练最简命令就是：

```powershell
python train.py
```

默认配置包括：

1. `backbone = mobilenet`
2. `model_path = model_data/deeplab_mobilenetv2.pth`
3. `num_classes = 3`
4. `input_shape = 512 512`
5. `freeze_epoch = 50`
6. `unfreeze_epoch = 500`
7. `freeze_batch_size = 8`
8. `unfreeze_batch_size = 4`
9. 权重输出目录：

```text
outputs/grape_seg/weights
```

10. 日志输出目录：

```text
outputs/grape_seg/logs
```

## 2. 打开 PowerShell

建议打开：

```text
Windows PowerShell
```

或者：

```text
Anaconda PowerShell Prompt
```

如果你用的是普通 PowerShell，而 `conda activate` 不生效，先执行：

```powershell
conda init powershell
```

然后关闭 PowerShell，重新打开。

## 3. 进入工程目录

```powershell
cd D:\Code\all\deeplabv3-plus-pytorch-main
```

## 4. 激活 conda 环境

```powershell
conda activate pytorch
```

检查 Python：

```powershell
python -V
```

## 5. 检查 PyTorch 和 CUDA

这一步非常重要，先确认环境没问题：

```powershell
python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('gpu_count=', torch.cuda.device_count())"
```

如果输出中：

```text
cuda_available=True
```

说明本地 GPU 训练环境是通的。

## 6. 如缺依赖，安装依赖

如果你这个 `pytorch` 环境里还没装齐依赖，就执行：

```powershell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.timeout 120
pip install tensorboard h5py -i https://pypi.tuna.tsinghua.edu.cn/simple
```

当前这台机器我已经核过的环境版本是：

```text
Python 3.11.14
torch 2.7.0+cu128
torchvision 0.22.0+cu128
numpy 2.2.6
scikit-image 0.25.2
opencv-python 4.12.0
Pillow 12.0.0
```

所以原则是：

1. 不要随便重装 `torch`
2. 只补缺失包
3. 如果确实要重装，就与当前版本保持一致

如果你发现 `scikit-image` 缺失，而你又要跑 `VOC2` 的 superpixel 伪标签流程，再补这一条：

```powershell
pip install scikit-image==0.25.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果连 `torch` 都没装或者版本损坏，再按当前环境兼容版本重装：

```powershell
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 -i https://pypi.tuna.tsinghua.edu.cn/simple -f https://mirrors.aliyun.com/pytorch-wheels/cu128/
```

装完后再检查一次：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 7. 检查数据划分文件

当前你真正需要的是：

```text
VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt
VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt
```

在 PowerShell 中查看：

```powershell
Get-ChildItem .\VOCdevkit\VOC2007\ImageSets\Segmentation
```

## 8. 前台启动训练

如果你想直接在当前窗口看训练过程：

```powershell
python train.py
```

## 9. 后台启动训练

如果你不想一直盯着窗口，当前工程里已经给你加了一个启动脚本 [train_grape.sh](/D:/Code/all/deeplabv3-plus-pytorch-main/train_grape.sh)，但那个是 Linux 用的。

Windows 本地最直接的后台方案建议用：

```powershell
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Code\all\deeplabv3-plus-pytorch-main; conda activate pytorch; python train.py"
```

如果你希望把日志写到文件里，可以用：

```powershell
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Code\all\deeplabv3-plus-pytorch-main; conda activate pytorch; python train.py *> outputs\grape_seg\train_local.log"
```

## 10. 查看输出目录

训练产生的权重会保存在：

[outputs/grape_seg/weights](/D:/Code/all/deeplabv3-plus-pytorch-main/outputs/grape_seg/weights)

日志和 TensorBoard 记录会保存在：

[outputs/grape_seg/logs](/D:/Code/all/deeplabv3-plus-pytorch-main/outputs/grape_seg/logs)

在 PowerShell 里查看：

```powershell
Get-ChildItem .\outputs\grape_seg\weights
Get-ChildItem .\outputs\grape_seg\logs
```

最常用的权重是：

```text
outputs/grape_seg/weights/best_epoch_weights.pth
```

## 11. 训练完成后评估

```powershell
python get_miou.py
```

## 12. 训练完成后预测

```powershell
python predict.py
```

然后输入一张图片路径，例如：

```text
VOCdevkit/VOC2007/JPEGImages/IMG2006_4909.jpg
```

## 13. 如果显存不够

虽然默认参数已经写在 `train.py` 里，但如果你本地显存不够，可以只覆盖少量参数，而不是打一长串。

例如：

```powershell
python train.py --freeze-batch-size 4 --unfreeze-batch-size 2
```

如果你还想把输入图改小一点：

```powershell
python train.py --freeze-batch-size 4 --unfreeze-batch-size 2 --input-shape 448 448
```

## 14. 最推荐的本地执行顺序

你本地最推荐的执行顺序就是：

```powershell
cd D:\Code\all\deeplabv3-plus-pytorch-main
conda activate pytorch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python train.py
```

## 15. 一份真正最短的版本

如果你的 `pytorch` 环境已经可用，那你以后本地训练真的只需要这几行：

```powershell
cd D:\Code\all\deeplabv3-plus-pytorch-main
conda activate pytorch
python train.py
```

## 16. VOC2 伪标签数据集流程

你现在的新数据位于：

```text
D:\Code\all\deeplabv3-plus-pytorch-main\dataset
```

这个目录里的新标注规则是：

1. `leaf` 是叶片 polygon
2. `lesion` 是白粉病正例点
3. `no_lesion` 是叶片内负例点

当前工程已经补成了可选数据集模式，你可以在训练时通过：

```powershell
--dataset-name VOC
--dataset-name VOC2
--dataset-name VOC2_iter1
```

来切换不同数据集，不需要再手改代码路径。

## 17. 第一步：从点标注生成 VOC2 初始伪标签

先进入工程并激活环境：

```powershell
cd D:\Code\all\deeplabv3-plus-pytorch-main
conda activate pytorch
```

然后生成第一版 `VOC2`：

```powershell
python .\utils\build_voc2_from_points.py --source-dir .\dataset --dataset-name VOC2 --train-ratio 0.7 --slic-segments 350 --slic-compactness 12 --min-lesion-area 64 --max-side 1280
```

生成完成后，新数据集会在：

```text
VOC2devkit\VOC2007
```

其中最重要的是：

```text
VOC2devkit\VOC2007\JPEGImages
VOC2devkit\VOC2007\SegmentationClass
VOC2devkit\VOC2007\SegmentationClassVis
VOC2devkit\VOC2007\ImageSets\Segmentation\train.txt
VOC2devkit\VOC2007\ImageSets\Segmentation\val.txt
```

## 18. 第二步：训练第一轮 VOC2 伪标签模型

这一步建议先训练一轮比较充分但不过分激进的 baseline。

推荐命令：

```powershell
python train.py --dataset-name VOC2 --model-path model_data\deeplab_mobilenetv2.pth --freeze-epoch 30 --unfreeze-epoch 240 --freeze-batch-size 4 --unfreeze-batch-size 2 --num-workers 0 --init-lr 0.0035 --dice-loss true --save-period 10 --eval-period 10 --save-dir outputs\voc2_seed\weights --log-dir outputs\voc2_seed\logs
```

这一轮的目标不是一步到位，而是先把点标注传播得到的第一版伪标签学稳。

## 19. 第三步：用第一轮模型生成 VOC2_iter1 精炼伪标签

用第一轮最优权重继续做一次图像引导 refine：

```powershell
python .\utils\refine_voc2_pseudo_labels.py --source-dir .\dataset --input-dataset-name VOC2 --output-dataset-name VOC2_iter1 --model-path .\outputs\voc2_seed\weights\best_epoch_weights.pth --slic-segments 350 --slic-compactness 12 --threshold 0.5 --smooth-alpha 0.65 --smooth-iters 8 --min-lesion-area 64 --max-side 1280
```

生成完成后，第二轮数据集会在：

```text
VOC2_iter1devkit\VOC2007
```

## 20. 第四步：训练第二轮 VOC2_iter1 精炼模型

这是 iterative refinement 的第二轮训练，建议继续使用第一轮最优权重作为起点。

推荐命令：

```powershell
python train.py --dataset-name VOC2_iter1 --model-path outputs\voc2_seed\weights\best_epoch_weights.pth --freeze-epoch 20 --unfreeze-epoch 180 --freeze-batch-size 4 --unfreeze-batch-size 2 --num-workers 0 --init-lr 0.0025 --dice-loss true --save-period 10 --eval-period 10 --save-dir outputs\voc2_iter1\weights --log-dir outputs\voc2_iter1\logs
```

## 23. DeepLabV3+ + CBAM 说明

当前这份工程已经将 `CBAM` 正式接入 `DeepLabV3+`，并且训练默认参数已经切到：

```text
attention_type = cbam
```

当前 CBAM 注入位置包括：

1. backbone low-level feature
2. backbone high-level feature
3. ASPP output
4. decoder fusion output

如果你直接运行：

```powershell
python train.py
```

现在默认就是：

```text
DeepLabV3+ + CBAM
```

如果你想显式指定，也可以这样写：

```powershell
python train.py --attention-type cbam
```

如果你想回退成原始不带注意力的 V3+ baseline，可以这样写：

```powershell
python train.py --attention-type none
```

## 24. CBAM 版推荐命令

### 训练 baseline CBAM 版

```powershell
python train.py --attention-type cbam
```

### 训练 VOC2 第一轮 CBAM 版

```powershell
python train.py --dataset-name VOC2 --attention-type cbam --model-path model_data\deeplab_mobilenetv2.pth --freeze-epoch 30 --unfreeze-epoch 240 --freeze-batch-size 4 --unfreeze-batch-size 2 --num-workers 0 --init-lr 0.0035 --dice-loss true --save-period 10 --eval-period 10 --save-dir outputs\voc2_seed_cbam\weights --log-dir outputs\voc2_seed_cbam\logs
```

### 训练 VOC2_iter1 第二轮 CBAM 版

```powershell
python train.py --dataset-name VOC2_iter1 --attention-type cbam --model-path outputs\voc2_seed_cbam\weights\best_epoch_weights.pth --freeze-epoch 20 --unfreeze-epoch 180 --freeze-batch-size 4 --unfreeze-batch-size 2 --num-workers 0 --init-lr 0.0025 --dice-loss true --save-period 10 --eval-period 10 --save-dir outputs\voc2_iter1_cbam\weights --log-dir outputs\voc2_iter1_cbam\logs
```

## 25. 推理与评估兼容说明

推理端现在支持自动识别权重是否带注意力：

1. 如果权重里包含 `attention_low / attention_high / attention_aspp / attention_decoder` 这些参数，就自动按 `cbam` 加载。
2. 如果权重是旧版无注意力模型，就自动按普通 `DeepLabV3+` 加载。

这意味着：

- 旧权重还能继续看结果
- 新训练出来的 `CBAM` 权重也能直接用于 `predict.py`、`get_miou.py` 和验证集可视化脚本

这一步的目标是让模型在更干净的 `VOC2_iter1` 上进一步收敛。

## 21. 第五步：评估第二轮结果

如果你要评估第二轮 refined 模型：

```powershell
python get_miou.py --dataset-name VOC2_iter1 --model-path outputs\voc2_iter1\weights\best_epoch_weights.pth
```

如果你要导出验证集叠加图：

```powershell
python .\utils\export_val_visualizations.py --vocdevkit-path .\VOC2_iter1devkit --model-path .\outputs\voc2_iter1\weights\best_epoch_weights.pth
```

## 22. 这一套 VOC2 流程你真正需要记住的顺序

```powershell
cd D:\Code\all\deeplabv3-plus-pytorch-main
conda activate pytorch

python .\utils\build_voc2_from_points.py --source-dir .\dataset --dataset-name VOC2 --train-ratio 0.7 --slic-segments 350 --slic-compactness 12 --min-lesion-area 64 --max-side 1280

python train.py --dataset-name VOC2 --model-path model_data\deeplab_mobilenetv2.pth --freeze-epoch 30 --unfreeze-epoch 240 --freeze-batch-size 4 --unfreeze-batch-size 2 --num-workers 0 --init-lr 0.0035 --dice-loss true --save-period 10 --eval-period 10 --save-dir outputs\voc2_seed\weights --log-dir outputs\voc2_seed\logs

python .\utils\refine_voc2_pseudo_labels.py --source-dir .\dataset --input-dataset-name VOC2 --output-dataset-name VOC2_iter1 --model-path .\outputs\voc2_seed\weights\best_epoch_weights.pth --slic-segments 350 --slic-compactness 12 --threshold 0.5 --smooth-alpha 0.65 --smooth-iters 8 --min-lesion-area 64 --max-side 1280

python train.py --dataset-name VOC2_iter1 --model-path outputs\voc2_seed\weights\best_epoch_weights.pth --freeze-epoch 20 --unfreeze-epoch 180 --freeze-batch-size 4 --unfreeze-batch-size 2 --num-workers 0 --init-lr 0.0025 --dice-loss true --save-period 10 --eval-period 10 --save-dir outputs\voc2_iter1\weights --log-dir outputs\voc2_iter1\logs
```
