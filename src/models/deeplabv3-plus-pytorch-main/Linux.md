# GRAPE 服务器从零训练命令版

本文面向当前工程：

```text
D:\Code\all\deeplabv3-plus-pytorch-main
```

目标是给出一套可以直接复制执行的服务器训练流程，从服务器没有 `conda` 开始，到安装 Python 环境、上传工程、安装 PyTorch、启动训练为止。

## 1. 先回答 backbone 问题

当前这套工程里，默认推荐的主干仍然是：

```python
backbone = "mobilenet"
```

它的判断是：

1. `mobilenetv2` 从时间上看确实偏老。
2. 但它在这套仓库里依然是合理的第一轮基线。
3. 它的优势是显存占用更小、训练更稳、迁移到云服务器更省事。
4. 对你现在这种“先把叶片和病斑稳定区分开”的目标来说，优先级高于追求最新骨干。

我的建议是：

1. 第一轮正式训练先用 `mobilenet` 跑通。
2. 如果结果稳定、服务器显存足够，再做 `xception` 对照实验。
3. 不建议第一次上云就同时改更新骨干和训练流程。

当前这套仓库原生主要支持：

1. `mobilenet`
2. `xception`

## 2. 当前工程中要保留的预训练权重

请保留：

```text
model_data/deeplab_mobilenetv2.pth
```

它是当前仓库用于 `mobilenet` 主干的预训练权重。虽然你的任务是三分类，但 `train.py` 会按键名和形状加载，骨干部分仍然可以吃到预训练，分类头不匹配的部分会自动跳过。

## 3. 当前工程中训练输出的规范位置

当前工程已经统一成：

1. 权重输出目录：

```text
outputs/grape_seg/weights
```

2. 日志输出目录：

```text
outputs/grape_seg/logs
```

3. 推理默认加载最优权重：

```text
outputs/grape_seg/weights/best_epoch_weights.pth
```

如果以后你想改保存位置，训练时直接改：

```bash
--save-dir <新的权重目录>
--log-dir <新的日志目录>
```

## 4. 本地 Windows：打包并上传工程

在本地 Windows PowerShell 中执行：

```powershell
cd D:\Code\all

Compress-Archive `
  -Path .\deeplabv3-plus-pytorch-main\* `
  -DestinationPath .\deeplabv3-plus-pytorch-main.zip `
  -Force
```

然后上传到服务器的 `/root/`：

```powershell
scp .\deeplabv3-plus-pytorch-main.zip root@<你的服务器IP>:/root/
```

如果服务器 SSH 端口不是 22：

```powershell
scp -P <端口> .\deeplabv3-plus-pytorch-main.zip <用户名>@<你的服务器IP>:/root/
```

结合你当前服务器截图，压缩包放在 `/root/` 是最符合现状的，后面再从 `/root/` 解压到 `/root/autodl-tmp/`。

## 5. 登录服务器

```bash
ssh root@<你的服务器IP>
```

如果有端口：

```bash
ssh -p <端口> <用户名>@<你的服务器IP>
```

## 6. 服务器没有 conda 时，先用国内镜像安装 Miniconda

下面这套命令可以从零安装：

```bash
cd /root

wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3

source /root/miniconda3/etc/profile.d/conda.sh
conda init bash
source ~/.bashrc
source /root/miniconda3/etc/profile.d/conda.sh
```

检查 conda 是否安装成功：

```bash
conda --version
```

如果你的服务器已经和截图一样存在：

```text
/root/miniconda3
```

那么这一节可以直接跳过。

## 7. 创建 Python 训练环境

```bash
conda config --remove-key channels
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/nvidia
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --set show_channel_urls yes

conda create -n grape_seg python=3.10 -y
conda activate grape_seg

python -V
```

如果你已经像截图里那样进入了：

```text
(grape_seg) root@...
```

那么说明环境已经建好，这一节也可以直接跳过。

以后如果你重新登录服务器，先执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate grape_seg
```

然后把 `pip` 也切到国内镜像：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.timeout 120
```

## 8. 按你这台服务器的实际路径解压工程

你当前服务器的现实情况是：

1. 当前用户目录是 `/root`
2. `conda` 安装在 `/root/miniconda3`
3. 工作盘目录里已经有 `/root/autodl-tmp`
4. 你上传的压缩包大概率在 `/root/deeplabv3-plus-pytorch-main (2).zip`

建议先把压缩包改成一个不带空格和括号的名字，再解压。

```bash
cd /root

mv "/root/deeplabv3-plus-pytorch-main (2).zip" /root/deeplabv3-plus-pytorch-main.zip

mkdir -p /root/autodl-tmp
rm -rf /root/autodl-tmp/deeplabv3-plus-pytorch-main
mkdir -p /root/autodl-tmp/deeplabv3-plus-pytorch-main

unzip -oq /root/deeplabv3-plus-pytorch-main.zip -d /root/autodl-tmp/deeplabv3-plus-pytorch-main

cd /root/autodl-tmp/deeplabv3-plus-pytorch-main
```

如果你上传后的文件名本来就已经是：

```text
/root/deeplabv3-plus-pytorch-main.zip
```

那就把上面那条 `mv` 跳过即可。

检查关键文件：

```bash
ls
ls model_data
ls VOCdevkit/VOC2007/ImageSets/Segmentation
```

你应该能看到：

1. `model_data/deeplab_mobilenetv2.pth`
2. `train.py`
3. `VOCdevkit/VOC2007/JPEGImages`
4. `VOCdevkit/VOC2007/SegmentationClass`
5. `VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt`
6. `VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt`

## 9. 检查服务器 GPU

```bash
nvidia-smi
```

## 10. 用国内镜像安装 PyTorch 和依赖

先升级基础工具：

```bash
python -m pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
```

注意：你这台服务器上，`conda` 国内镜像里不一定有 `pytorch==2.7.1`、`pytorch-cuda=12.8` 这些新包，所以不要再用上一版文档里的 `conda install pytorch==...`。

对你当前环境，更稳的方式是：

1. 普通 Python 依赖走清华 `pip` 镜像
2. PyTorch 三件套走阿里云的 `pytorch-wheels/cu128` 镜像
3. 如果 `pip install ... -f ...` 还是慢，先 `wget -c` 断点续传 wheel，再本地安装

你当前环境是：

1. Linux
2. x86_64
3. Python 3.10

所以直接执行下面这条：

```bash
pip install \
  torch==2.7.1+cu128 \
  torchvision==0.22.1+cu128 \
  torchaudio==2.7.1+cu128 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -f https://mirrors.aliyun.com/pytorch-wheels/cu128/
```

如果你现在网络很慢，推荐直接走“先下载再安装”的版本：

```bash
mkdir -p /root/autodl-tmp/wheels
cd /root/autodl-tmp/wheels

wget -c "https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.7.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl"
wget -c "https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.22.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl"
wget -c "https://mirrors.aliyun.com/pytorch-wheels/cu128/torchaudio-2.7.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl"

pip install /root/autodl-tmp/wheels/torch-2.7.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl
pip install /root/autodl-tmp/wheels/torchvision-0.22.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl
pip install /root/autodl-tmp/wheels/torchaudio-2.7.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl
```

安装其他依赖：

```bash
pip install tensorboard scipy matplotlib opencv-python tqdm pillow h5py -i https://pypi.tuna.tsinghua.edu.cn/simple
```

检查 PyTorch 是否能识别 GPU：

```bash
python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('gpu_count=', torch.cuda.device_count())"
```

如果输出中：

```text
cuda_available=True
```

说明训练环境已经基本就绪。

如果上面这条仍然不稳，就改成“直接指定 wheel 文件”的最强硬版本：

```bash
pip install \
  https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.7.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl \
  https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.22.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl \
  https://mirrors.aliyun.com/pytorch-wheels/cu128/torchaudio-2.7.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

这种写法最适合你现在这种“镜像里有文件，但解析索引或依赖时容易卡住”的情况。

## 11. 创建训练输出目录

```bash
cd /root/autodl-tmp/deeplabv3-plus-pytorch-main

mkdir -p outputs/grape_seg/weights
mkdir -p outputs/grape_seg/logs
```

## 12. 正式启动训练

现在这套工程我已经帮你把默认参数写进 `train.py` 了，所以在服务器上不需要再手敲一长串参数。

最简单的启动方式有两种。

方式一：直接前台启动

```bash
cd /root/autodl-tmp/deeplabv3-plus-pytorch-main
python train.py
```

方式二：后台启动，推荐服务器实际使用

```bash
cd /root/autodl-tmp/deeplabv3-plus-pytorch-main
nohup python train.py > outputs/grape_seg/train.log 2>&1 &
```

如果你想用我已经准备好的启动脚本，也可以：

```bash
cd /root/autodl-tmp/deeplabv3-plus-pytorch-main
bash train_grape.sh
```

当前默认参数等价于下面这套配置：

```bash
num_classes = 3
backbone = mobilenet
model_path = model_data/deeplab_mobilenetv2.pth
vocdevkit_path = VOCdevkit
input_shape = 512 512
freeze_train = true
freeze_epoch = 50
unfreeze_epoch = 100
freeze_batch_size = 8
unfreeze_batch_size = 4
optimizer_type = sgd
init_lr = 0.007
save_dir = outputs/grape_seg/weights
log_dir = outputs/grape_seg/logs
eval_flag = true
eval_period = 5
```

实时查看日志：

```bash
tail -f outputs/grape_seg/train.log
```

## 13. 显存不够时的降配版本

如果显存不够，优先去 `train.py` 里把默认 batch size 改小，再重新启动。你现在不需要再拼长命令。

建议改成：

```bash
freeze_batch_size = 4
unfreeze_batch_size = 2
```

然后重新运行：

```bash
cd /root/autodl-tmp/deeplabv3-plus-pytorch-main
nohup python train.py > outputs/grape_seg/train.log 2>&1 &
```

## 14. 训练完成后查看权重

```bash
cd /root/autodl-tmp/deeplabv3-plus-pytorch-main
ls -lh outputs/grape_seg/weights
```

最常用的是：

```text
outputs/grape_seg/weights/best_epoch_weights.pth
```

## 15. 训练完成后评估

```bash
cd /root/autodl-tmp/deeplabv3-plus-pytorch-main
python get_miou.py
```

## 16. 训练完成后推理

```bash
cd /root/autodl-tmp/deeplabv3-plus-pytorch-main
python predict.py
```

输入一张图，例如：

```text
VOCdevkit/VOC2007/JPEGImages/IMG2006_4909.jpg
```

## 17. 你真正需要记住的几个位置

1. 预训练权重：

```text
model_data/deeplab_mobilenetv2.pth
```

2. 训练权重输出：

```text
outputs/grape_seg/weights
```

3. 训练日志输出：

```text
outputs/grape_seg/logs
```

4. 训练集划分：

```text
VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt
```

5. 验证集划分：

```text
VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt
```

## 18. 推荐实际执行顺序

你在服务器上的操作链可以理解成：

1. 安装 Miniconda
2. 创建 `grape_seg` Python 环境
3. 上传并解压项目
4. 安装 PyTorch 和依赖
5. 检查 `torch.cuda.is_available()`
6. 启动训练
7. 查看日志
8. 看 `best_epoch_weights.pth`
9. 跑 `get_miou.py`
10. 跑 `predict.py`
