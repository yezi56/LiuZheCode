# Focal Loss Reference

Source repository:

- `https://github.com/cloudpark93/pytorch-multi-class-segmentation-focal-loss`

Tracked reference file:

- `focal_loss.py`

How it relates to the main project:

- The active loss used by DeepLabV3+ lives in
  `src/models/deeplabv3-plus-pytorch-main/nets/deeplabv3_training.py`
- That implementation extends the basic reference in two ways:
  - it supports the segmentation ignore label used in this repository
  - it supports both hard labels and soft labels for MixUp/CutMix training
