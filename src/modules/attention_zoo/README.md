# Attention Zoo

This directory collects reusable attention and lightweight feature modules copied from:

```text
D:\Code\seg\unet\module
```

They are kept as a module library for future semantic segmentation experiments, especially ablation studies around:

- channel attention,
- spatial attention,
- lightweight attention,
- plug-and-play feature enhancement,
- possible fusion into DeepLabV3+, U-Net, U-Net++, PSPNet, HRNet, or SegNeXt variants.

The code here is not tied to any specific dataset. Dataset-specific scripts should live in the model or dataset preparation directory for that experiment.

## Current Role

Treat this folder as a reference and reusable component pool. Before adding one module into a training pipeline, prefer copying or wrapping only the specific module needed for that experiment, then document the experiment name in `outputs/<model>/<dataset>/<experiment>`.
