# PPM Reference

Source repository:

- `https://github.com/cheeyeo/PSPNET_tutorial`

Tracked reference file:

- `pyramid_pooling_module.py`

How it relates to the main project:

- The active DeepLabV3+ implementation places PPM **after ASPP**
- The active PPM used for training lives in
  `src/models/deeplabv3-plus-pytorch-main/nets/deeplabv3_plus_dual.py`
- The tracked reference file here keeps a clean PSP-style PPM implementation
  for comparison and ablation documentation
