# Third-Party Reference Modules

This directory stores small, versioned reference implementations that are
useful for semantic segmentation experiments in this workspace.

Current references:

- `focal_loss_reference`
  - Source repository:
    `https://github.com/cloudpark93/pytorch-multi-class-segmentation-focal-loss`
  - Used to align the focal-loss formulation in
    `src/models/deeplabv3-plus-pytorch-main/nets/deeplabv3_training.py`
- `ppm_reference`
  - Source repository:
    `https://github.com/cheeyeo/PSPNET_tutorial`
  - Used to align the PPM design in
    `src/models/deeplabv3-plus-pytorch-main/nets/deeplabv3_plus_dual.py`

Notes:

- Raw clone folders downloaded during integration are intentionally ignored by
  the top-level `.gitignore` to keep the main repository clean.
- The tracked reference files here are the minimal pieces needed for paper
  reproduction, comparison, and code provenance.
