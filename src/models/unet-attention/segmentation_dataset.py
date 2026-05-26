import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """Generic segmentation dataset.

    Expected layout:

    dataset_root/
      train/images
      train/masks
      val/images
      val/masks
    """

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    mask_suffixes = {".png", ".bmp", ".gif", ".tif", ".tiff", ".jpg", ".jpeg"}

    def __init__(self, root, split, transforms=None, image_dir="images", mask_dir="masks"):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.image_root = self.root / split / image_dir
        self.mask_root = self.root / split / mask_dir

        if not self.image_root.exists():
            raise FileNotFoundError(f"Image folder not found: {self.image_root}")
        if not self.mask_root.exists():
            raise FileNotFoundError(f"Mask folder not found: {self.mask_root}")

        self.images = sorted(p for p in self.image_root.iterdir() if p.suffix.lower() in self.image_suffixes)
        if not self.images:
            raise FileNotFoundError(f"No images found in: {self.image_root}")

        self.masks = [self._find_mask(p) for p in self.images]

    def _find_mask(self, image_path):
        for suffix in self.mask_suffixes:
            candidate = self.mask_root / f"{image_path.stem}{suffix}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Mask not found for image: {image_path.name}")

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")
        mask = Image.fromarray(np.asarray(mask, dtype=np.uint8))

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return image, mask

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs
