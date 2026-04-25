from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
MASK_DIR = ROOT / "VOCdevkit" / "VOC2007" / "SegmentationClass"
VIS_DIR = ROOT / "VOCdevkit" / "VOC2007" / "SegmentationClassVis"

# 0=background, 1=leaf, 2=lesion
PALETTE = np.array(
    [
        [0, 0, 0],
        [60, 180, 75],
        [230, 25, 75],
    ],
    dtype=np.uint8,
)


def main():
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for mask_path in sorted(MASK_DIR.glob("*.png")):
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        if mask.ndim != 2:
            raise ValueError(f"{mask_path.name} is not a single-channel mask.")

        if np.max(mask) >= len(PALETTE):
            raise ValueError(f"{mask_path.name} contains label id {int(np.max(mask))}, beyond the palette range.")

        vis = PALETTE[mask]
        Image.fromarray(vis, mode="RGB").save(VIS_DIR / mask_path.name)
        count += 1

    print(f"visualized_masks={count}")
    print(f"output_dir={VIS_DIR}")


if __name__ == "__main__":
    main()
