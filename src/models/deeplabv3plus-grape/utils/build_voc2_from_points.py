import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.voc2_pipeline import build_initial_pseudo_mask, colorize_mask, load_labelme_sample


def parse_args():
    parser = argparse.ArgumentParser(description="Build VOC2 pseudo labels from point annotations.")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--dataset-name", type=str, default="VOC2")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slic-segments", type=int, default=350)
    parser.add_argument("--slic-compactness", type=float, default=12.0)
    parser.add_argument("--min-lesion-area", type=int, default=64)
    parser.add_argument("--max-side", type=int, default=1280)
    return parser.parse_args()


def ensure_dirs(voc_root: Path):
    for directory in [
        voc_root / "JPEGImages",
        voc_root / "SegmentationClass",
        voc_root / "SegmentationClassVis",
        voc_root / "ImageSets" / "Segmentation",
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def clear_dir(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except PermissionError:
            print(f"warning: could not remove {child}, keeping existing file")


def write_split_file(path: Path, items):
    path.write_text("".join(f"{item}\n" for item in items), encoding="utf-8")


def main():
    args = parse_args()
    if not args.source_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {args.source_dir}")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")

    voc_root = ROOT / f"{args.dataset_name}devkit" / "VOC2007"
    jpeg_dir = voc_root / "JPEGImages"
    mask_dir = voc_root / "SegmentationClass"
    vis_dir = voc_root / "SegmentationClassVis"
    split_dir = voc_root / "ImageSets" / "Segmentation"

    ensure_dirs(voc_root)
    clear_dir(jpeg_dir)
    clear_dir(mask_dir)
    clear_dir(vis_dir)
    clear_dir(split_dir)

    stems = []
    json_files = sorted(args.source_dir.glob("*.json"))
    for json_path in json_files:
        sample = load_labelme_sample(json_path)
        if sample is None:
            continue

        mask = build_initial_pseudo_mask(
            sample,
            n_segments=args.slic_segments,
            compactness=args.slic_compactness,
            min_lesion_area=args.min_lesion_area,
            max_side=args.max_side,
        )
        stems.append(sample.stem)

        Image.fromarray(sample.image).save(jpeg_dir / f"{sample.stem}.jpg", quality=95)
        Image.fromarray(mask).save(mask_dir / f"{sample.stem}.png")
        Image.fromarray(colorize_mask(mask)).save(vis_dir / f"{sample.stem}.png")

    stems = sorted(stems)
    random.Random(args.seed).shuffle(stems)
    train_count = int(len(stems) * args.train_ratio)
    train_items = stems[:train_count]
    val_items = stems[train_count:]

    write_split_file(split_dir / "train.txt", train_items)
    write_split_file(split_dir / "val.txt", val_items)

    print(f"dataset_name={args.dataset_name}")
    print(f"json_total={len(json_files)}")
    print(f"labeled_used={len(stems)}")
    print(f"train={len(train_items)}")
    print(f"val={len(val_items)}")
    print(f"voc_root={voc_root}")


if __name__ == "__main__":
    main()
