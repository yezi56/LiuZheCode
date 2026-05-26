import argparse
import csv
import random
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.grape_mildew_weak_pipeline import (  # noqa: E402
    build_initial_pseudo_mask,
    colorize_mask,
    ensure_empty_dir,
    load_labelme_sample,
    mask_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a conservative VOC dataset from grape mildew point labels.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, default="VOC_GRAPE_MILDEW_SEED")
    parser.add_argument("--train-ratio", type=float, default=0.82)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slic-segments", type=int, default=900)
    parser.add_argument("--slic-compactness", type=float, default=8.0)
    parser.add_argument("--lesion-threshold", type=float, default=0.68)
    parser.add_argument("--ignore-threshold", type=float, default=0.56)
    parser.add_argument("--color-weight", type=float, default=0.25)
    parser.add_argument("--min-lesion-area", type=int, default=48)
    parser.add_argument("--seed-radius", type=int, default=5)
    parser.add_argument("--ignore-ring", type=int, default=2)
    parser.add_argument("--max-side", type=int, default=1024)
    return parser.parse_args()


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
    report_path = voc_root / "pseudo_label_report.csv"

    for directory in [jpeg_dir, mask_dir, vis_dir, split_dir]:
        ensure_empty_dir(directory)

    stems = []
    rows = []
    json_files = sorted(args.source_dir.glob("*.json"))
    for index, json_path in enumerate(json_files, start=1):
        sample = load_labelme_sample(json_path)
        if sample is None:
            continue
        mask = build_initial_pseudo_mask(
            sample,
            n_segments=args.slic_segments,
            compactness=args.slic_compactness,
            lesion_threshold=args.lesion_threshold,
            ignore_threshold=args.ignore_threshold,
            color_weight=args.color_weight,
            min_lesion_area=args.min_lesion_area,
            seed_radius=args.seed_radius,
            ignore_ring=args.ignore_ring,
            max_side=args.max_side,
        )
        stems.append(sample.stem)
        stats = mask_stats(mask)
        rows.append({"stem": sample.stem, **stats})

        Image.fromarray(sample.image).save(jpeg_dir / f"{sample.stem}.jpg", quality=95)
        Image.fromarray(mask).save(mask_dir / f"{sample.stem}.png")
        Image.fromarray(colorize_mask(mask)).save(vis_dir / f"{sample.stem}.png")

        if index % 25 == 0:
            print(f"processed={index}/{len(json_files)} used={len(stems)}", flush=True)

    stems = sorted(stems)
    random.Random(args.seed).shuffle(stems)
    train_count = int(len(stems) * args.train_ratio)
    train_items = stems[:train_count]
    val_items = stems[train_count:]

    write_split_file(split_dir / "train.txt", train_items)
    write_split_file(split_dir / "val.txt", val_items)

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "background", "leaf", "lesion", "ignore"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"dataset_name={args.dataset_name}")
    print(f"json_total={len(json_files)}")
    print(f"labeled_used={len(stems)}")
    print(f"train={len(train_items)}")
    print(f"val={len(val_items)}")
    print(f"voc_root={voc_root}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
