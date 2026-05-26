import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deeplab import DeeplabV3  # noqa: E402
from utils.grape_mildew_weak_pipeline import (  # noqa: E402
    LabelmeSample,
    colorize_mask,
    ensure_empty_dir,
    load_labelme_sample,
    refine_mask_with_probabilities,
)
from utils.utils import cvtColor, preprocess_input, resize_image  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Refine grape mildew weak pseudo labels with model probabilities.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--input-dataset-name", type=str, default="VOC_GRAPE_MILDEW_SEED")
    parser.add_argument("--output-dataset-name", type=str, default="VOC_GRAPE_MILDEW_ITER1")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--slic-segments", type=int, default=900)
    parser.add_argument("--slic-compactness", type=float, default=8.0)
    parser.add_argument("--threshold", type=float, default=0.68)
    parser.add_argument("--ignore-threshold", type=float, default=0.48)
    parser.add_argument("--smooth-alpha", type=float, default=0.70)
    parser.add_argument("--smooth-iters", type=int, default=8)
    parser.add_argument("--min-lesion-area", type=int, default=48)
    parser.add_argument("--seed-radius", type=int, default=5)
    parser.add_argument("--ignore-ring", type=int, default=2)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument(
        "--save-resized",
        action="store_true",
        default=False,
        help="Refine and save resized images/masks at --max-side instead of full-resolution masks.",
    )
    return parser.parse_args()


def copy_tree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def predict_probabilities(model: DeeplabV3, image: Image.Image) -> np.ndarray:
    image = cvtColor(image)
    original_h = np.array(image).shape[0]
    original_w = np.array(image).shape[1]
    image_data, nw, nh = resize_image(image, (model.input_shape[1], model.input_shape[0]))
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

    with torch.no_grad():
        images = torch.from_numpy(image_data)
        if model.cuda:
            images = images.cuda()
        pr = model.net(images)[0]
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
        pr = pr[
            int((model.input_shape[0] - nh) // 2) : int((model.input_shape[0] - nh) // 2 + nh),
            int((model.input_shape[1] - nw) // 2) : int((model.input_shape[1] - nw) // 2 + nw),
        ]
        pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    return pr


def resize_sample(sample: LabelmeSample, max_side: int) -> LabelmeSample:
    height, width = sample.image.shape[:2]
    scale = min(1.0, float(max_side) / float(max(height, width)))
    if scale >= 1.0:
        return sample

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    image = cv2.resize(sample.image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    leaf_mask = cv2.resize(sample.leaf_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    seed_points = []
    for x, y, cls_id in sample.seed_points:
        sx = int(np.clip(round(x * scale), 0, new_width - 1))
        sy = int(np.clip(round(y * scale), 0, new_height - 1))
        seed_points.append((sx, sy, cls_id))
    return LabelmeSample(
        stem=sample.stem,
        image_path=sample.image_path,
        image=image,
        leaf_mask=leaf_mask,
        seed_points=seed_points,
    )


def main():
    args = parse_args()
    input_root = ROOT / f"{args.input_dataset_name}devkit" / "VOC2007"
    output_root = ROOT / f"{args.output_dataset_name}devkit" / "VOC2007"
    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_root}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    input_jpeg_dir = input_root / "JPEGImages"
    input_split_dir = input_root / "ImageSets" / "Segmentation"
    output_jpeg_dir = output_root / "JPEGImages"
    output_mask_dir = output_root / "SegmentationClass"
    output_vis_dir = output_root / "SegmentationClassVis"
    output_split_dir = output_root / "ImageSets" / "Segmentation"

    ensure_empty_dir(output_mask_dir)
    ensure_empty_dir(output_vis_dir)
    if args.save_resized:
        ensure_empty_dir(output_jpeg_dir)
    else:
        copy_tree(input_jpeg_dir, output_jpeg_dir)
    copy_tree(input_split_dir, output_split_dir)

    image_ids = []
    for split_name in ["train.txt", "val.txt"]:
        split_path = input_split_dir / split_name
        if split_path.exists():
            image_ids.extend([line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()])
    image_ids = sorted(set(image_ids))

    model = DeeplabV3(model_path=str(args.model_path), cuda=args.cuda)
    refined_count = 0
    for image_id in tqdm(image_ids, desc="Refining grape mildew labels"):
        sample = load_labelme_sample(args.source_dir / f"{image_id}.json")
        if sample is None:
            continue
        work_sample = resize_sample(sample, args.max_side) if args.save_resized else sample
        probabilities = predict_probabilities(model, Image.fromarray(work_sample.image))
        refined_mask = refine_mask_with_probabilities(
            work_sample,
            probabilities,
            n_segments=args.slic_segments,
            compactness=args.slic_compactness,
            threshold=args.threshold,
            ignore_threshold=args.ignore_threshold,
            smooth_alpha=args.smooth_alpha,
            smooth_iters=args.smooth_iters,
            min_lesion_area=args.min_lesion_area,
            seed_radius=args.seed_radius,
            ignore_ring=args.ignore_ring,
            max_side=args.max_side,
        )
        if args.save_resized:
            Image.fromarray(work_sample.image).save(output_jpeg_dir / f"{image_id}.jpg", quality=95)
        Image.fromarray(refined_mask).save(output_mask_dir / f"{image_id}.png")
        Image.fromarray(colorize_mask(refined_mask)).save(output_vis_dir / f"{image_id}.png")
        refined_count += 1

    print(f"input_dataset={args.input_dataset_name}")
    print(f"output_dataset={args.output_dataset_name}")
    print(f"refined_images={refined_count}")
    print(f"output_root={output_root}")


if __name__ == "__main__":
    main()
