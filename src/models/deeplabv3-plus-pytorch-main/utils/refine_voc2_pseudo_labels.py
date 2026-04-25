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

from deeplab import DeeplabV3
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.voc2_pipeline import colorize_mask, load_labelme_sample, refine_mask_with_probabilities


def parse_args():
    parser = argparse.ArgumentParser(description="Refine VOC2 pseudo labels with model predictions.")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--input-dataset-name", type=str, default="VOC2")
    parser.add_argument("--output-dataset-name", type=str, default="VOC2_iter1")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / "outputs" / "voc2_seed" / "weights" / "best_epoch_weights.pth",
    )
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--slic-segments", type=int, default=350)
    parser.add_argument("--slic-compactness", type=float, default=12.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--smooth-alpha", type=float, default=0.65)
    parser.add_argument("--smooth-iters", type=int, default=8)
    parser.add_argument("--min-lesion-area", type=int, default=64)
    parser.add_argument("--max-side", type=int, default=1280)
    return parser.parse_args()


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
            int((model.input_shape[0] - nh) // 2): int((model.input_shape[0] - nh) // 2 + nh),
            int((model.input_shape[1] - nw) // 2): int((model.input_shape[1] - nw) // 2 + nw),
        ]
        pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    return pr


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


def copy_tree(src: Path, dst: Path):
    if dst.exists():
        clear_dir(dst)
        for item in src.iterdir():
            target = dst / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
        return
    shutil.copytree(src, dst)


def main():
    args = parse_args()
    input_root = ROOT / f"{args.input_dataset_name}devkit" / "VOC2007"
    output_root = ROOT / f"{args.output_dataset_name}devkit" / "VOC2007"

    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_root}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    jpeg_dir = input_root / "JPEGImages"
    split_dir = input_root / "ImageSets" / "Segmentation"

    output_jpeg_dir = output_root / "JPEGImages"
    output_mask_dir = output_root / "SegmentationClass"
    output_vis_dir = output_root / "SegmentationClassVis"
    output_split_dir = output_root / "ImageSets" / "Segmentation"

    clear_dir(output_mask_dir)
    clear_dir(output_vis_dir)
    if output_root.exists() and not output_jpeg_dir.exists():
        output_jpeg_dir.mkdir(parents=True, exist_ok=True)
    copy_tree(jpeg_dir, output_jpeg_dir)
    copy_tree(split_dir, output_split_dir)

    image_ids = []
    for split_name in ["train.txt", "val.txt"]:
        split_path = split_dir / split_name
        if split_path.exists():
            image_ids.extend([line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()])
    image_ids = sorted(set(image_ids))

    model = DeeplabV3(model_path=str(args.model_path), cuda=args.cuda)

    refined_count = 0
    for image_id in tqdm(image_ids, desc="Refining VOC2 pseudo labels"):
        sample = load_labelme_sample(args.source_dir / f"{image_id}.json")
        if sample is None:
            continue
        probabilities = predict_probabilities(model, Image.fromarray(sample.image))
        refined_mask = refine_mask_with_probabilities(
            sample,
            probabilities,
            n_segments=args.slic_segments,
            compactness=args.slic_compactness,
            threshold=args.threshold,
            smooth_alpha=args.smooth_alpha,
            smooth_iters=args.smooth_iters,
            min_lesion_area=args.min_lesion_area,
            max_side=args.max_side,
        )
        Image.fromarray(refined_mask).save(output_mask_dir / f"{image_id}.png")
        Image.fromarray(colorize_mask(refined_mask)).save(output_vis_dir / f"{image_id}.png")
        refined_count += 1

    print(f"input_dataset={args.input_dataset_name}")
    print(f"output_dataset={args.output_dataset_name}")
    print(f"refined_images={refined_count}")
    print(f"output_root={output_root}")


if __name__ == "__main__":
    main()
