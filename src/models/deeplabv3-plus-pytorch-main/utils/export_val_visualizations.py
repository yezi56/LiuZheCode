import argparse
import os
import sys
from shutil import copy2

from PIL import Image
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from deeplab import DeeplabV3


def parse_args():
    parser = argparse.ArgumentParser(description="Export validation-set segmentation visualizations.")
    parser.add_argument(
        "--vocdevkit-path",
        type=str,
        default=os.path.join(REPO_ROOT, "VOCdevkit"),
        help="VOCdevkit root path.",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=os.path.join(REPO_ROOT, "VOCdevkit", "VOC2007", "ImageSets", "Segmentation", "val.txt"),
        help="Validation split txt file.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(REPO_ROOT, "outputs", "grape_seg", "weights", "best_epoch_weights.pth"),
        help="Model weights path.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(REPO_ROOT, "outputs", "grape_seg", "val_visualizations"),
        help="Directory to save originals, overlays, and colored masks.",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Use CUDA for inference when available.",
    )
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    default_split = os.path.join(REPO_ROOT, "VOCdevkit", "VOC2007", "ImageSets", "Segmentation", "val.txt")
    if args.split_file == default_split:
        args.split_file = os.path.join(args.vocdevkit_path, "VOC2007", "ImageSets", "Segmentation", "val.txt")

    with open(args.split_file, "r", encoding="utf-8") as f:
        image_ids = [line.strip() for line in f if line.strip()]

    jpeg_dir = os.path.join(args.vocdevkit_path, "VOC2007", "JPEGImages")
    save_originals = os.path.join(args.save_dir, "originals")
    save_overlays = os.path.join(args.save_dir, "overlays")
    save_masks = os.path.join(args.save_dir, "masks_color")

    ensure_dir(args.save_dir)
    ensure_dir(save_originals)
    ensure_dir(save_overlays)
    ensure_dir(save_masks)

    deeplab = DeeplabV3(model_path=args.model_path, cuda=args.cuda)

    for image_id in tqdm(image_ids, desc="Export val visualizations"):
        image_path = os.path.join(jpeg_dir, image_id + ".jpg")
        if not os.path.exists(image_path):
            continue

        copy2(image_path, os.path.join(save_originals, image_id + ".jpg"))

        image = Image.open(image_path)

        deeplab.mix_type = 0
        overlay = deeplab.detect_image(image)
        overlay.save(os.path.join(save_overlays, image_id + ".png"))

        image = Image.open(image_path)
        deeplab.mix_type = 1
        mask_color = deeplab.detect_image(image)
        mask_color.save(os.path.join(save_masks, image_id + ".png"))

    print(f"Saved originals to: {save_originals}")
    print(f"Saved overlays to: {save_overlays}")
    print(f"Saved color masks to: {save_masks}")


if __name__ == "__main__":
    main()
