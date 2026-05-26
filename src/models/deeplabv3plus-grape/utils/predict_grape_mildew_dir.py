import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deeplab import DeeplabV3  # noqa: E402
from utils.grape_mildew_weak_pipeline import BACKGROUND_ID, LEAF_ID, LESION_ID, colorize_mask  # noqa: E402
from utils.utils import cvtColor, preprocess_input, resize_image  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP"}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict grape mildew masks and write LabelMe JSON files.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--only-unlabeled", action="store_true", default=True)
    parser.add_argument("--overwrite-json", action="store_true", default=False)
    parser.add_argument("--lesion-threshold", type=float, default=0.55)
    parser.add_argument("--min-lesion-area", type=int, default=80)
    parser.add_argument("--min-leaf-area", type=int, default=1000)
    parser.add_argument("--contour-epsilon", type=float, default=0.0025)
    parser.add_argument("--max-lesion-polygons", type=int, default=300)
    parser.add_argument("--work-max-side", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--summary-name", type=str, default="summary.csv")
    return parser.parse_args()


def get_work_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    if max_side <= 0 or max(width, height) <= max_side:
        return width, height
    scale = max_side / float(max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def predict_probabilities(model: DeeplabV3, image: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    image = cvtColor(image)
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
        pr = cv2.resize(pr, target_size, interpolation=cv2.INTER_LINEAR)
    return pr


def postprocess_prediction(probabilities: np.ndarray, lesion_threshold: float, min_lesion_area: int) -> np.ndarray:
    mask = probabilities.argmax(axis=-1).astype(np.uint8)
    lesion = (mask == LESION_ID) & (probabilities[:, :, LESION_ID] >= lesion_threshold)
    leaf = (mask == LEAF_ID) | lesion

    if leaf.any():
        leaf = binary_closing(leaf, disk(2))
        leaf = remove_small_objects(leaf, min_size=500)
        leaf = remove_small_holes(leaf, area_threshold=500)
    if lesion.any():
        lesion = binary_closing(lesion, disk(1))
        lesion = remove_small_objects(lesion, min_size=min_lesion_area)
        lesion = remove_small_holes(lesion, area_threshold=min_lesion_area)
        lesion = lesion & leaf

    out = np.zeros(mask.shape, dtype=np.uint8)
    out[leaf] = LEAF_ID
    out[lesion] = LESION_ID
    return out


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = colorize_mask(mask)
    overlay = image.copy()
    region = mask != BACKGROUND_ID
    overlay[region] = (image[region] * (1.0 - alpha) + color[region] * alpha).astype(np.uint8)
    return overlay


def contours_to_shapes(mask: np.ndarray, label: str, min_area: int, epsilon_ratio: float, limit: int = 100000):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    shapes = []
    for contour in contours[:limit]:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        epsilon = max(1.0, epsilon_ratio * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        points = [[float(x), float(y)] for [[x, y]] in approx]
        shapes.append(
            {
                "label": label,
                "points": points,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            }
        )
    return shapes


def write_labelme_json(image_path: Path, image: np.ndarray, mask: np.ndarray, output_json: Path, args):
    leaf_mask = (mask == LEAF_ID) | (mask == LESION_ID)
    lesion_mask = mask == LESION_ID
    shapes = []
    shapes.extend(contours_to_shapes(leaf_mask, "leaf", args.min_leaf_area, args.contour_epsilon, limit=20))
    shapes.extend(
        contours_to_shapes(
            lesion_mask,
            "lesion",
            args.min_lesion_area,
            args.contour_epsilon,
            limit=args.max_lesion_polygons,
        )
    )

    data = {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": int(image.shape[0]),
        "imageWidth": int(image.shape[1]),
    }
    output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "masks"
    vis_dir = args.output_dir / "vis"
    mask_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    model = DeeplabV3(model_path=str(args.model_path), cuda=args.cuda)
    image_paths = sorted([p for p in args.source_dir.iterdir() if p.suffix in IMAGE_SUFFIXES])
    if args.only_unlabeled:
        image_paths = [p for p in image_paths if not (p.with_suffix(".json")).exists()]
    if args.start_index > 0:
        image_paths = image_paths[args.start_index :]
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    summary_rows = ["image,json,mask,vis,leaf_pixels,lesion_pixels\n"]
    for image_path in tqdm(image_paths, desc="Predicting grape mildew"):
        output_json = image_path.with_suffix(".json")
        if output_json.exists() and not args.overwrite_json:
            continue
        image = np.array(Image.open(image_path).convert("RGB"))
        original_h, original_w = image.shape[:2]
        work_w, work_h = get_work_size(original_w, original_h, args.work_max_side)
        probabilities = predict_probabilities(model, Image.fromarray(image), (work_w, work_h))
        area_scale = (work_w * work_h) / float(original_w * original_h)
        work_min_lesion_area = max(4, int(round(args.min_lesion_area * area_scale)))
        mask = postprocess_prediction(probabilities, args.lesion_threshold, work_min_lesion_area)
        if (work_w, work_h) != (original_w, original_h):
            mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        stem = image_path.stem
        mask_path = mask_dir / f"{stem}.png"
        vis_path = vis_dir / f"{stem}.jpg"

        Image.fromarray(mask).save(mask_path)
        Image.fromarray(overlay_mask(image, mask)).save(vis_path, quality=92)
        write_labelme_json(image_path, image, mask, output_json, args)

        leaf_pixels = int(np.count_nonzero((mask == LEAF_ID) | (mask == LESION_ID)))
        lesion_pixels = int(np.count_nonzero(mask == LESION_ID))
        summary_rows.append(f"{image_path.name},{output_json.name},{mask_path.name},{vis_path.name},{leaf_pixels},{lesion_pixels}\n")

    (args.output_dir / args.summary_name).write_text("".join(summary_rows), encoding="utf-8")
    print(f"predicted={len(summary_rows) - 1}")
    print(f"json_dir={args.source_dir}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
