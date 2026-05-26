import base64
import json
import random
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "datasets" / "grape"
VOC_ROOT = ROOT / "VOCdevkit" / "VOC2007"
JPEG_DIR = VOC_ROOT / "JPEGImages"
MASK_DIR = VOC_ROOT / "SegmentationClass"
SPLIT_DIR = VOC_ROOT / "ImageSets" / "Segmentation"

CLASS_TO_ID = {
    "leaf": 1,
    "lesion": 2,
}

TRAIN_RATIO = 0.7
VAL_RATIO = 0.3
RANDOM_SEED = 42


def ensure_dirs():
    for directory in [JPEG_DIR, MASK_DIR, SPLIT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def load_image(data: dict, json_path: Path) -> Image.Image:
    image_data = data.get("imageData")
    if image_data:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
    else:
        image_path = data.get("imagePath")
        if not image_path:
            raise ValueError(f"{json_path.name} missing both imageData and imagePath")
        image = Image.open(json_path.parent / image_path)
    return image.convert("RGB")


def draw_shape(draw: ImageDraw.ImageDraw, shape: dict, fill_value: int):
    points = [tuple(point) for point in shape.get("points", [])]
    shape_type = shape.get("shape_type") or "polygon"

    if shape_type in {"polygon", "points"}:
        if len(points) == 1:
            x, y = points[0]
            radius = 2
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill_value)
            return
        if len(points) == 2:
            draw.line(points, fill=fill_value, width=3)
            return
        draw.polygon(points, fill=fill_value)
        return

    if shape_type == "rectangle":
        if len(points) != 2:
            raise ValueError(f"rectangle requires 2 points, got {len(points)}")
        draw.rectangle([points[0], points[1]], fill=fill_value)
        return

    if shape_type == "circle":
        if len(points) != 2:
            raise ValueError(f"circle requires 2 points, got {len(points)}")
        (x0, y0), (x1, y1) = points
        radius = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        draw.ellipse([x0 - radius, y0 - radius, x0 + radius, y0 + radius], fill=fill_value)
        return

    raise ValueError(f"Unsupported shape_type={shape_type}")


def convert_one(json_path: Path):
    data = load_json(json_path)
    image = load_image(data, json_path)
    width, height = image.size

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in CLASS_TO_ID:
            raise ValueError(f"{json_path.name} contains unknown label: {label}")
        draw_shape(draw, shape, CLASS_TO_ID[label])

    stem = json_path.stem
    image.save(JPEG_DIR / f"{stem}.jpg", quality=95)
    mask.save(MASK_DIR / f"{stem}.png")

    return stem, np.unique(np.array(mask)).tolist()


def write_split_file(path: Path, items):
    path.write_text("".join(f"{name}\n" for name in items), encoding="utf-8")


def generate_splits(stems):
    stems = sorted(stems)
    random.Random(RANDOM_SEED).shuffle(stems)

    total = len(stems)
    train_count = int(total * TRAIN_RATIO)
    val_count = total - train_count
    train = stems[:train_count]
    val = stems[train_count:train_count + val_count]

    write_split_file(SPLIT_DIR / "train.txt", train)
    write_split_file(SPLIT_DIR / "val.txt", val)
    for legacy_name in ["test.txt", "trainval.txt"]:
        legacy_path = SPLIT_DIR / legacy_name
        if legacy_path.exists():
            try:
                legacy_path.unlink()
            except PermissionError:
                print(f"warning: could not delete legacy split file {legacy_path}")

    print(f"split_total={total}")
    print(f"train={len(train)}")
    print(f"val={len(val)}")


def main():
    ensure_dirs()

    json_files = sorted(SOURCE_DIR.glob("*.json"))
    image_stems = {path.stem for path in SOURCE_DIR.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}}
    json_stems = {path.stem for path in json_files}
    only_images = sorted(image_stems - json_stems)

    if only_images:
        print("images_without_json:")
        for stem in only_images:
            print(stem)

    stems = []
    label_values_seen = set()
    for json_path in json_files:
        stem, label_values = convert_one(json_path)
        stems.append(stem)
        label_values_seen.update(label_values)

    generate_splits(stems)

    print(f"converted_json={len(json_files)}")
    print(f"saved_jpg={len(list(JPEG_DIR.glob('*.jpg')))}")
    print(f"saved_png={len(list(MASK_DIR.glob('*.png')))}")
    print(f"mask_values={sorted(label_values_seen)}")


if __name__ == "__main__":
    if abs(TRAIN_RATIO + VAL_RATIO - 1.0) > 1e-8:
        raise ValueError("TRAIN_RATIO + VAL_RATIO must equal 1.0")
    main()
