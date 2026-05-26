import heapq
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.color import rgb2lab
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    disk,
    remove_small_holes,
    remove_small_objects,
)
from skimage.segmentation import slic


BACKGROUND_ID = 0
LEAF_ID = 1
LESION_ID = 2
IGNORE_ID = 255

POINT_TO_CLASS = {
    "lesion": LESION_ID,
    "mildew": LESION_ID,
    "mildew_visible": LESION_ID,
    "visible_mildew": LESION_ID,
    "no_lesion": LEAF_ID,
    "healthy_leaf": LEAF_ID,
    "leaf_negative": LEAF_ID,
    "ignore": IGNORE_ID,
}


@dataclass
class LabelmeSample:
    stem: str
    image_path: Path
    image: np.ndarray
    leaf_mask: np.ndarray
    seed_points: List[Tuple[int, int, int]]


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def find_image_for_stem(source_dir: Path, stem: str) -> Optional[Path]:
    for suffix in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP"]:
        candidate = source_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def polygon_to_mask(size: Tuple[int, int], polygons: Sequence[Sequence[Sequence[float]]]) -> np.ndarray:
    width, height = size
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for points in polygons:
        if len(points) >= 3:
            draw.polygon([tuple(point) for point in points], fill=1)
    return np.array(mask_img, dtype=np.uint8)


def load_labelme_sample(json_path: Path) -> Optional[LabelmeSample]:
    data = load_json(json_path)
    stem = json_path.stem
    source_dir = json_path.parent
    image_path = find_image_for_stem(source_dir, stem)
    if image_path is None:
        image_name = data.get("imagePath")
        if image_name:
            candidate = source_dir / image_name
            if candidate.exists():
                image_path = candidate
    if image_path is None or not image_path.exists():
        return None

    image = np.array(Image.open(image_path).convert("RGB"))
    height, width = image.shape[:2]
    leaf_polygons: List[Sequence[Sequence[float]]] = []
    seed_points: List[Tuple[int, int, int]] = []

    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        shape_type = shape.get("shape_type") or "polygon"
        points = shape.get("points", [])
        if label == "leaf" and shape_type == "polygon" and len(points) >= 3:
            leaf_polygons.append(points)
        elif label in POINT_TO_CLASS and shape_type == "point" and points:
            x, y = points[0]
            x = int(np.clip(round(x), 0, width - 1))
            y = int(np.clip(round(y), 0, height - 1))
            seed_points.append((x, y, POINT_TO_CLASS[label]))

    if not leaf_polygons:
        return None

    leaf_mask = polygon_to_mask((width, height), leaf_polygons)
    seed_points = [(x, y, cls_id) for x, y, cls_id in seed_points if leaf_mask[y, x] > 0]
    if not seed_points:
        return None

    return LabelmeSample(stem=stem, image_path=image_path, image=image, leaf_mask=leaf_mask, seed_points=seed_points)


def resize_for_superpixels(
    image: np.ndarray,
    leaf_mask: np.ndarray,
    seed_points: Sequence[Tuple[int, int, int]],
    max_side: int,
    probabilities: Optional[np.ndarray] = None,
):
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / float(max(height, width)))
    if scale >= 1.0:
        return image, leaf_mask, list(seed_points), probabilities, scale

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    resized_leaf_mask = cv2.resize(leaf_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    resized_seed_points = []
    for x, y, cls_id in seed_points:
        sx = int(np.clip(round(x * scale), 0, new_width - 1))
        sy = int(np.clip(round(y * scale), 0, new_height - 1))
        resized_seed_points.append((sx, sy, cls_id))

    resized_probabilities = None
    if probabilities is not None:
        resized_probabilities = cv2.resize(probabilities, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_image, resized_leaf_mask, resized_seed_points, resized_probabilities, scale


def compute_superpixels(image: np.ndarray, leaf_mask: np.ndarray, n_segments: int, compactness: float) -> np.ndarray:
    if int(np.count_nonzero(leaf_mask)) == 0:
        return np.zeros(leaf_mask.shape, dtype=np.int32)
    image_float = image.astype(np.float32) / 255.0
    segments = slic(
        image_float,
        n_segments=n_segments,
        compactness=compactness,
        start_label=1,
        mask=leaf_mask.astype(bool),
        channel_axis=-1,
    )
    return segments.astype(np.int32)


def collect_segment_stats(
    image: np.ndarray, segments: np.ndarray, leaf_mask: np.ndarray
) -> Tuple[List[int], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    ids = np.unique(segments[leaf_mask > 0]).tolist()
    ids = [int(seg_id) for seg_id in ids if int(seg_id) != 0]
    lab = rgb2lab(image.astype(np.float32) / 255.0)
    mean_lab: Dict[int, np.ndarray] = {}
    centroids: Dict[int, np.ndarray] = {}
    for seg_id in ids:
        ys, xs = np.where(segments == seg_id)
        if len(xs) == 0:
            continue
        mean_lab[seg_id] = lab[ys, xs].mean(axis=0)
        centroids[seg_id] = np.array([ys.mean(), xs.mean()], dtype=np.float32)
    ids = [seg_id for seg_id in ids if seg_id in mean_lab]
    return ids, mean_lab, centroids


def build_adjacency(segments: np.ndarray, leaf_mask: np.ndarray) -> Dict[int, Set[int]]:
    valid = leaf_mask > 0
    pairs = []
    for a, b, va, vb in [
        (segments[:, :-1], segments[:, 1:], valid[:, :-1], valid[:, 1:]),
        (segments[:-1, :], segments[1:, :], valid[:-1, :], valid[1:, :]),
    ]:
        mask = va & vb & (a != b) & (a > 0) & (b > 0)
        if mask.any():
            edge = np.stack([a[mask], b[mask]], axis=1).astype(np.int32)
            edge.sort(axis=1)
            pairs.append(edge)

    adjacency: Dict[int, Set[int]] = {}
    if not pairs:
        return adjacency
    all_pairs = np.unique(np.concatenate(pairs, axis=0), axis=0)
    for a, b in all_pairs:
        a = int(a)
        b = int(b)
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    return adjacency


def seed_segments_from_points(
    segments: np.ndarray, seed_points: Sequence[Tuple[int, int, int]]
) -> Dict[int, int]:
    votes: Dict[int, List[int]] = {}
    for x, y, cls_id in seed_points:
        if cls_id == IGNORE_ID:
            continue
        seg_id = int(segments[y, x])
        if seg_id == 0:
            continue
        votes.setdefault(seg_id, []).append(cls_id)

    seed_labels: Dict[int, int] = {}
    for seg_id, labels in votes.items():
        lesion_votes = sum(1 for label in labels if label == LESION_ID)
        leaf_votes = sum(1 for label in labels if label == LEAF_ID)
        seed_labels[seg_id] = LESION_ID if lesion_votes >= leaf_votes else LEAF_ID
    return seed_labels


def edge_weight(a: int, b: int, mean_lab: Dict[int, np.ndarray], centroids: Dict[int, np.ndarray]) -> float:
    color_cost = float(np.linalg.norm(mean_lab[a] - mean_lab[b]) / 30.0)
    spatial_cost = float(np.linalg.norm(centroids[a] - centroids[b]) / 200.0)
    return 1e-4 + 0.80 * color_cost + 0.20 * spatial_cost


def multi_source_dijkstra(
    segment_ids: Sequence[int],
    adjacency: Dict[int, Set[int]],
    seeds: Iterable[int],
    mean_lab: Dict[int, np.ndarray],
    centroids: Dict[int, np.ndarray],
) -> Dict[int, float]:
    dist = {seg_id: float("inf") for seg_id in segment_ids}
    heap: List[Tuple[float, int]] = []
    for seg_id in seeds:
        if seg_id not in dist:
            continue
        dist[seg_id] = 0.0
        heapq.heappush(heap, (0.0, seg_id))

    while heap:
        cur_dist, seg_id = heapq.heappop(heap)
        if cur_dist > dist[seg_id]:
            continue
        for neighbor in adjacency.get(seg_id, []):
            weight = edge_weight(seg_id, neighbor, mean_lab, centroids)
            new_dist = cur_dist + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))
    return dist


def nearest_color_score(
    segment_ids: Sequence[int],
    mean_lab: Dict[int, np.ndarray],
    lesion_seed_ids: Sequence[int],
    leaf_seed_ids: Sequence[int],
) -> Dict[int, float]:
    if not lesion_seed_ids or not leaf_seed_ids:
        return {seg_id: 0.5 for seg_id in segment_ids}
    lesion_proto = np.stack([mean_lab[seg_id] for seg_id in lesion_seed_ids if seg_id in mean_lab], axis=0)
    leaf_proto = np.stack([mean_lab[seg_id] for seg_id in leaf_seed_ids if seg_id in mean_lab], axis=0)
    scores: Dict[int, float] = {}
    for seg_id in segment_ids:
        lab = mean_lab[seg_id][None, :]
        lesion_dist = float(np.min(np.linalg.norm(lesion_proto - lab, axis=1)))
        leaf_dist = float(np.min(np.linalg.norm(leaf_proto - lab, axis=1)))
        scores[seg_id] = leaf_dist / max(lesion_dist + leaf_dist, 1e-6)
    return scores


def propagate_conservative_labels(
    segment_ids: Sequence[int],
    adjacency: Dict[int, Set[int]],
    seed_labels: Dict[int, int],
    mean_lab: Dict[int, np.ndarray],
    centroids: Dict[int, np.ndarray],
    lesion_threshold: float = 0.68,
    ignore_threshold: float = 0.56,
    color_weight: float = 0.25,
) -> Dict[int, int]:
    lesion_seeds = [seg_id for seg_id, cls_id in seed_labels.items() if cls_id == LESION_ID]
    leaf_seeds = [seg_id for seg_id, cls_id in seed_labels.items() if cls_id == LEAF_ID]
    if not lesion_seeds:
        return {seg_id: LEAF_ID for seg_id in segment_ids}
    if not leaf_seeds:
        return {seg_id: IGNORE_ID for seg_id in segment_ids}

    lesion_dist = multi_source_dijkstra(segment_ids, adjacency, lesion_seeds, mean_lab, centroids)
    leaf_dist = multi_source_dijkstra(segment_ids, adjacency, leaf_seeds, mean_lab, centroids)
    color_scores = nearest_color_score(segment_ids, mean_lab, lesion_seeds, leaf_seeds)

    labels: Dict[int, int] = {}
    for seg_id in segment_ids:
        if seg_id in seed_labels:
            labels[seg_id] = seed_labels[seg_id]
            continue
        ld = lesion_dist.get(seg_id, float("inf"))
        hd = leaf_dist.get(seg_id, float("inf"))
        if math.isinf(ld) and math.isinf(hd):
            labels[seg_id] = IGNORE_ID
            continue
        dist_score = hd / max(ld + hd, 1e-6)
        score = (1.0 - color_weight) * dist_score + color_weight * color_scores[seg_id]
        if score >= lesion_threshold:
            labels[seg_id] = LESION_ID
        elif score >= ignore_threshold:
            labels[seg_id] = IGNORE_ID
        else:
            labels[seg_id] = LEAF_ID
    return labels


def mask_from_segment_labels(
    segments: np.ndarray,
    leaf_mask: np.ndarray,
    segment_labels: Dict[int, int],
) -> np.ndarray:
    mask = np.zeros(segments.shape, dtype=np.uint8)
    mask[leaf_mask > 0] = LEAF_ID
    for seg_id, cls_id in segment_labels.items():
        if cls_id == IGNORE_ID:
            mask[segments == seg_id] = IGNORE_ID
        else:
            mask[segments == seg_id] = cls_id
    mask[leaf_mask == 0] = BACKGROUND_ID
    return mask


def enforce_seed_points(mask: np.ndarray, seed_points: Sequence[Tuple[int, int, int]], radius: int = 5) -> np.ndarray:
    updated = mask.copy()
    height, width = updated.shape
    for x, y, cls_id in seed_points:
        if cls_id == IGNORE_ID:
            continue
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        patch = updated[y0:y1, x0:x1]
        patch[circle] = cls_id
        updated[y0:y1, x0:x1] = patch
    return updated


def postprocess_mask(
    mask: np.ndarray,
    leaf_mask: np.ndarray,
    seed_points: Sequence[Tuple[int, int, int]],
    min_lesion_area: int = 48,
    seed_radius: int = 5,
    ignore_ring: int = 2,
) -> np.ndarray:
    original_ignore = mask == IGNORE_ID
    lesion = mask == LESION_ID
    if lesion.any():
        lesion = binary_closing(lesion, disk(1))
        lesion = remove_small_objects(lesion, min_size=min_lesion_area)
        lesion = remove_small_holes(lesion, area_threshold=min_lesion_area)

    result = np.zeros_like(mask, dtype=np.uint8)
    result[leaf_mask > 0] = LEAF_ID
    result[original_ignore & (leaf_mask > 0)] = IGNORE_ID
    if ignore_ring > 0 and lesion.any():
        ring = binary_dilation(lesion, disk(ignore_ring)) & (~lesion) & (leaf_mask > 0)
        result[ring] = IGNORE_ID
    result[lesion] = LESION_ID
    result = enforce_seed_points(result, seed_points, radius=seed_radius)
    result[leaf_mask == 0] = BACKGROUND_ID
    return result


def build_initial_pseudo_mask(
    sample: LabelmeSample,
    n_segments: int = 900,
    compactness: float = 8.0,
    lesion_threshold: float = 0.68,
    ignore_threshold: float = 0.56,
    color_weight: float = 0.25,
    min_lesion_area: int = 48,
    seed_radius: int = 5,
    ignore_ring: int = 2,
    max_side: int = 1024,
) -> np.ndarray:
    image_small, leaf_mask_small, seed_points_small, _, _ = resize_for_superpixels(
        sample.image, sample.leaf_mask, sample.seed_points, max_side=max_side
    )
    segments = compute_superpixels(image_small, leaf_mask_small, n_segments=n_segments, compactness=compactness)
    segment_ids, mean_lab, centroids = collect_segment_stats(image_small, segments, leaf_mask_small)
    adjacency = build_adjacency(segments, leaf_mask_small)
    seed_labels = seed_segments_from_points(segments, seed_points_small)
    segment_labels = propagate_conservative_labels(
        segment_ids,
        adjacency,
        seed_labels,
        mean_lab,
        centroids,
        lesion_threshold=lesion_threshold,
        ignore_threshold=ignore_threshold,
        color_weight=color_weight,
    )
    mask_small = mask_from_segment_labels(segments, leaf_mask_small, segment_labels)
    mask = cv2.resize(mask_small, (sample.image.shape[1], sample.image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return postprocess_mask(
        mask,
        sample.leaf_mask,
        sample.seed_points,
        min_lesion_area=min_lesion_area,
        seed_radius=seed_radius,
        ignore_ring=ignore_ring,
    )


def refine_mask_with_probabilities(
    sample: LabelmeSample,
    probabilities: np.ndarray,
    n_segments: int = 900,
    compactness: float = 8.0,
    threshold: float = 0.68,
    ignore_threshold: float = 0.48,
    smooth_alpha: float = 0.70,
    smooth_iters: int = 8,
    min_lesion_area: int = 48,
    seed_radius: int = 5,
    ignore_ring: int = 2,
    max_side: int = 1024,
) -> np.ndarray:
    image_small, leaf_mask_small, seed_points_small, probabilities_small, _ = resize_for_superpixels(
        sample.image,
        sample.leaf_mask,
        sample.seed_points,
        max_side=max_side,
        probabilities=probabilities,
    )
    segments = compute_superpixels(image_small, leaf_mask_small, n_segments=n_segments, compactness=compactness)
    segment_ids, mean_lab, centroids = collect_segment_stats(image_small, segments, leaf_mask_small)
    adjacency = build_adjacency(segments, leaf_mask_small)
    seed_labels = seed_segments_from_points(segments, seed_points_small)

    lesion_prob = probabilities_small[:, :, LESION_ID]
    leaf_prob = probabilities_small[:, :, LEAF_ID]
    seg_scores: Dict[int, float] = {}
    for seg_id in segment_ids:
        region = segments == seg_id
        lesion_score = float(lesion_prob[region].mean())
        leaf_score = float(leaf_prob[region].mean())
        seg_scores[seg_id] = lesion_score / max(lesion_score + leaf_score, 1e-6)

    for seg_id, cls_id in seed_labels.items():
        seg_scores[seg_id] = 1.0 if cls_id == LESION_ID else 0.0

    for _ in range(smooth_iters):
        updated = dict(seg_scores)
        for seg_id in segment_ids:
            if seg_id in seed_labels:
                continue
            neighbors = list(adjacency.get(seg_id, []))
            if not neighbors:
                continue
            weights = np.array(
                [1.0 / edge_weight(seg_id, neighbor, mean_lab, centroids) for neighbor in neighbors],
                dtype=np.float32,
            )
            neighbor_scores = np.array([seg_scores[neighbor] for neighbor in neighbors], dtype=np.float32)
            neighbor_mean = float(np.sum(weights * neighbor_scores) / max(np.sum(weights), 1e-6))
            updated[seg_id] = smooth_alpha * seg_scores[seg_id] + (1.0 - smooth_alpha) * neighbor_mean
        seg_scores = updated
        for seg_id, cls_id in seed_labels.items():
            seg_scores[seg_id] = 1.0 if cls_id == LESION_ID else 0.0

    mask_small = np.zeros(leaf_mask_small.shape, dtype=np.uint8)
    mask_small[leaf_mask_small > 0] = LEAF_ID
    for seg_id in segment_ids:
        score = seg_scores[seg_id]
        if score >= threshold:
            mask_small[segments == seg_id] = LESION_ID
        elif score >= ignore_threshold:
            mask_small[segments == seg_id] = IGNORE_ID

    mask = cv2.resize(mask_small, (sample.image.shape[1], sample.image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return postprocess_mask(
        mask,
        sample.leaf_mask,
        sample.seed_points,
        min_lesion_area=min_lesion_area,
        seed_radius=seed_radius,
        ignore_ring=ignore_ring,
    )


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    vis[mask == BACKGROUND_ID] = [0, 0, 0]
    vis[mask == LEAF_ID] = [60, 180, 75]
    vis[mask == LESION_ID] = [230, 25, 75]
    vis[mask == IGNORE_ID] = [150, 150, 150]
    return vis


def ensure_empty_dir(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def mask_stats(mask: np.ndarray) -> Dict[str, int]:
    return {
        "background": int(np.count_nonzero(mask == BACKGROUND_ID)),
        "leaf": int(np.count_nonzero(mask == LEAF_ID)),
        "lesion": int(np.count_nonzero(mask == LESION_ID)),
        "ignore": int(np.count_nonzero(mask == IGNORE_ID)),
    }
