import argparse
import os

from PIL import Image
from tqdm import tqdm

from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate mIoU on a selected VOC-style dataset.")
    parser.add_argument("--miou-mode", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--dataset-name", type=str, default="VOC")
    parser.add_argument("--datasets-root", type=str, default=".")
    parser.add_argument("--vocdevkit-path", type=str, default="VOCdevkit")
    parser.add_argument("--miou-out-path", type=str, default="miou_out")
    parser.add_argument("--model-path", type=str, default=os.path.join("outputs", "grape_seg", "weights", "best_epoch_weights.pth"))
    parser.add_argument("--cuda", action="store_true", default=False)
    return parser.parse_args()


def resolve_dataset_path(args):
    if args.vocdevkit_path != "VOCdevkit":
        return args.vocdevkit_path
    return os.path.join(args.datasets_root, f"{args.dataset_name}devkit")


if __name__ == "__main__":
    args = parse_args()
    name_classes = ["background", "leaf", "lesion"]
    vocdevkit_path = resolve_dataset_path(args)

    image_ids = open(os.path.join(vocdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r", encoding="utf-8").read().splitlines()
    gt_dir = os.path.join(vocdevkit_path, "VOC2007/SegmentationClass/")
    pred_dir = os.path.join(args.miou_out_path, "detection-results")

    if args.miou_mode in {0, 1}:
        os.makedirs(pred_dir, exist_ok=True)

        print("Load model.")
        deeplab = DeeplabV3(model_path=args.model_path, cuda=args.cuda)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(vocdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            image = deeplab.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if args.miou_mode in {0, 2}:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, args.num_classes, name_classes)
        print("Get miou done.")
        show_results(args.miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
