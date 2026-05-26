import argparse
import datetime
import os
import time

import torch

import transforms as T
from segmentation_dataset import SegmentationDataset
from src import MobileV3Unet, ResNetUNet, UNet, VGG16UNet
from train_utils import create_lr_scheduler, evaluate, train_one_epoch


class SegmentationPresetTrain:
    def __init__(
        self,
        base_size,
        crop_size,
        hflip_prob=0.5,
        vflip_prob=0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)
        transforms = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            transforms.append(T.RandomVerticalFlip(vflip_prob))
        transforms.extend([T.RandomCrop(crop_size), T.ToTensor(), T.Normalize(mean=mean, std=std)])
        self.transforms = T.Compose(transforms)

    def __call__(self, image, target):
        return self.transforms(image, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

    def __call__(self, image, target):
        return self.transforms(image, target)


def get_transform(train, input_size, mean, std):
    if train:
        return SegmentationPresetTrain(input_size, input_size, mean=mean, std=std)
    return SegmentationPresetEval(mean=mean, std=std)


def create_model(model_name, num_classes):
    model_name = model_name.lower()
    if model_name == "unet":
        return UNet(in_channels=3, num_classes=num_classes)
    if model_name == "resnet_unet":
        return ResNetUNet(num_classes=num_classes)
    if model_name == "mobilenet_unet":
        return MobileV3Unet(num_classes=num_classes)
    if model_name == "vgg_unet":
        return VGG16UNet(num_classes=num_classes)
    raise ValueError(f"Unsupported model name: {model_name}")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_classes = args.num_classes + 1
    mean = tuple(args.mean)
    std = tuple(args.std)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    results_file = os.path.join(args.log_dir, f"results_{datetime.datetime.now():%Y%m%d-%H%M%S}.txt")

    train_dataset = SegmentationDataset(
        args.data_path,
        split="train",
        transforms=get_transform(True, args.input_size, mean, std),
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
    )
    val_dataset = SegmentationDataset(
        args.data_path,
        split="val",
        transforms=get_transform(False, args.input_size, mean, std),
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
    )

    batch_size = args.batch_size
    num_workers = args.num_workers
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
    )

    model = create_model(args.model_name, num_classes=num_classes)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            num_classes,
            lr_scheduler=lr_scheduler,
            print_freq=args.print_freq,
            scaler=scaler,
        )
        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")

        with open(results_file, "a", encoding="utf-8") as f:
            train_info = (
                f"[epoch: {epoch}]\n"
                f"train_loss: {mean_loss:.4f}\n"
                f"lr: {lr:.6f}\n"
                f"dice coefficient: {dice:.3f}\n"
            )
            f.write(train_info + val_info + "\n\n")

        if args.save_best and best_dice >= dice:
            continue
        best_dice = max(best_dice, dice)

        save_file = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        filename = "best_model.pth" if args.save_best else f"model_{epoch}.pth"
        torch.save(save_file, os.path.join(args.save_dir, filename))

    total_time = time.time() - start_time
    print(f"training time {datetime.timedelta(seconds=int(total_time))}")


def parse_args():
    parser = argparse.ArgumentParser(description="U-Net attention experiment training")
    parser.add_argument("--data-path", default=r"D:\SegData\new_dataset", help="dataset root")
    parser.add_argument("--image-dir", default="images", help="image folder name under split")
    parser.add_argument("--mask-dir", default="masks", help="mask folder name under split")
    parser.add_argument("--model-name", default="unet", choices=["unet", "resnet_unet", "mobilenet_unet", "vgg_unet"])
    parser.add_argument("--num-classes", default=1, type=int, help="foreground classes, excluding background")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=2, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--input-size", default=480, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, dest="weight_decay")
    parser.add_argument("--print-freq", default=10, type=int)
    parser.add_argument("--resume", default="", help="resume checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--save-best", default=True, type=bool)
    parser.add_argument("--save-dir", default=r"D:\SegRuns\outputs\unet-attention\default\exp01\weights")
    parser.add_argument("--log-dir", default=r"D:\SegRuns\logs\unet-attention\default\exp01")
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--amp", default=False, type=bool)
    parser.add_argument("--mean", default=(0.485, 0.456, 0.406), nargs=3, type=float)
    parser.add_argument("--std", default=(0.229, 0.224, 0.225), nargs=3, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
