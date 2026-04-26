import argparse
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.deeplabv3_plus_dual import DeepLab
from nets.deeplabv3_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import download_weights, seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ on the grape VOC dataset.")
    parser.add_argument("--cuda", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--distributed", type=str2bool, default=False)
    parser.add_argument("--sync-bn", type=str2bool, default=False)
    parser.add_argument("--fp16", type=str2bool, default=False)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--backbone", type=str, default="mobilenet", choices=["mobilenet", "mobilenet_swin", "xception"])
    parser.add_argument("--pretrained", type=str2bool, default=False)
    parser.add_argument("--model-path", type=str, default="model_data/deeplab_mobilenetv2.pth")
    parser.add_argument("--downsample-factor", type=int, default=16, choices=[8, 16])
    parser.add_argument("--attention-type", type=str, default="cbam")
    parser.add_argument("--use-ppm", type=str2bool, default=False)
    parser.add_argument("--ppm-bins", nargs="+", type=int, default=[1, 2, 3, 6])
    parser.add_argument("--input-shape", nargs=2, type=int, default=[512, 512], metavar=("H", "W"))
    parser.add_argument("--init-epoch", type=int, default=0)
    parser.add_argument("--freeze-epoch", type=int, default=50)
    parser.add_argument("--freeze-batch-size", type=int, default=8)
    parser.add_argument("--unfreeze-epoch", type=int, default=500)
    parser.add_argument("--unfreeze-batch-size", type=int, default=4)
    parser.add_argument("--freeze-train", type=str2bool, default=True)
    parser.add_argument("--init-lr", type=float, default=7e-3)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--optimizer-type", type=str, default="sgd", choices=["adam", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-decay-type", type=str, default="cos", choices=["cos", "step"])
    parser.add_argument("--save-period", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default=os.path.join("outputs", "grape_seg", "weights"))
    parser.add_argument("--log-dir", type=str, default=os.path.join("outputs", "grape_seg", "logs"))
    parser.add_argument("--eval-flag", type=str2bool, default=True)
    parser.add_argument("--eval-period", type=int, default=5)
    parser.add_argument("--dataset-name", type=str, default="VOC")
    parser.add_argument("--datasets-root", type=str, default=".")
    parser.add_argument("--vocdevkit-path", type=str, default="VOCdevkit")
    parser.add_argument("--dice-loss", type=str2bool, default=False)
    parser.add_argument("--focal-loss", type=str2bool, default=False)
    parser.add_argument("--focal-alpha", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--mix-mode", type=str, default="none", choices=["none", "mixup", "cutmix"])
    parser.add_argument("--mix-prob", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--cls-weights", nargs="+", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def build_cls_weights(args):
    if args.cls_weights is None:
        return np.ones([args.num_classes], np.float32)
    if len(args.cls_weights) != args.num_classes:
        raise ValueError("--cls-weights length must match --num-classes")
    return np.array(args.cls_weights, np.float32)


def resolve_dataset_path(args):
    if args.vocdevkit_path != "VOCdevkit":
        return args.vocdevkit_path
    return os.path.join(args.datasets_root, f"{args.dataset_name}devkit")


if __name__ == "__main__":
    args = parse_args()
    input_shape = list(args.input_shape)
    min_lr = args.min_lr if args.min_lr is not None else args.init_lr * 0.01
    cls_weights = build_cls_weights(args)
    use_cuda = args.cuda and torch.cuda.is_available()
    dataset_path = resolve_dataset_path(args)

    seed_everything(args.seed)
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device("cuda" if use_cuda else "cpu")
        local_rank = 0
        rank = 0

    if args.pretrained:
        if args.distributed:
            if local_rank == 0:
                download_weights(args.backbone)
            dist.barrier()
        else:
            download_weights(args.backbone)

    model = DeepLab(
        num_classes=args.num_classes,
        backbone=args.backbone,
        downsample_factor=args.downsample_factor,
        pretrained=args.pretrained,
        attention_type=args.attention_type,
        use_ppm=args.use_ppm,
        ppm_bins=args.ppm_bins,
    )
    if not args.pretrained:
        weights_init(model)

    if args.model_path:
        if local_rank == 0:
            print(f"Load weights {args.model_path}.")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "...\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "...\nFail To Load Key Num:", len(no_load_key))
            print("\n\033[1;33;44mHead mismatch is normal when switching to 3 classes, but backbone mismatch is not.\033[0m")

    if local_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
        log_dir = os.path.join(args.log_dir, "loss_" + time_str)
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        log_dir = None
        loss_history = None

    if args.fp16:
        from torch.cuda.amp import GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if args.sync_bn and ngpus_per_node > 1 and args.distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif args.sync_bn:
        print("sync_bn is only supported in distributed multi-GPU mode.")

    if use_cuda:
        if args.distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    with open(os.path.join(dataset_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r", encoding="utf-8") as f:
        train_lines = f.readlines()
    with open(os.path.join(dataset_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r", encoding="utf-8") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            dataset_name=args.dataset_name,
            dataset_path=dataset_path,
            num_classes=args.num_classes,
            backbone=args.backbone,
            attention_type=args.attention_type,
            use_ppm=args.use_ppm,
            ppm_bins=args.ppm_bins,
            model_path=args.model_path,
            input_shape=input_shape,
            Init_Epoch=args.init_epoch,
            Freeze_Epoch=args.freeze_epoch,
            UnFreeze_Epoch=args.unfreeze_epoch,
            Freeze_batch_size=args.freeze_batch_size,
            Unfreeze_batch_size=args.unfreeze_batch_size,
            Freeze_Train=args.freeze_train,
            Init_lr=args.init_lr,
            Min_lr=min_lr,
            optimizer_type=args.optimizer_type,
            momentum=args.momentum,
            lr_decay_type=args.lr_decay_type,
            save_period=args.save_period,
            save_dir=args.save_dir,
            log_dir=log_dir,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            mix_mode=args.mix_mode,
            mix_prob=args.mix_prob,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            num_workers=args.num_workers,
            num_train=num_train,
            num_val=num_val,
        )

        wanted_step = 1.5e4 if args.optimizer_type == "sgd" else 0.5e4
        total_step = num_train // args.unfreeze_batch_size * args.unfreeze_epoch
        if total_step <= wanted_step:
            if num_train // args.unfreeze_batch_size == 0:
                raise ValueError("Dataset is too small to continue training.")
            wanted_epoch = wanted_step // (num_train // args.unfreeze_batch_size) + 1
            print(f"\n\033[1;33;44m[Warning] {args.optimizer_type} is usually trained for at least {wanted_step:.0f} steps.\033[0m")
            print(
                f"\033[1;33;44m[Warning] Current setup: num_train={num_train}, "
                f"unfreeze_batch_size={args.unfreeze_batch_size}, epochs={args.unfreeze_epoch}, total_step={total_step}.\033[0m"
            )
            print(f"\033[1;33;44m[Warning] Suggested total epochs: {wanted_epoch}.\033[0m")

    unfreeze_flag = False
    if args.freeze_train:
        for param in model.backbone.parameters():
            param.requires_grad = False

    batch_size = args.freeze_batch_size if args.freeze_train else args.unfreeze_batch_size

    nbs = 16
    lr_limit_max = 5e-4 if args.optimizer_type == "adam" else 1e-1
    lr_limit_min = 3e-4 if args.optimizer_type == "adam" else 5e-4
    if args.backbone == "xception":
        lr_limit_max = 1e-4 if args.optimizer_type == "adam" else 1e-1
        lr_limit_min = 1e-4 if args.optimizer_type == "adam" else 5e-4
    init_lr_fit = min(max(batch_size / nbs * args.init_lr, lr_limit_min), lr_limit_max)
    min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = {
        "adam": optim.Adam(model.parameters(), init_lr_fit, betas=(args.momentum, 0.999), weight_decay=args.weight_decay),
        "sgd": optim.SGD(model.parameters(), init_lr_fit, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay),
    }[args.optimizer_type]

    lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, init_lr_fit, min_lr_fit, args.unfreeze_epoch)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("Dataset is too small to continue training.")

    train_dataset = DeeplabDataset(train_lines, input_shape, args.num_classes, True, dataset_path)
    val_dataset = DeeplabDataset(val_lines, input_shape, args.num_classes, False, dataset_path)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        batch_size = batch_size // ngpus_per_node
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    gen = DataLoader(
        train_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=deeplab_dataset_collate,
        sampler=train_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed),
    )
    gen_val = DataLoader(
        val_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=deeplab_dataset_collate,
        sampler=val_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed),
    )

    if local_rank == 0:
        eval_callback = EvalCallback(
            model,
            input_shape,
            args.num_classes,
            val_lines,
            dataset_path,
            log_dir,
            use_cuda,
            eval_flag=args.eval_flag,
            period=args.eval_period,
        )
    else:
        eval_callback = None

    for epoch in range(args.init_epoch, args.unfreeze_epoch):
        if epoch >= args.freeze_epoch and not unfreeze_flag and args.freeze_train:
            batch_size = args.unfreeze_batch_size

            init_lr_fit = min(max(batch_size / nbs * args.init_lr, lr_limit_min), lr_limit_max)
            min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, init_lr_fit, min_lr_fit, args.unfreeze_epoch)

            for param in model.backbone.parameters():
                param.requires_grad = True

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size
            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("Dataset is too small to continue training.")

            if args.distributed:
                batch_size = batch_size // ngpus_per_node

            gen = DataLoader(
                train_dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=deeplab_dataset_collate,
                sampler=train_sampler,
                worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed),
            )
            gen_val = DataLoader(
                val_dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=deeplab_dataset_collate,
                sampler=val_sampler,
                worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed),
            )
            unfreeze_flag = True

        if args.distributed:
            train_sampler.set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(
            model_train,
            model,
            loss_history,
            eval_callback,
            optimizer,
            epoch,
            epoch_step,
            epoch_step_val,
            gen,
            gen_val,
            args.unfreeze_epoch,
            use_cuda,
            args.dice_loss,
            args.focal_loss,
            cls_weights,
            args.num_classes,
            args.fp16,
            scaler,
            args.save_period,
            args.save_dir,
            local_rank,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            mix_mode=args.mix_mode,
            mix_prob=args.mix_prob,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
        )

        if args.distributed:
            dist.barrier()

    if local_rank == 0 and loss_history is not None:
        loss_history.writer.close()
