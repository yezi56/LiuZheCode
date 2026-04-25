import argparse
import os
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training curves from DeepLabV3+ logs.")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=os.path.join("outputs", "grape_seg", "logs"),
        help="Base log directory or a specific loss_* run directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output image path. Defaults to training_curves.png inside the selected run directory.",
    )
    return parser.parse_args()


def read_float_lines(path: str) -> List[float]:
    if not os.path.exists(path):
        return []
    values: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values.append(float(line))
    return values


def resolve_run_dir(log_dir: str) -> str:
    if os.path.basename(log_dir).startswith("loss_"):
        return log_dir

    candidates = [
        os.path.join(log_dir, name)
        for name in os.listdir(log_dir)
        if name.startswith("loss_") and os.path.isdir(os.path.join(log_dir, name))
    ]
    if not candidates:
        raise FileNotFoundError(f"No loss_* run directory found under: {log_dir}")
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def plot_curves(run_dir: str, output_path: Optional[str] = None) -> str:
    train_loss = read_float_lines(os.path.join(run_dir, "epoch_loss.txt"))
    val_loss = read_float_lines(os.path.join(run_dir, "epoch_val_loss.txt"))
    miou = read_float_lines(os.path.join(run_dir, "epoch_miou.txt"))

    if not train_loss and not val_loss and not miou:
        raise FileNotFoundError(f"No curve text files found in: {run_dir}")

    has_miou = len(miou) > 0
    rows = 2 if has_miou else 1
    fig, axes = plt.subplots(rows, 1, figsize=(12, 5 * rows))
    if rows == 1:
        axes = [axes]

    if train_loss:
        axes[0].plot(range(1, len(train_loss) + 1), train_loss, color="red", linewidth=2, label="train loss")
    if val_loss:
        axes[0].plot(range(1, len(val_loss) + 1), val_loss, color="darkorange", linewidth=2, label="val loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    if train_loss or val_loss:
        axes[0].legend(loc="best")

    if has_miou:
        axes[1].plot(range(1, len(miou) + 1), miou, color="blue", linewidth=2, label="mIoU")
        axes[1].set_title("Validation mIoU")
        axes[1].set_xlabel("Evaluation Index")
        axes[1].set_ylabel("mIoU")
        axes[1].grid(True, linestyle="--", alpha=0.4)
        axes[1].legend(loc="best")

    fig.suptitle(os.path.basename(run_dir), fontsize=14)
    fig.tight_layout()

    if output_path is None:
        output_path = os.path.join(run_dir, "training_curves.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    run_dir = resolve_run_dir(args.log_dir)
    output_path = plot_curves(run_dir, args.output)
    print(f"Run directory: {run_dir}")
    print(f"Saved curve image: {output_path}")


if __name__ == "__main__":
    main()
