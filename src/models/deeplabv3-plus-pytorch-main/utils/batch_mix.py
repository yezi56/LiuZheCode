import numpy as np
import torch


def _sample_lambda(alpha):
    if alpha <= 0:
        return 1.0
    return float(np.random.beta(alpha, alpha))


def _rand_bbox(size, lam):
    _, _, h, w = size
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    return x1, y1, x2, y2


def apply_batch_mix(imgs, pngs, labels, mix_mode="none", mix_prob=0.0, mixup_alpha=0.4, cutmix_alpha=1.0):
    mix_mode = (mix_mode or "none").lower()
    if mix_mode in {"none", "off"} or mix_prob <= 0 or imgs.size(0) < 2:
        return imgs, pngs, labels

    if torch.rand(1, device=imgs.device).item() > mix_prob:
        return imgs, pngs, labels

    perm = torch.randperm(imgs.size(0), device=imgs.device)

    if mix_mode == "mixup":
        lam = _sample_lambda(mixup_alpha)
        lam_tensor = imgs.new_tensor(lam)
        imgs = imgs * lam_tensor + imgs[perm] * (1.0 - lam_tensor)
        labels = labels * lam_tensor + labels[perm] * (1.0 - lam_tensor)
        pngs = labels[..., :-1].argmax(dim=-1).long()
        ignore_mask = labels[..., -1] >= 0.5
        if ignore_mask.any():
            pngs = pngs.masked_fill(ignore_mask, labels.size(-1) - 1)
        return imgs, pngs, labels

    if mix_mode == "cutmix":
        lam = _sample_lambda(cutmix_alpha)
        x1, y1, x2, y2 = _rand_bbox(imgs.size(), lam)
        imgs[:, :, y1:y2, x1:x2] = imgs[perm, :, y1:y2, x1:x2]
        pngs[:, y1:y2, x1:x2] = pngs[perm, y1:y2, x1:x2]
        labels[:, y1:y2, x1:x2, :] = labels[perm, y1:y2, x1:x2, :]
        return imgs, pngs, labels

    raise ValueError(f"Unsupported mix_mode `{mix_mode}`. Use none, mixup, or cutmix.")
