import numpy as np
import torch


def pil_list_to_tensor(imgs):
    """List of T PIL images → (T, C, H, W) float32 tensor in [0, 1]."""
    arr = np.stack([np.array(img) for img in imgs])  # (T, H, W) or (T, H, W, C)
    t = torch.from_numpy(arr.copy()).float() / 255.0
    if t.dim() == 3:          # grayscale: (T, H, W) → (T, 1, H, W)
        t = t.unsqueeze(1)
    else:                     # RGB: (T, H, W, C) → (T, C, H, W)
        t = t.permute(0, 3, 1, 2)
    return t.contiguous()
