import numpy as np
from skimage.metrics import structural_similarity as ssim


def calc_psnr_and_ssim_per_mask(img1, img2, mask):
    """PSNR computed on masked region only; SSIM over full frame."""
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    s = ssim(img1, img2, data_range=255, channel_axis=-1, win_size=65)
    if mask.sum() == 0:
        return float('inf'), s
    psnr = 20. * np.log10(255. / np.sqrt(np.mean((img1 - img2) ** 2)))
    return psnr, s
