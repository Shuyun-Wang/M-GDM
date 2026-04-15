"""
MGDM Inference Script
Blind Bitstream-corrupted Video Recovery via Metadata-guided Diffusion Model
CVPR 2025
"""
import os
import argparse
from datetime import datetime
import logging

import torch
import numpy as np
from einops import rearrange
from PIL import Image
import imageio
from torch.utils.data import DataLoader
from accelerate.utils import set_seed
from diffusers.models import AutoencoderKL

from core.dataset import InferenceDataset
from model.unet_metadata import MGDMModel
from model.simple_unet import UNet
from core.pipeline import MGDMPipeline


def setup_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def load_prm(ckpt_path, device):
    net = UNet(upscale=1, in_chans=3, img_size=126, window_size=7, img_range=255.,
               depths=[6,6,6,6,6,6], embed_dim=180, num_heads=[6,6,6,6,6,6],
               mlp_ratio=2, upsampler='', resi_connection='1conv')
    net.load_state_dict(torch.load(os.path.join(ckpt_path, 'unet.pth'), map_location='cpu', weights_only=False))
    return net.to(device, torch.float32).eval()


def prm_forward(net, frames_bt3hw, window_size=7):
    _, _, H, W = frames_bt3hw.shape
    h_pad = (H // window_size + 1) * window_size - H
    w_pad = (W // window_size + 1) * window_size - W
    pad = torch.cat([frames_bt3hw, torch.flip(frames_bt3hw, [2])], 2)[:, :, :H + h_pad, :]
    pad = torch.cat([pad, torch.flip(pad, [3])], 3)[:, :, :, :W + w_pad]
    chunks = [net(pad[i:i+32]) for i in range(0, len(frames_bt3hw), 32)]
    return torch.cat(chunks, dim=0)[..., :H, :W]


@torch.no_grad()
def run_inference(args, logger):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    generator = torch.Generator(device)

    unet = MGDMModel.from_pretrained(args.ckpt_stage1, subfolder='unet',
                                     mask_decoder_type="Metadata_MaskDecoder",
                                     use_metadata_as_prompt=True).to(device, torch.float32)
    prm = load_prm(args.ckpt_prm, device)
    vae = AutoencoderKL.from_pretrained(args.ckpt_stage1, subfolder='vae').to(device, torch.float32)
    pipe = MGDMPipeline.from_pretrained(args.ckpt_stage1, unet=unet, vae=vae, torch_dtype=torch.float32).to(device)

    loader = DataLoader(InferenceDataset(args), batch_size=1, shuffle=False, num_workers=1)

    for items in loader:
        corrupts, mvs, ft, video_name = items
        B, T = corrupts.shape[:2]
        vid = video_name[0]
        logger.info(f'Processing {vid} ({T} frames)')

        corrupts = corrupts.to(device)

        # Stage 1: diffusion
        corrupts_flat     = rearrange(corrupts, "b f c h w -> (b f) c h w")
        condition_latents = pipe.vae.encode(corrupts_flat).latent_dist.sample()
        condition_latents = rearrange(condition_latents, "(b f) c h w -> b c f h w", f=T) * 0.18215
        out1 = pipe(args.expr, mvs, ft, img_condition=condition_latents,
                    num_inference_steps=50, guidance_scale=args.gs,
                    image_guidance_scale=args.igs, generator=generator,
                    return_attn_map=False, high_resolution_masks=True)

        stage1  = rearrange(out1.videos, "b c t h w -> b t c h w").to(device)
        masks   = (out1.mask > 0.9).float().repeat(1,3,1,1,1).permute(0,2,1,3,4).to(device)
        corrupt = ((corrupts / 2) + 0.5).to(device)
        torch.cuda.empty_cache()

        # Stage 2: PRM on composited stage1, then alpha blend
        stage1_comp = stage1 * masks + corrupt * (1 - masks)
        refined     = prm_forward(prm, rearrange(stage1_comp, "b t c h w -> (b t) c h w"))
        refined     = rearrange(refined, "(b t) c h w -> b t c h w", b=B, t=T)
        output      = (args.alpha * stage1 + (1 - args.alpha) * refined) * masks + corrupt * (1 - masks)
        output      = output.clamp(0, 1)

        comp_frames = (output[0].cpu().permute(0,2,3,1).numpy() * 255).astype(np.uint8)

        img_dir = os.path.join(args.output_dir, "images", vid)
        os.makedirs(img_dir, exist_ok=True)
        for i, frame in enumerate(comp_frames):
            Image.fromarray(frame).save(os.path.join(img_dir, f"frame_{i:08d}.png"))

        gif_dir = os.path.join(args.output_dir, "gifs")
        os.makedirs(gif_dir, exist_ok=True)
        imageio.mimsave(os.path.join(gif_dir, f"{vid}.gif"), comp_frames, fps=10)

        logger.info(f'Done {vid}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   type=str, default="datasets")
    parser.add_argument("--ckpt_stage1", type=str, default="checkpoints/stage1")
    parser.add_argument("--ckpt_prm",    type=str, default="checkpoints/prm")
    parser.add_argument("--rep_txt",     type=str, default="checkpoints/stage1/unet/rep.txt")
    parser.add_argument("--json",        type=str, default="quick_test.json")
    parser.add_argument("--w",           type=int, default=448)
    parser.add_argument("--h",           type=int, default=256)
    parser.add_argument("--expr",        type=str, default='')
    parser.add_argument("--gs",          type=float, default=0)
    parser.add_argument("--igs",         type=float, default=0)
    parser.add_argument("--alpha",       type=float, default=0.4)
    parser.add_argument("--output_dir",  type=str, default="results/inference")
    parser.add_argument("--seed",        type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%m-%d_%H-%M")
    logger = setup_logger(os.path.join(args.output_dir, f"inference_{ts}.log"))
    run_inference(args, logger)
