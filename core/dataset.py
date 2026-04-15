import os
import json
import cv2
from PIL import Image
import numpy as np
import torch
from core.utils import pil_list_to_tensor


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.size = (args.w, args.h)

        json_file = os.path.join(args.data_root, args.json)
        with open(json_file, 'r') as f:
            self.video_dict = json.load(f)

        self.video_names = list(self.video_dict.keys())
        ft_path = os.path.join(args.data_root, 'frame_type.npy')
        self.ft = np.load(ft_path, allow_pickle=True).item()
        self.IBP_rep = np.loadtxt(args.rep_txt, dtype="float", delimiter=',')

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        return self.load_item(index)

    def load_item(self, index):
        video_name = self.video_names[index]
        ref_index  = list(range(self.video_dict[video_name]))

        f_type = [self.ft[video_name][i] for i in ref_index]
        type_to_idx = {'I': 0, 'B': 1, 'P': 2}
        IBP_rep = torch.from_numpy(
            np.array([self.IBP_rep[type_to_idx[ft]] for ft in f_type])
        )

        frames, corrupts, gt_masks, mvs = [], [], [], []
        for idx in ref_index:
            # GT frame
            gt_img = Image.open(
                os.path.join(self.args.data_root,
                             'GT_JPEGImages', video_name, f'{idx:05d}.jpg')
            ).convert('RGB').resize(self.size)
            frames.append(gt_img)

            # Corrupted frame
            corr_img = Image.open(
                os.path.join(self.args.data_root,
                             'BSC_JPEGImages', video_name, f'{idx:05d}.jpg')
            ).convert('RGB').resize(self.size)
            corrupts.append(corr_img)

            # GT mask
            m = self._load_mask(
                os.path.join(self.args.data_root,
                             'GT_masks', video_name, f'{idx:05d}.png')
            )
            gt_masks.append(m)

            # Motion vectors
            mv = np.load(
                os.path.join(self.args.data_root, 'BSC_mvs',
                             video_name, f'{idx:05d}.npz')
            )['arr_0'].astype(np.float32)
            mvs.append(cv2.resize(mv, self.size, interpolation=cv2.INTER_NEAREST))

        frames_PIL = [np.array(f).astype(np.uint8) for f in frames]

        return (
            frames_PIL,
            pil_list_to_tensor(corrupts) * 2.0 - 1.0,
            pil_list_to_tensor(gt_masks),
            torch.stack([torch.from_numpy(mv).float() for mv in mvs]),
            IBP_rep,
            video_name,
        )

    def _load_mask(self, path):
        mask = np.array(
            Image.open(path).resize(self.size, Image.NEAREST).convert('L')
        )
        m = cv2.dilate(
            (mask > 0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
            iterations=4,
        )
        return Image.fromarray(m * 255)


class InferenceDataset(torch.utils.data.Dataset):
    """Inference-only dataset: no ground truth or masks required.

    Expected layout under data_root:
        BSC_JPEGImages/{video}/{00000..N}.jpg
        BSC_mvs/{video}/{00000..N}.npz
        frame_type.npy
        <json>                              # {video_name: num_frames}
    """

    def __init__(self, args):
        self.args = args
        self.size = (args.w, args.h)

        json_file = os.path.join(args.data_root, args.json)
        with open(json_file, 'r') as f:
            self.video_dict = json.load(f)

        self.video_names = list(self.video_dict.keys())
        ft_path = os.path.join(args.data_root, 'frame_type.npy')
        self.ft = np.load(ft_path, allow_pickle=True).item()
        self.IBP_rep = np.loadtxt(args.rep_txt, dtype="float", delimiter=',')

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        video_name = self.video_names[index]
        ref_index  = list(range(self.video_dict[video_name]))

        f_type = [self.ft[video_name][i] for i in ref_index]
        type_to_idx = {'I': 0, 'B': 1, 'P': 2}
        IBP_rep = torch.from_numpy(
            np.array([self.IBP_rep[type_to_idx[ft]] for ft in f_type])
        )

        corrupts, mvs = [], []
        for idx in ref_index:
            corr_img = Image.open(
                os.path.join(self.args.data_root,
                             'BSC_JPEGImages', video_name, f'{idx:05d}.jpg')
            ).convert('RGB').resize(self.size)
            corrupts.append(corr_img)

            mv = np.load(
                os.path.join(self.args.data_root, 'BSC_mvs',
                             video_name, f'{idx:05d}.npz')
            )['arr_0'].astype(np.float32)
            mvs.append(cv2.resize(mv, self.size, interpolation=cv2.INTER_NEAREST))

        return (
            pil_list_to_tensor(corrupts) * 2.0 - 1.0,
            torch.stack([torch.from_numpy(mv).float() for mv in mvs]),
            IBP_rep,
            video_name,
        )
