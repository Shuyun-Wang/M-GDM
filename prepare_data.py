"""
Prepare motion vectors (MVs) and frame-type labels (I/P/B) from H.264 bitstreams.

Prerequisites:
    pip install motion-vector-extractor

Input layout:
    <data_root>/
    └── BSC_h264/{video}.h264    # bitstream-corrupted H.264 files

Output:
    <data_root>/
    ├── BSC_mvs/{video}/{00000..N}.npz   # per-frame flow array, shape H×W×4 (past_xy, future_xy)
    ├── BSC_JPEGImages/{video}/{00000..N}.jpg   # decoded frames (via ffmpeg)
    └── frame_type.npy                   # dict {video: ['I','P','B', ...]}

Usage:
    python prepare_data.py --data_root /path/to/data
    python prepare_data.py --data_root /path/to/data --workers 8
"""

import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
from mvextractor.videocap import VideoCap


# ---------------------------------------------------------------------------
# Motion-vector helpers  (logic identical to original extract_mv.py)
# ---------------------------------------------------------------------------

def mvtoflow_1(mvs_0, h, w):
    flow = np.array([
        mvs_0[:, 5], mvs_0[:, 6],
        mvs_0[:, 3] - mvs_0[:, 5],
        mvs_0[:, 4] - mvs_0[:, 6],
        mvs_0[:, 1], mvs_0[:, 2]
    ]).T

    vis_mat = np.zeros((h, w, 2))
    for start_x, start_y, delta_x, delta_y, blk_w, blk_h in flow:
        x, y = int(start_x - blk_w / 2), int(start_y - blk_h / 2)
        vis_mat[max(0, y):min(h, y + int(blk_h)),
                max(0, x):min(w, x + int(blk_w))] = (delta_x, delta_y)
    return vis_mat


def get_flow_from_mv_1(mvs_0, h, w):
    past, future = [], []
    for k in mvs_0:
        if k[0] == -1:
            past.append(k)
        else:
            future.append(k)

    past_flow   = np.zeros((h, w, 2)) if past   == [] else mvtoflow_1(np.array(past),   h, w)
    future_flow = np.zeros((h, w, 2)) if future == [] else mvtoflow_1(np.array(future), h, w)
    return np.concatenate((past_flow, future_flow), axis=2)


# ---------------------------------------------------------------------------
# Per-video extraction
# ---------------------------------------------------------------------------

def extract_mv_and_ft(video_path, out_mv_dir):
    """Extract MVs and frame types from one H.264 file.

    Returns:
        list[str]: per-frame frame-type labels ('I', 'P', 'B')
    """
    os.makedirs(out_mv_dir, exist_ok=True)
    cap = VideoCap()
    if not cap.open(video_path):
        raise RuntimeError(f"Cannot open {video_path}")

    frame_types = []
    idx = 0
    while True:
        ret, frame, motion_vectors, frame_type, timestamp = cap.read()
        if ret != True:
            break
        h, w, c = frame.shape
        flow = get_flow_from_mv_1(motion_vectors, h, w).astype(np.float32)
        np.savez_compressed(os.path.join(out_mv_dir, f"{idx:05d}.npz"), flow)
        if frame_type == 'I':
            frame_types.append('I')
        elif frame_type == 'P':
            frame_types.append('P')
        else:
            frame_types.append('B')
        idx += 1

    return frame_types


def decode_frames(video_path, out_jpg_dir):
    """Decode H.264 to JPEG frames using ffmpeg."""
    os.makedirs(out_jpg_dir, exist_ok=True)
    cmd = (f'ffmpeg -i "{video_path}" -start_number 0 -qscale:v 2 '
           f'"{out_jpg_dir}/%05d.jpg" -loglevel error')
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}")


def _process_one(args):
    video_path, out_mv_dir, out_jpg_dir, video_name = args
    try:
        ft = extract_mv_and_ft(video_path, out_mv_dir)
        decode_frames(video_path, out_jpg_dir)
        print(f"[OK] {video_name} ({len(ft)} frames)")
        return video_name, ft
    except Exception as e:
        print(f"[FAIL] {video_name}: {e}")
        return video_name, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing BSC_h264/")
    parser.add_argument("--workers",   type=int, default=4,
                        help="Number of parallel workers")
    args = parser.parse_args()

    h264_dir = os.path.join(args.data_root, "BSC_h264")
    video_paths = sorted(glob.glob(os.path.join(h264_dir, "*.h264")))
    if not video_paths:
        print(f"No .h264 files found in {h264_dir}")
        return

    tasks = []
    for vp in video_paths:
        name = os.path.splitext(os.path.basename(vp))[0]
        tasks.append((
            vp,
            os.path.join(args.data_root, "BSC_mvs",        name),
            os.path.join(args.data_root, "BSC_JPEGImages",  name),
            name,
        ))

    with Pool(processes=args.workers) as pool:
        results = pool.map(_process_one, tasks)

    ft_dict = {name: ft for name, ft in results if ft is not None}
    out_npy = os.path.join(args.data_root, "frame_type.npy")
    np.save(out_npy, ft_dict)
    print(f"\nSaved frame_type.npy with {len(ft_dict)} videos → {out_npy}")


if __name__ == "__main__":
    main()
