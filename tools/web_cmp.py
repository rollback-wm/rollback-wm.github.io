"""Frozen vs co-evolved WM, open-loop on the same real actions, for the project page.

  python web_cmp.py sheet success:523 fail:354 ...   -> web/sheet_<pool>_<ep>.jpg (8 time steps x 3 rows, head view)
  python web_cmp.py video success:523 ...            -> web/cmp_<pool>_<ep>.mp4 (Real | Frozen | Co-evolved, 3 views)
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

W = Path("/mnt/luojiayi/wm_fidelity_curling")
RUNS = {"frozen": W / "runs" / "frozen_b2b", "coev": W / "runs" / "coevolved_cycle0040"}
OUT = W / "web"
OUT.mkdir(exist_ok=True)


def load(pool, ep):
    fr = {n: np.load(RUNS[n] / pool / f"data_curling_ep{ep}" / "rollout_k0" / "frames.npz") for n in RUNS}
    gt = fr["frozen"]["gt"]
    fz, cv = fr["frozen"]["sim"], fr["coev"]["sim"]
    t = min(len(gt), len(fz), len(cv))
    return gt[:t], fz[:t], cv[:t]


def sheet(pool, ep):
    gt, fz, cv = load(pool, ep)
    cols = np.linspace(0, len(gt) - 1, 8).astype(int)
    rows = []
    for arr in (gt, fz, cv):
        rows.append(np.concatenate([cv2.resize(arr[i, 0], (256, 192)) for i in cols], 1))
    img = np.concatenate(rows, 0)
    for j, i in enumerate(cols):
        cv2.putText(img, str(i), (j * 256 + 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imwrite(str(OUT / f"sheet_{pool}_{ep}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def video(pool, ep):
    """one row per source; each row = head view (large) + the two wrist views stacked."""
    gt, fz, cv = load(pool, ep)
    hw, hh = 384, 288                     # head view size
    ww, wh = 192, 144                     # wrist view size (two stacked = hh)
    gap = 6
    def tile(fr):
        head = cv2.resize(fr[0], (hw, hh), interpolation=cv2.INTER_AREA)
        wr = np.concatenate([cv2.resize(fr[1], (ww, wh), interpolation=cv2.INTER_AREA),
                             cv2.resize(fr[2], (ww, wh), interpolation=cv2.INTER_AREA)], 0)
        return np.concatenate([head, np.full((hh, 2, 3), 255, np.uint8), wr], 1)
    tw = hw + 2 + ww
    H = hh
    Wd = 3 * tw + 2 * gap
    Wd += Wd % 2
    path = OUT / f"cmp_{pool}_{ep}.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{Wd}x{H}", "-r", "15", "-i", "-", "-c:v", "libx264", "-preset", "slow",
                          "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                         stdin=subprocess.PIPE)
    for i in range(len(gt)):
        canvas = np.full((H, Wd, 3), 255, np.uint8)
        for c, arr in enumerate((gt, fz, cv)):
            x = c * (tw + gap)
            canvas[:, x:x + tw] = tile(arr[i])
        p.stdin.write(canvas.tobytes())
    p.stdin.close()
    p.wait()
    # poster = last frame
    cv2.imwrite(str(path.with_suffix(".jpg")), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(path, len(gt), "frames", f"{Wd}x{H}")


if __name__ == "__main__":
    mode = sys.argv[1]
    for a in sys.argv[2:]:
        pool, ep = a.split(":")
        (sheet if mode == "sheet" else video)(pool, int(ep))
