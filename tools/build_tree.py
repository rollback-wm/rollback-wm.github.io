"""Per-node WM rollout clips + tree.json for the interactive search tree on the project page.

Input: one tree folder from wm_reward_samples (index.json + rollout_*.mp4 + *_vs_real.mp4).
  - rollout_*.mp4: top 384 rows = the three imagined camera views (head | left wrist | right wrist)
  - rollout_*_vs_real.mp4: rows 384:768 = the recorded real robot, aligned in time
Each node covers one action chunk = 25 WM frames = 50 control steps (30 Hz).
Spine nodes are the real trajectory (clips from the real recording); every other node is imagined.

  python build_tree.py <tree_dir> <out_dir> [--sheet]
"""
import json
import os
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np

CHUNK, STEPS = 25, 50
TOP = 40                                  # rows dropped from every view (debug text lives there)
HW, HH = 384, 258                         # head view in the clip (512 x 344 source)
WW, WH = 192, 129                         # each wrist view (two stacked = HH)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def read(path, row0):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        band = f[row0:row0 + 384]
        out.append([band[TOP:, i * 512:(i + 1) * 512] for i in range(3)])
    return out


def compose(views):
    head = cv2.resize(views[0], (HW, HH), interpolation=cv2.INTER_AREA)
    wr = np.concatenate([cv2.resize(v, (WW, WH), interpolation=cv2.INTER_AREA) for v in views[1:]], 0)
    return np.concatenate([head, np.full((HH, 2, 3), 255, np.uint8), wr], 1)


def encode(frames, path, fps=15):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                          "-preset", "slow", "-crf", "25", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                          path], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()


def main():
    tree, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    idx = json.load(open(os.path.join(tree, "index.json")))
    nodes = {n["node_id"]: n for n in idx["nodes"]}
    R = idx["rollouts"]
    full = next(r for r in R if nodes[r["node_ids"][0]]["depth"] == 1)
    real = read(os.path.join(tree, full["vs_real"]), 384)

    clips, src = {}, {}
    for n in idx["nodes"]:
        d = n["depth"]
        if d == 0:
            clips[n["node_id"]] = [real[0]]
            src[n["node_id"]] = "real"
        elif n["is_spine"]:
            clips[n["node_id"]] = real[(d - 1) * CHUNK:d * CHUNK]
            src[n["node_id"]] = "real"
    for r in R:
        fr = None
        for k, nid in enumerate(r["node_ids"]):
            if nid in clips:
                continue
            fr = fr or read(os.path.join(tree, r["file"]), 0)
            clips[nid] = fr[k * CHUNK:(k + 1) * CHUNK]
            src[nid] = "imagined"

    if "--sheet" in sys.argv:
        # one row per rollout (root -> leaf), head view only, for the hallucination check
        rows = []
        for r in R + [{"node_ids": [n["node_id"] for n in idx["nodes"] if n["is_spine"]][1:], "file": "REAL"}]:
            seq = sum((clips[i] for i in r["node_ids"]), [])
            pick = np.linspace(0, len(seq) - 1, 10).astype(int)
            row = np.concatenate([cv2.resize(seq[i][0], (200, 141)) for i in pick], 1)
            cv2.putText(row, r["file"][:22], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            rows.append(row)
        cv2.imwrite(os.path.join(out, "sheet.jpg"), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return

    # outcome of each leaf: the real leaf uses the recorded result, imagined leaves the WM reward judge
    ok = {r["node_ids"][-1]: bool(r["success"]) for r in R}
    data = []
    for n in idx["nodes"]:
        nid = n["node_id"]
        name = "n%02d" % nid
        frames = [compose(v) for v in clips[nid]]
        if len(frames) > 1:
            encode(frames, os.path.join(out, name + ".mp4"))
        cv2.imwrite(os.path.join(out, name + ".jpg"), frames[-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
        leaf = not n["children_ids"]
        data.append({
            "id": nid, "parent": n["parent_id"], "depth": n["depth"],
            "real": src[nid] == "real", "leaf": leaf,
            "success": (bool(n["success"]) if n["is_spine"] else ok.get(nid)) if leaf else None,
            "steps": [max(0, (n["depth"] - 1) * STEPS), n["depth"] * STEPS],
            "clip": name + ".mp4" if len(frames) > 1 else None, "poster": name + ".jpg",
        })
    payload = {"task": idx["meta"]["task_name"], "episode": idx["meta"]["episode_id"], "nodes": data}
    json.dump(payload, open(os.path.join(out, "tree.json"), "w"), indent=1)
    # same data as a script, so the page also works when opened from disk (no fetch)
    with open(os.path.join(out, "tree.js"), "w") as f:
        f.write("window.ROLLBACK_TREE = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(len(data), "nodes ->", out)


if __name__ == "__main__":
    main()
