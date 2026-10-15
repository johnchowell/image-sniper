#!/usr/bin/env python3
"""Novel-view scoring of a scene against rebuilt-room truth (render_scene.py --novel writes novel.json and
novel-<name>.png beside each view's gt.json).

  cameras  write worlds/<world>/output/scene/.novel-cameras.json: each true novel camera placed through the
           layout's source camera (relative pose from novel.json, its translation divided by the view's depth
           scale, so a wrong metric scale does not move the camera into or out of the room)
  score    PSNR and LPIPS (AlexNet) of <scene stem>-novel-<name>.png renders (render_check.py --cameras)
           against the truth, per view and mean; the truth is box-downsampled to the render size. PSNR favors a
           flat mean color over a texture with the wrong phase; LPIPS does not
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate  # noqa: E402

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # Blender camera axes <-> OpenCV camera axes


def view_dir(world):
    bench = json.load(open(os.path.join(evaluate.REPO, "worlds", world, "bench.json")))
    return os.path.join(evaluate.REPO, bench["view_dir"])


def depth_scale(world, layout_path, vdir):
    """True over predicted median depth on the layout grid, windows and sky excluded (as evaluate.py)."""
    layout = json.load(open(layout_path))
    index = int(os.path.basename(layout_path).split("-")[0])
    arrays = np.load(os.path.join(vdir, "gt.npz"))
    depth = arrays["depth"].astype(np.float64)
    H, W = depth.shape
    pred = np.asarray(Image.open(os.path.join(os.path.dirname(layout_path), f"{index}-layout-depth.png"))).astype(np.float64) / 1000
    step = layout["depth"]["grid_step_px"]
    ys = np.minimum(H - 1, np.arange(pred.shape[0]) * step + step // 2)
    xs = np.minimum(W - 1, np.arange(pred.shape[1]) * step + step // 2)
    truth = depth[ys][:, xs]
    valid = (pred > 0) & np.isfinite(truth) & (truth > 0) & (truth < 50) & ~arrays["window"][ys][:, xs]
    return float(np.median(truth[valid] / pred[valid]))


def cameras(args):
    for world in args.worlds:
        vdir = view_dir(world)
        novel = json.load(open(os.path.join(vdir, "novel.json")))
        layout_path = evaluate.latest(os.path.join(evaluate.REPO, "worlds", world, "output", "layout"), "[0-9]*-layout.json")
        layout = json.load(open(layout_path))
        scale = depth_scale(world, layout_path, vdir)
        source = np.array(layout["camera"]["matrix_world_from_camera_opencv"])
        views = []
        for view in novel["views"]:
            rel = FLIP @ np.array(view["relative_to_source"]) @ FLIP  # OpenCV axes
            rel[:3, 3] /= scale
            views.append({"name": view["name"], "matrix_world_from_camera_opencv": (source @ rel).tolist(), "fov_x_deg": novel["fov_x_deg"]})
        out = os.path.join(evaluate.REPO, "worlds", world, "output", "scene", ".novel-cameras.json")
        json.dump({"view_dir": os.path.relpath(vdir, evaluate.REPO), "scale_gt_over_pred": scale, "views": views}, open(out, "w"), indent=2)
        print(out)


def psnr(a, b):
    return 10 * math.log10(1 / max(float(np.mean((a - b) ** 2)), 1e-10))


def score(args):
    import lpips
    import torch
    net = lpips.LPIPS(net="alex", verbose=False)
    to_tensor = lambda a: torch.from_numpy(a.transpose(2, 0, 1)[None]).float() * 2 - 1
    rows = []
    for spec in args.scenes:
        world, stem = spec.split("=", 1) if "=" in spec else (spec, None)
        scene_dir = os.path.join(evaluate.REPO, "worlds", world, "output", "scene")
        if stem is None:
            renders = glob.glob(os.path.join(scene_dir, "[0-9]*-novel-*.png"))
            stem = max(renders, key=lambda p: int(os.path.basename(p).split("-")[0])).rsplit("-novel-", 1)[0]
        vdir = view_dir(world)
        novel = json.load(open(os.path.join(vdir, "novel.json")))
        per, perceptual = {}, {}
        for view in novel["views"]:
            render = np.asarray(Image.open(f"{stem}-novel-{view['name']}.png").convert("RGB")).astype(np.float64) / 255
            truth = Image.open(os.path.join(vdir, view["image"])).convert("RGB").resize(render.shape[1::-1], Image.BOX)
            truth = np.asarray(truth).astype(np.float64) / 255
            per[view["name"]] = round(psnr(render, truth), 3)
            with torch.no_grad():
                perceptual[view["name"]] = round(float(net(to_tensor(render), to_tensor(truth))), 4)
        rows.append({"world": world, "scene": os.path.relpath(stem, evaluate.REPO), "psnr_db": per,
                     "psnr_db_mean": round(float(np.mean(list(per.values()))), 3), "lpips": perceptual,
                     "lpips_mean": round(float(np.mean(list(perceptual.values()))), 4)})
        print(json.dumps(rows[-1]))
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("cameras")
    p.add_argument("--worlds", nargs="+", required=True)
    p = sub.add_parser("score")
    p.add_argument("--scenes", nargs="+", required=True, help="world or world=<scene stem> (default: the latest scene with novel renders)")
    p.add_argument("--out", help="write the rows to this JSON file")
    args = parser.parse_args()
    {"cameras": cameras, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
