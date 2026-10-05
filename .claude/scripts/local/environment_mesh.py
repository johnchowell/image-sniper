#!/usr/bin/env python3
"""Environment mesh of the empty room: MoGe-2 depth of the clean plate, textured with the plate, in the layout frame.

The plate's depth is predicted with the layout camera's field of view, then scaled to the original photo's
depth using pixels that the plate did not change (outside the removal mask).
Writes worlds/<world>/output/scene/N-scene-environment.glb and its request JSON.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import trimesh
from PIL import Image

from common import camera_matrix, load_layout, next_index, world_path, write_request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--grid", type=int, default=400, help="mesh grid width in vertices")
    parser.add_argument("--edge", type=float, default=0.04, help="max relative depth jump inside one triangle")
    args = parser.parse_args()

    layout_index, layout_path, layout = load_layout(args.world)
    world_from_cam, K, (width, height) = camera_matrix(layout)
    source_dir = world_path(args.world, "source")
    plates = [p for p in os.listdir(source_dir) if p.endswith("-plate.png") and not p.startswith(".")]
    if not plates:
        raise SystemExit("No clean plate in source/. Run make_plate.py first.")
    plate_path = os.path.join(source_dir, max(plates, key=lambda p: int(p.split("-")[0])))
    plate = Image.open(plate_path).convert("RGB")
    if plate.size != (width, height):
        raise SystemExit(f"Plate size {plate.size} differs from the layout camera {width}x{height}.")

    from moge.model.v2 import MoGeModel

    started = time.time()
    torch.set_num_threads(os.cpu_count())
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").eval()
    with torch.no_grad():
        out = model.infer(torch.tensor(np.asarray(plate) / 255.0, dtype=torch.float32).permute(2, 0, 1),
                          fov_x=float(layout["camera"]["fov_x_deg"]), use_fp16=False)
    points = out["points"].numpy().astype(np.float64)
    valid = out["mask"].numpy().astype(bool) & np.isfinite(points).all(-1)

    # Scale to the original photo's metric depth on unchanged pixels.
    depth_png = os.path.join(world_path(args.world, "output", "layout"), f"{layout_index}-layout-depth.png")
    original = np.asarray(Image.open(depth_png)).astype(np.float64) / 1000.0
    gh, gw = original.shape
    step = layout["depth"]["grid_step_px"]
    ys = np.minimum(height - 1, np.arange(gh) * step + step // 2)
    xs = np.minimum(width - 1, np.arange(gw) * step + step // 2)
    plate_z = points[ys][:, xs, 2]
    plate_request = os.path.join(source_dir, "." + os.path.basename(plate_path)[:-4] + "-request.json")
    mask_file = json.load(open(plate_request)).get("mask_file") if os.path.exists(plate_request) else None
    removed = np.asarray(Image.open(mask_file)) > 0 if mask_file and os.path.exists(mask_file) else np.zeros((height, width), bool)
    keep = (original > 0) & valid[ys][:, xs] & ~removed[ys][:, xs] & (plate_z > 0)
    ratios = original[keep] / plate_z[keep]
    scale = float(np.median(ratios))
    spread = float(np.percentile(ratios, 90) / np.percentile(ratios, 10))
    points *= scale

    # Grid mesh with depth-discontinuity cuts.
    stride = max(1, -(-width // args.grid))
    P = points[::stride, ::stride]
    V = valid[::stride, ::stride]
    rows, cols = V.shape
    index = np.arange(rows * cols).reshape(rows, cols)
    z = P[..., 2]
    quads = []
    for a, b, c in (((0, 0), (1, 0), (0, 1)), ((1, 0), (1, 1), (0, 1))):
        ia = index[a[0]:rows - 1 + a[0], a[1]:cols - 1 + a[1]]
        ib = index[b[0]:rows - 1 + b[0], b[1]:cols - 1 + b[1]]
        ic = index[c[0]:rows - 1 + c[0], c[1]:cols - 1 + c[1]]
        tri = np.stack([ia, ic, ib], -1).reshape(-1, 3)
        zs = z.reshape(-1)[tri]
        ok = V.reshape(-1)[tri].all(1) & (zs.min(1) > 0) & (zs.max(1) / np.maximum(zs.min(1), 1e-6) < 1 + args.edge)
        quads.append(tri[ok])
    faces = np.concatenate(quads)
    used = np.unique(faces)
    remap = -np.ones(rows * cols, dtype=np.int64)
    remap[used] = np.arange(len(used))
    cam_points = P.reshape(-1, 3)[used]
    homogeneous = np.c_[cam_points, np.ones(len(cam_points))]
    vertices = (homogeneous @ world_from_cam.T)[:, :3]
    r, c = np.divmod(used, cols)
    uv = np.c_[(c * stride + 0.5) / width, 1 - (r * stride + 0.5) / height]

    texture = plate if max(plate.size) <= 2048 else plate.resize((2048, int(2048 * height / width)))
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=texture, metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=remap[faces], visual=trimesh.visual.TextureVisuals(uv=uv, material=material), process=False)

    scene_dir = world_path(args.world, "output", "scene")
    n = next_index(scene_dir)
    out_path = os.path.join(scene_dir, f"{n}-scene-environment.glb")
    mesh.export(out_path)
    summary = {
        "environment": out_path,
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "depth_scale": round(scale, 4),
        "scale_spread_p90_p10": round(spread, 3),
        "scale_samples": int(keep.sum()),
        "seconds": round(time.time() - started, 1),
    }
    write_request(os.path.join(scene_dir, f".{n}-scene-environment-request.json"), {
        "kind": "environment",
        "provider": "local/moge-2-vitl-normal",
        "endpoint": "local/moge-2-vitl-normal",
        "index": n,
        "status": "completed",
        "input_files": [plate_path],
        "layout": layout_path,
        "frame": "layout (meters, +Y up, camera forward -Z)",
        "output_files": [out_path],
        "result": summary,
    })
    print(summary)


if __name__ == "__main__":
    main()
