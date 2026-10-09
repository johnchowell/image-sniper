#!/usr/bin/env python3
"""TripoSR geometry from the photo crop vs from the lighting-free albedo crop, against the true meshes.

For each whole object (not cut by the frame, at least 1% of the image) of a split: crops of the true instance
mask from the photo and from the light stage's albedo go through TripoSR; each mesh is compared with the true
Poly Haven mesh, both normalized to their bounding box (shape only: scale is the layout's job).

Pose: TripoSR's canonical frame has +Y up and +Z toward the input camera. The true mesh is put in that frame
from the render's object yaw and camera position ("expected pose"). A yaw search also finds the best-fitting
rotation; when it agrees with the expected pose, TripoSR keeps the pose seen in the photo, so placing the mesh
front toward the camera (object_mesh.py) is right and the layout box yaw must not be added on top.

Scores: Chamfer distance (mean of both directions, in bounding-box diagonals) and F-score at 5%.
Writes benchmark/results/mesh-ablation-<split>.json.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, ".claude", "scripts", "local"))
sys.path.insert(0, HERE)
import evaluate  # noqa: E402
from object_mesh import TSR_TO_CANONICAL, crop, largest_parts  # noqa: E402
from tsr.system import TSR  # noqa: E402

GLTF_TO_BLENDER = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)  # glTF +Y up -> Blender +Z up


def normalized(points):
    low, high = points.min(0), points.max(0)
    return (points - (low + high) / 2) / max(np.linalg.norm(high - low), 1e-9)


def chamfer(a, b, tau=0.05):
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    precision, recall = (da < tau).mean(), (db < tau).mean()
    return float((da.mean() + db.mean()) / 2), float(2 * precision * recall / max(precision + recall, 1e-9))


def rot_y(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def true_points(obj, camera, n):
    """True mesh surface points in TripoSR's canonical frame for this view (expected pose)."""
    mesh = trimesh.load(os.path.join(REPO, obj["model_path"]), force="mesh")
    pts = trimesh.sample.sample_surface(mesh, n, seed=0)[0] @ GLTF_TO_BLENDER.T
    yaw = math.radians(obj["yaw_deg"])
    pts = pts @ np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]).T
    w = np.array(camera[:2]) - np.array(obj["center"][:2])
    w /= np.linalg.norm(w)
    basis = np.array([[-w[1], w[0], 0], [0, 0, 1], [w[0], w[1], 0]])  # rows: canonical X (right), Y (up), Z (toward camera)
    return normalized(pts @ basis.T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-objects", type=int, default=24)
    parser.add_argument("--min-fraction", type=float, default=0.01)
    parser.add_argument("--points", type=int, default=4000)
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count())
    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)

    rows = []
    for view_dir in evaluate.views(args.split):
        gt = json.load(open(os.path.join(view_dir, "gt.json")))
        arrays = np.load(os.path.join(view_dir, "gt.npz"))
        world = os.path.join(REPO, "worlds", evaluate.world_name(view_dir))
        light = evaluate.latest(os.path.join(world, "output", "light"), "[0-9]*-light.json")
        if not light:
            continue
        photo = Image.open(os.path.join(view_dir, "rgb.png")).convert("RGB")
        albedo = Image.open(json.load(open(light))["files"]["albedo"]).convert("RGB").resize(photo.size, Image.BICUBIC)
        R, t = evaluate.gt_camera(gt)
        K = gt["camera"]["intrinsics_px"]
        H, W = arrays["instance"].shape
        for obj in gt["objects"]:
            if obj.get("class", "furniture") != "furniture" or not obj.get("model_path"):
                continue
            mask = arrays["instance"] == obj["pass_index"]
            if mask.mean() < args.min_fraction:
                continue
            c, s = math.cos(math.radians(obj["yaw_deg"])), math.sin(math.radians(obj["yaw_deg"]))
            sx, sy, sz = obj["size"]
            corners = np.array([[x, y, z] for x in (-sx / 2, sx / 2) for y in (-sy / 2, sy / 2) for z in (-sz / 2, sz / 2)])
            corners = evaluate.to_cam(corners @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + np.array(obj["center"]), R, t)
            px = evaluate.project(corners, K)
            if (corners[:, 2] <= 0.05).any() or ((px < 0) | (px > [W, H])).any():
                continue  # cut by the frame: TripoSR sees a partial object
            truth = true_points(obj, gt["camera"]["position"], args.points)
            row = {"view": os.path.basename(world), "object": obj["id"], "name": obj["name"], "pixel_fraction": float(mask.mean())}
            for reference, image in (("photo", photo), ("albedo", albedo)):
                started = time.time()
                with torch.no_grad():
                    codes = model([crop(image, mask, 0.85)], device="cpu")
                    mesh = largest_parts(model.extract_mesh(codes, False, resolution=256)[0])
                pred = normalized(trimesh.sample.sample_surface(mesh, args.points, seed=0)[0] @ TSR_TO_CANONICAL.T)
                expected = chamfer(pred, truth)
                yaws = [(chamfer(pred[::4] @ rot_y(math.radians(d)).T, truth[::4])[0], d) for d in range(-180, 180, 5)]
                best_cd, best_yaw = min(yaws)
                row[reference] = {"chamfer_expected_pose": expected[0], "fscore_expected_pose": expected[1],
                                  "chamfer_best_yaw": chamfer(pred @ rot_y(math.radians(best_yaw)).T, truth)[0],
                                  "best_yaw_offset_deg": best_yaw, "seconds": round(time.time() - started, 1)}
            rows.append(row)
            print(json.dumps(row), flush=True)
            if len(rows) >= args.max_objects:
                break
        if len(rows) >= args.max_objects:
            break

    rng = np.random.default_rng(0)
    diff = np.array([r["albedo"]["chamfer_expected_pose"] - r["photo"]["chamfer_expected_pose"] for r in rows])
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    offsets = np.array([abs(r["photo"]["best_yaw_offset_deg"]) for r in rows])
    summary = {
        "objects": len(rows),
        "photo_chamfer_mean": float(np.mean([r["photo"]["chamfer_expected_pose"] for r in rows])),
        "albedo_chamfer_mean": float(np.mean([r["albedo"]["chamfer_expected_pose"] for r in rows])),
        "photo_fscore_mean": float(np.mean([r["photo"]["fscore_expected_pose"] for r in rows])),
        "albedo_fscore_mean": float(np.mean([r["albedo"]["fscore_expected_pose"] for r in rows])),
        "albedo_minus_photo_chamfer_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "photo_best_yaw_offset_deg_median": float(np.median(offsets)),
        "photo_best_yaw_within_20deg": float((offsets <= 20).mean()),
        "photo_best_yaw_within_20deg_of_180": float((np.abs(offsets - 180) <= 20).mean()),
    }
    json.dump({"split": args.split, "summary": summary, "objects": rows},
              open(os.path.join(REPO, "benchmark", "results", f"mesh-ablation-{args.split}.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
