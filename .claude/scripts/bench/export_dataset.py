#!/usr/bin/env python3
"""Exports the benchmark renders as a training set for open models (detection, segmentation, depth, normals,
intrinsic decomposition, window segmentation), with the same train/val/test splits as the benchmark.

benchmark/dataset/<split>/
  images/<room>.png             rendered photo (camera model applied: exposure, tone curve, noise)
  empty/<room>.png              the same view without furniture (inpainting target)
  depth/<room>.png              uint16 millimeters, planar depth along the optical axis; 0 = sky or none
  normals/<room>.png            uint8 RGB, camera-frame (OpenCV) unit normals mapped from [-1, 1]
  albedo/<room>.png             uint8 sRGB albedo
  windows/<room>.png            uint8 mask, 255 = window (panel or sky seen through an opening)
  meta/<room>.json              intrinsics, camera pose, exposure, tone, noise, object boxes in the camera frame
  annotations.json              COCO instances (boxes, uncompressed RLE masks, names as categories)
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
import evaluate  # noqa: E402


def rle(mask):
    """COCO uncompressed RLE (column-major run lengths, starting with zeros)."""
    flat = mask.T.reshape(-1).astype(np.uint8)
    changes = np.flatnonzero(np.diff(np.concatenate([[0], flat, [1 - flat[-1]]])))
    counts = np.diff(np.concatenate([[0], changes])).tolist()
    return {"size": list(mask.shape), "counts": counts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--out", default="benchmark/dataset")
    args = parser.parse_args()
    for split in args.splits:
        out = os.path.join(REPO, args.out, split)
        for sub in ("images", "empty", "depth", "normals", "albedo", "windows", "meta"):
            os.makedirs(os.path.join(out, sub), exist_ok=True)
        coco = {"images": [], "annotations": [], "categories": []}
        categories = {}
        for k, view_dir in enumerate(evaluate.views(split)):
            room = view_dir.rstrip("/").split("/")[-2]
            gt = json.load(open(os.path.join(view_dir, "gt.json")))
            arrays = np.load(os.path.join(view_dir, "gt.npz"))
            H, W = arrays["depth"].shape
            R, t = evaluate.gt_camera(gt)
            Image.open(os.path.join(view_dir, "rgb.png")).save(os.path.join(out, "images", f"{room}.png"))
            if os.path.exists(os.path.join(view_dir, "empty.png")):
                Image.open(os.path.join(view_dir, "empty.png")).save(os.path.join(out, "empty", f"{room}.png"))
            depth = arrays["depth"].astype(np.float64)
            Image.fromarray(np.where(np.isfinite(depth), np.clip(np.round(depth * 1000), 0, 65535), 0).astype(np.uint16)).save(os.path.join(out, "depth", f"{room}.png"))
            normals = arrays["normal_world"].astype(np.float64) @ R.T  # world -> OpenCV camera
            normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-9)
            Image.fromarray(np.round((normals + 1) * 127.5).clip(0, 255).astype(np.uint8)).save(os.path.join(out, "normals", f"{room}.png"))
            albedo = np.clip(arrays["albedo"].astype(np.float64), 0, 1)
            Image.fromarray(np.round(np.where(albedo <= 0.0031308, albedo * 12.92, 1.055 * albedo ** (1 / 2.4) - 0.055) * 255).astype(np.uint8)).save(os.path.join(out, "albedo", f"{room}.png"))
            Image.fromarray((arrays["window"] * 255).astype(np.uint8)).save(os.path.join(out, "windows", f"{room}.png"))
            boxes = []
            for obj in gt["objects"]:
                mask = arrays["instance"] == obj["pass_index"]
                if mask.sum() < evaluate.TRUTH_PX:
                    continue
                name = obj["name"]
                if name not in categories:
                    categories[name] = len(categories) + 1
                    coco["categories"].append({"id": categories[name], "name": name})
                ys, xs = np.nonzero(mask)
                coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": k + 1, "category_id": categories[name],
                                            "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
                                            "area": int(mask.sum()), "iscrowd": 0, "segmentation": rle(mask)})
                boxes.append({"name": name, "center_cam": evaluate.to_cam(obj["center"], R, t).tolist(), "size": obj["size"],
                              "yaw_deg_world": obj["yaw_deg"], "support": obj["support"]})
            coco["images"].append({"id": k + 1, "file_name": f"images/{room}.png", "width": W, "height": H})
            json.dump({"room": room, "intrinsics_px": gt["camera"]["intrinsics_px"], "camera_height_m": gt["camera"]["height_m"],
                       "pitch_deg": gt["camera"]["pitch_deg"], "roll_deg": gt["camera"]["roll_deg"], "fov_x_deg": gt["camera"]["fov_x_deg"],
                       "world_from_camera_opencv": np.vstack([np.c_[R.T, -R.T @ t], [0, 0, 0, 1]]).tolist(),
                       "exposure": gt["exposure"], "tone": gt.get("tone"), "noise": gt.get("noise"), "objects": boxes,
                       "windows": gt["windows"], "source": "rendered with .claude/scripts/bench/render_scene.py, CC0 assets from Poly Haven"},
                      open(os.path.join(out, "meta", f"{room}.json"), "w"), indent=1)
        json.dump(coco, open(os.path.join(out, "annotations.json"), "w"))
        print(split, len(coco["images"]), "images", len(coco["annotations"]), "instances", len(categories), "categories")


if __name__ == "__main__":
    main()
