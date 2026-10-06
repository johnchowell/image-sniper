#!/usr/bin/env python3
"""Scores the IMAGE-BLASTER pipeline against rendered ground truth.

  prepare  make worlds/bench-<split>-<room>-<view>/ from each render (source photo + object.json per true object)
  run      run the light and layout stages on every bench world (local models, see local_pass.py)
  score    compare outputs with gt.npz/gt.json; writes benchmark/results/<name>.json

All geometry is compared in the OpenCV camera frame (x right, y down, z forward) of the rendered camera.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LOCAL = os.path.join(REPO, ".claude", "scripts", "local")
LUMA = np.array([0.2126, 0.7152, 0.0722])
BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def views(split):
    return sorted(glob.glob(os.path.join(REPO, "benchmark", "renders", split, "room-*", "view-*")))


def world_name(view_dir):
    room, view = view_dir.rstrip("/").split("/")[-2:]
    split = view_dir.rstrip("/").split("/")[-3]
    return f"bench-{split}-{room}-{view}"


def prepare(split):
    for view_dir in views(split):
        gt = json.load(open(os.path.join(view_dir, "gt.json")))
        world = os.path.join(REPO, "worlds", world_name(view_dir))
        shutil.rmtree(world, ignore_errors=True)
        os.makedirs(os.path.join(world, "source"))
        shutil.copy(os.path.join(view_dir, "rgb.png"), os.path.join(world, "source", "0-render.png"))
        json.dump({"schema_version": 1, "slug": os.path.basename(world), "display_name": os.path.basename(world)}, open(os.path.join(world, "project.json"), "w"))
        instance = np.load(os.path.join(view_dir, "gt.npz"))["instance"]
        for obj in gt["objects"]:
            if (instance == obj["pass_index"]).mean() < 0.002:  # not visible enough to confirm in the photo
                continue
            object_id = slug(obj["id"])
            os.makedirs(os.path.join(world, "output", object_id))
            json.dump({"schema_version": 1, "world": os.path.basename(world), "object": {
                "id": object_id, "name": obj["name"], "description": obj["name"], "count_estimate": 1,
                "generate_as_3d_object": True, "working_dir": f"worlds/{os.path.basename(world)}/output/{object_id}"}},
                open(os.path.join(world, "output", object_id, "object.json"), "w"), indent=2)
        print(os.path.basename(world))


def run(split, python, skip_light):
    for view_dir in views(split):
        world = world_name(view_dir)
        command = [python, os.path.join(LOCAL, "local_pass.py"), "--world", world,
                   "--skip", "plate", "--skip", "environment", "--skip", "objects", "--skip", "scene"]
        if skip_light:
            command += ["--skip", "light"]
        process = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
        status = "ok" if process.returncode == 0 else "failed: " + (process.stderr or process.stdout).strip().splitlines()[-1]
        print(world, status, flush=True)


def latest(directory, pattern):
    files = glob.glob(os.path.join(directory, pattern))
    return max(files, key=lambda f: int(os.path.basename(f).split("-")[0])) if files else None


def gt_camera(gt):
    """Blender camera-to-world (4x4) and world-to-OpenCV-camera rotation/translation."""
    M = np.array(gt["camera"]["matrix_world"])
    R_wc = M[:3, :3] @ BLENDER_TO_OPENCV  # OpenCV camera axes in world
    return R_wc.T, -R_wc.T @ M[:3, 3]


def to_cam(points, R, t):
    return np.asarray(points) @ R.T + t


def plane_in_cam(point, normal, R, t):
    n = R @ np.asarray(normal, float)
    d = -n @ to_cam(point, R, t)
    return (n, d) if d >= 0 else (-n, -d)  # oriented with the camera on the positive side


def pred_cam(layout):
    W_from_C = np.array(layout["camera"]["matrix_world_from_camera_opencv"])
    R, t = W_from_C[:3, :3], W_from_C[:3, 3]
    return R.T, -R.T @ t  # layout -> camera


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def si_rmse(pred, gt, mask):
    p, g = pred[mask].reshape(-1), gt[mask].reshape(-1)
    s = (p @ g) / max(p @ p, 1e-12)
    return float(np.sqrt(np.mean((s * p - g) ** 2)) / max(np.sqrt(np.mean(g ** 2)), 1e-12))


def score_view(view_dir):
    gt = json.load(open(os.path.join(view_dir, "gt.json")))
    arrays = np.load(os.path.join(view_dir, "gt.npz"))
    world = os.path.join(REPO, "worlds", world_name(view_dir))
    layout_path = latest(os.path.join(world, "output", "layout"), "[0-9]*-layout.json")
    if not layout_path:
        return {"view": world_name(view_dir), "status": "no_layout"}
    layout = json.load(open(layout_path))
    index = int(os.path.basename(layout_path).split("-")[0])
    R_gt, t_gt = gt_camera(gt)
    R_pc, t_pc = pred_cam(layout)
    H, W = arrays["depth"].shape
    out = {"view": world_name(view_dir)}

    # Camera.
    cam = layout["camera"]
    out["camera"] = {
        "height_err_m": cam["height_m"] - gt["camera"]["height_m"],
        "pitch_err_deg": cam["pitch_deg"] - gt["camera"]["pitch_deg"],
        "roll_err_deg": cam["roll_deg"] - gt["camera"]["roll_deg"],
        "fov_err_deg": cam["fov_x_deg"] - gt["camera"]["fov_x_deg"],
    }

    # Depth (metric and median-scaled) from the layout depth PNG (mm, grid resolution).
    pred_depth = np.asarray(Image.open(os.path.join(world, "output", "layout", f"{index}-layout-depth.png"))).astype(np.float64) / 1000
    gh, gw = pred_depth.shape
    step = layout["depth"]["grid_step_px"]
    ys = np.minimum(H - 1, np.arange(gh) * step + step // 2)
    xs = np.minimum(W - 1, np.arange(gw) * step + step // 2)
    gt_depth = arrays["depth"][ys][:, xs]
    emitting = (arrays["emission"][ys][:, xs] @ LUMA) > 1e-4
    valid = (pred_depth > 0) & (gt_depth > 0) & (gt_depth < 50) & ~emitting
    ratio = gt_depth[valid] / pred_depth[valid]
    scale = np.median(ratio)
    out["depth"] = {
        "absrel_metric": float(np.mean(np.abs(pred_depth[valid] - gt_depth[valid]) / gt_depth[valid])),
        "absrel_scaled": float(np.mean(np.abs(scale * pred_depth[valid] - gt_depth[valid]) / gt_depth[valid])),
        "delta1_metric": float(np.mean(np.maximum(ratio, 1 / ratio) < 1.25)),
        "scale_gt_over_pred": float(scale),
    }

    # Structure: each visible true plane against the best predicted plane (camera frame).
    normal_cam = arrays["normal_world"] @ R_gt.T
    structure = []
    for plane in gt["structure"]:
        n_gt, d_gt = plane_in_cam(plane["point"], plane["normal"], R_gt, t_gt)
        visible = ((arrays["instance"] == 0) & (np.abs(normal_cam @ n_gt) > 0.95) & ~((arrays["emission"] @ LUMA) > 1e-4)).mean()
        if visible < 0.02:
            continue
        best = None
        for pred in layout["structure"]:
            n_p, d_p = plane_in_cam(pred["center"], pred["normal"], R_pc, t_pc)
            err = (angle(n_p, n_gt), abs(d_p - d_gt))
            if best is None or err[0] + 20 * err[1] < best[1] + 20 * best[2]:
                best = (pred["id"], *err)
        found = best is not None and best[1] < 10 and best[2] < 0.25
        structure.append({"gt": plane["id"], "class": plane["class"], "visible_fraction": float(visible), "found": found,
                          "match": best[0] if best else None, "normal_err_deg": best[1] if best else None, "offset_err_m": best[2] if best else None})
    out["structure"] = structure

    # Objects: recall, mask IoU, box center and height errors, support.
    instances = {}
    for obj in layout["objects"]:
        for inst in obj["instances"]:
            instances.setdefault(obj["object_id"], []).append(inst)
    pred_by_gt = {}
    objects = []
    for obj in gt["objects"]:
        gt_mask = arrays["instance"] == obj["pass_index"]
        if gt_mask.mean() < 0.002:
            continue
        candidates = instances.get(slug(obj["id"]), [])
        best, best_iou = None, 0.0
        for inst in candidates:
            mask = np.asarray(Image.open(os.path.join(REPO, inst["mask_file"])).convert("L").resize((W, H), Image.NEAREST)) > 127
            iou = (mask & gt_mask).sum() / max((mask | gt_mask).sum(), 1)
            if iou > best_iou:
                best, best_iou = inst, iou
        record = {"gt": obj["id"], "name": obj["name"], "found": best is not None and best_iou >= 0.25, "mask_iou": float(best_iou)}
        if best is not None:
            pred_center = to_cam(np.array(best["center"]), R_pc, t_pc)
            gt_center = to_cam(np.array(obj["center"]), R_gt, t_gt)
            record.update({
                "center_err_m": float(np.linalg.norm(pred_center - gt_center)),
                "height_err_m": best["size"][1] - obj["size"][2],
                "height_rel_err": (best["size"][1] - obj["size"][2]) / obj["size"][2],
                "gt_support": obj["support"],
                "pred_support": best["support"],
            })
            pred_by_gt[best["id"]] = obj["id"]
        objects.append(record)
    for record in objects:
        if "pred_support" in record:
            pred = record["pred_support"]
            mapped = "floor" if pred in ("floor", "floor_estimate") else pred_by_gt.get(pred, pred)
            record["support_correct"] = mapped == record["gt_support"]
    out["objects"] = objects

    # Lighting: decomposition vs true albedo/shading, window regions and 3D positions.
    light_json = latest(os.path.join(world, "output", "light"), "[0-9]*-light.json")
    if light_json:
        light = json.load(open(light_json))
        albedo = np.asarray(Image.open(light["files"]["albedo"]).convert("RGB")).astype(np.float64) / 255
        albedo = np.where(albedo <= 0.04045, albedo / 12.92, ((albedo + 0.055) / 1.055) ** 2.4)
        surfaces = ((arrays["emission"] @ LUMA) < 1e-4)[..., None].repeat(3, -1)
        emit_gt = (arrays["emission"] @ LUMA) > 1e-4
        emit_pred = np.zeros((H, W), bool)
        ids = np.asarray(Image.open(light["files"]["emitters"]))
        for e in light["emitters"]:
            if e["kind"] == "light_source":
                emit_pred |= ids == e["id"]
        windows = []
        for window in gt["windows"]:
            gt_c = to_cam(np.array(window["center"]), R_gt, t_gt)
            pred = layout.get("lighting", {}).get("emitters", [])
            dists = [float(np.linalg.norm(to_cam(np.array(e["center"]), R_pc, t_pc) - gt_c)) for e in pred]
            windows.append({"gt": window["id"], "nearest_light_err_m": min(dists) if dists else None})
        out["light"] = {
            "albedo_si_rmse": si_rmse(albedo, arrays["albedo"], surfaces),
            "decomposition_r2": light["reconstruction_r2"],
            "light_source_iou": float((emit_pred & emit_gt).sum() / max((emit_pred | emit_gt).sum(), 1)),
            "windows": windows,
            "dominant_confidence": layout.get("lighting", {}).get("dominant_light", {}).get("confidence"),
        }
    return out


def summarize(results):
    def mean(values):
        values = [v for v in values if v is not None]
        return round(float(np.mean(values)), 4) if values else None

    def median_abs(values):
        values = [abs(v) for v in values if v is not None]
        return round(float(np.median(values)), 4) if values else None

    scored = [r for r in results if "camera" in r]
    planes = [p for r in scored for p in r["structure"]]
    objects = [o for r in scored for o in r["objects"]]
    found = [o for o in objects if o["found"]]
    light = [r["light"] for r in scored if "light" in r]
    windows = [w for l in light for w in l["windows"]]
    return {
        "views": len(results),
        "scored": len(scored),
        "camera": {k: median_abs([r["camera"][k] for r in scored]) for k in ("height_err_m", "pitch_err_deg", "roll_err_deg", "fov_err_deg")},
        "depth": {k: mean([r["depth"][k] for r in scored]) for k in ("absrel_metric", "absrel_scaled", "delta1_metric")},
        "structure": {
            "floor_found": mean([p["found"] for p in planes if p["class"] == "floor"]),
            "wall_found": mean([p["found"] for p in planes if p["class"] == "wall"]),
            "ceiling_found": mean([p["found"] for p in planes if p["class"] == "ceiling"]),
            "normal_err_deg_median": median_abs([p["normal_err_deg"] for p in planes if p["found"]]),
            "offset_err_m_median": median_abs([p["offset_err_m"] for p in planes if p["found"]]),
        },
        "objects": {
            "recall": mean([o["found"] for o in objects]),
            "mask_iou_mean": mean([o["mask_iou"] for o in objects]),
            "center_err_m_median": median_abs([o.get("center_err_m") for o in found]),
            "height_rel_err_median": median_abs([o.get("height_rel_err") for o in found]),
            "support_accuracy": mean([o.get("support_correct") for o in found]),
        },
        "light": {
            "albedo_si_rmse_mean": mean([l["albedo_si_rmse"] for l in light]),
            "decomposition_r2_mean": mean([l["decomposition_r2"] for l in light]),
            "light_source_iou_mean": mean([l["light_source_iou"] for l in light]),
            "window_position_err_m_median": median_abs([w["nearest_light_err_m"] for w in windows]),
            "windows_located_within_1m": mean([w["nearest_light_err_m"] is not None and w["nearest_light_err_m"] < 1 for w in windows]),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run", "score"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-light", action="store_true")
    parser.add_argument("--name", default="baseline")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.split)
    elif args.command == "run":
        run(args.split, args.python, args.skip_light)
    else:
        results = [score_view(v) for v in views(args.split)]
        report = {"name": args.name, "split": args.split, "summary": summarize(results), "views": results}
        os.makedirs(os.path.join(REPO, "benchmark", "results"), exist_ok=True)
        path = os.path.join(REPO, "benchmark", "results", f"{args.name}-{args.split}.json")
        json.dump(report, open(path, "w"), indent=2, default=float)
        print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
