#!/usr/bin/env python3
"""Scores the IMAGE-BLASTER pipeline against rendered ground truth.

  prepare  make worlds/bench-<split>-<room>-<view>[-noisy]/ from each render: source photo plus one object.json per
           visible true object. --names oracle uses the true names; --names noisy uses the short or synonym names
           an image-analysis model gives, drops some objects and over-counts some (tests the uncover step's input)
  score    compare a world's outputs with gt.npz/gt.json; writes benchmark/results/<name>-<split>.json with 95%
           bootstrap intervals over views. Every test-split score is logged in benchmark/results/test-reads.log
  compare  paired bootstrap of the difference between two scored result files on the same views; a metric whose
           interval contains 0 is a tie

All geometry is compared in the OpenCV camera frame (x right, y down, z forward) of the rendered camera.
"aligned" metrics first scale the prediction by the view's depth scale (true / predicted median depth).
"""
import argparse
import datetime
import glob
import json
import os
import random
import re
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RESULTS = os.path.join(REPO, "benchmark", "results")
LUMA = np.array([0.2126, 0.7152, 0.0722])
BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0])
CONFIRM_PX = 100  # an object visible on this many pixels is named by the uncover step
TRUTH_PX = 20  # objects smaller than this are not scoreable
SIZE_BUCKETS = [(0, 0.002, "under_0.2pct"), (0.002, 0.01, "0.2_to_1pct"), (0.01, 0.05, "1_to_5pct"), (0.05, 1.01, "over_5pct")]
PLANE_TOL_M = 0.25  # a predicted plane or window within this offset of the true one counts as found
WINDOW_HIT_IOU = 0.3

# Names an image-analysis model gives for the benchmark assets (short forms and synonyms).
NOISY_NAMES = {
    "chinese stool": ["stool", "wooden stool", "small stool"], "modern coffee table": ["coffee table", "low table"],
    "ceramic vase": ["vase", "pot"], "wooden bookshelf worn": ["bookshelf", "bookcase", "shelf"],
    "brass vase": ["vase", "metal vase"], "metal detector": ["metal detector", "tool"],
    "wooden table": ["table", "desk", "side table"], "barber shop chair": ["barber chair", "chair", "salon chair"],
    "coffee table round": ["round table", "coffee table"], "planter box": ["planter", "box"],
    "painted wooden cabinet": ["cabinet", "cupboard", "dresser"], "vintage oil lamp": ["oil lamp", "lamp", "lantern"],
    "brass diya lantern": ["lantern", "oil lamp"], "cassette player": ["radio", "cassette player", "stereo"],
    "gothic bed": ["bed", "bed frame"], "cash register": ["cash register", "register"],
    "rockingchair": ["rocking chair", "chair"], "planter pot clay": ["clay pot", "plant pot", "pot"],
    "wooden display shelves": ["shelves", "shelving unit", "display shelf"], "antique ceramic vase": ["vase", "urn"],
    "television": ["tv", "television"], "scandinavian masonry heater": ["stove", "fireplace", "heater"],
    "bar chair round": ["bar stool", "stool", "chair"], "painted wooden nightstand": ["nightstand", "bedside table", "side table"],
    "russian food cans": ["cans", "tin cans", "canned food"], "painted wooden shelves": ["shelves", "shelf", "bookcase"],
    "door": ["door", "doorway"],
}


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def views(split):
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(REPO, "benchmark", "renders", split, "room-*", "view-*", "gt.json")))


def world_name(view_dir, names="oracle"):
    room, view = view_dir.rstrip("/").split("/")[-2:]
    split = view_dir.rstrip("/").split("/")[-3]
    return f"bench-{split}-{room}-{view}" + ("-noisy" if names == "noisy" else "")


def prepare(split, names):
    for view_dir in views(split):
        gt = json.load(open(os.path.join(view_dir, "gt.json")))
        name = world_name(view_dir, names)
        world = os.path.join(REPO, "worlds", name)
        shutil.rmtree(world, ignore_errors=True)
        os.makedirs(os.path.join(world, "source"))
        shutil.copy(os.path.join(view_dir, "rgb.png"), os.path.join(world, "source", "0-render.png"))
        json.dump({"schema_version": 1, "slug": name, "display_name": name}, open(os.path.join(world, "project.json"), "w"))
        json.dump({"view_dir": os.path.relpath(view_dir, REPO), "names": names}, open(os.path.join(world, "bench.json"), "w"))
        instance = np.load(os.path.join(view_dir, "gt.npz"))["instance"]
        rng = random.Random(gt["seed"] * 7 + gt["view"])
        for obj in gt["objects"]:
            if (instance == obj["pass_index"]).sum() < CONFIRM_PX:
                continue
            label, count = obj["name"], 1
            if names == "noisy":
                if rng.random() < 0.15:  # missed by the analysis
                    continue
                label = rng.choice(NOISY_NAMES.get(obj["name"], [obj["name"].split()[-1]]))
                count = 2 if rng.random() < 0.1 else 1
            object_id = slug(obj["id"])  # ids stay the true ids, so scoring can match instances to truth
            os.makedirs(os.path.join(world, "output", object_id))
            json.dump({"schema_version": 1, "world": name, "object": {
                "id": object_id, "name": label, "description": label, "count_estimate": count,
                "generate_as_3d_object": True, "working_dir": f"worlds/{name}/output/{object_id}"}},
                open(os.path.join(world, "output", object_id, "object.json"), "w"), indent=2)
        print(name)


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


def project(points_cam, K):
    p = np.asarray(points_cam, float)
    return np.stack([K["fx"] * p[:, 0] / p[:, 2] + K["cx"], K["fy"] * p[:, 1] / p[:, 2] + K["cy"]], 1)


def polygon_mask(points_px, shape):
    canvas = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(canvas).polygon([tuple(p) for p in points_px], fill=1)
    return np.asarray(canvas, bool)


def window_rect(window, R, t):
    """True window in the camera frame: center, in-plane axes (u along the wall, v up), normal, size."""
    n = R @ np.asarray(window["normal"], float)
    u = R @ np.cross([0, 0, 1], window["normal"]) / np.linalg.norm(np.cross([0, 0, 1], window["normal"]))
    v = R @ np.array([0, 0, 1.0])
    return to_cam(window["center"], R, t), u, v, n, window["size"]


def rect_iou_on_plane(center, u, v, n, size, corners):
    """IoU of a predicted quad (camera frame) with the true rectangle, both in the true window plane; 0 when
    the quad lies more than PLANE_TOL_M off that plane. Sampled on a 2 cm grid."""
    rel = np.asarray(corners) - center
    if np.mean(np.abs(rel @ n)) > PLANE_TOL_M:
        return 0.0
    quad = np.stack([rel @ u, rel @ v], 1)
    lo = np.minimum(quad.min(0), [-size[0] / 2, -size[1] / 2])
    hi = np.maximum(quad.max(0), [size[0] / 2, size[1] / 2])
    if np.any(hi - lo > 20):
        return 0.0
    gx, gy = np.meshgrid(np.arange(lo[0] + 0.01, hi[0], 0.02), np.arange(lo[1] + 0.01, hi[1], 0.02))  # cell centers
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    in_true = (np.abs(pts[:, 0]) <= size[0] / 2) & (np.abs(pts[:, 1]) <= size[1] / 2)
    edges = np.roll(quad, -1, 0) - quad
    cross = edges[None, :, 0] * (pts[:, None, 1] - quad[None, :, 1]) - edges[None, :, 1] * (pts[:, None, 0] - quad[None, :, 0])
    in_pred = (cross >= 0).all(1) | (cross <= 0).all(1)
    union = (in_true | in_pred).sum()
    return float((in_true & in_pred).sum() / union) if union else 0.0


def score_view(view_dir, names="oracle", layout_index=None):
    gt = json.load(open(os.path.join(view_dir, "gt.json")))
    arrays = np.load(os.path.join(view_dir, "gt.npz"))
    name = world_name(view_dir, names)
    world = os.path.join(REPO, "worlds", name)
    layout_path = (os.path.join(world, "output", "layout", f"{layout_index}-layout.json") if layout_index is not None
                   else latest(os.path.join(world, "output", "layout"), "[0-9]*-layout.json"))
    if not layout_path or not os.path.exists(layout_path):
        return {"view": name, "status": "no_layout"}
    layout = json.load(open(layout_path))
    index = int(os.path.basename(layout_path).split("-")[0])
    R_gt, t_gt = gt_camera(gt)
    R_pc, t_pc = pred_cam(layout)
    depth_true = arrays["depth"].astype(np.float64)
    H, W = depth_true.shape
    window_mask = arrays["window"] if "window" in arrays else (arrays["emission"].astype(np.float32) @ LUMA) > 1e-4
    instance = arrays["instance"]
    K = gt["camera"]["intrinsics_px"]
    out = {"view": name, "layout_index": index}

    # Camera.
    cam = layout["camera"]
    out["camera"] = {
        "height_err_m": cam["height_m"] - gt["camera"]["height_m"],
        "pitch_err_deg": cam["pitch_deg"] - gt["camera"]["pitch_deg"],
        "roll_err_deg": cam["roll_deg"] - gt["camera"]["roll_deg"],
        "fov_err_deg": cam["fov_x_deg"] - gt["camera"]["fov_x_deg"],
    }

    # Depth (metric and median-scaled) from the layout depth PNG (mm, grid resolution); windows and sky excluded.
    pred_depth = np.asarray(Image.open(os.path.join(world, "output", "layout", f"{index}-layout-depth.png"))).astype(np.float64) / 1000
    gh, gw = pred_depth.shape
    step = layout["depth"]["grid_step_px"]
    ys = np.minimum(H - 1, np.arange(gh) * step + step // 2)
    xs = np.minimum(W - 1, np.arange(gw) * step + step // 2)
    gt_depth = depth_true[ys][:, xs]
    valid = (pred_depth > 0) & np.isfinite(gt_depth) & (gt_depth > 0) & (gt_depth < 50) & ~window_mask[ys][:, xs]
    ratio = gt_depth[valid] / pred_depth[valid]
    scale = float(np.median(ratio))
    out["depth"] = {
        "absrel_metric": float(np.mean(np.abs(pred_depth[valid] - gt_depth[valid]) / gt_depth[valid])),
        "absrel_scaled": float(np.mean(np.abs(scale * pred_depth[valid] - gt_depth[valid]) / gt_depth[valid])),
        "delta1_metric": float(np.mean(np.maximum(ratio, 1 / ratio) < 1.25)),
        "scale_gt_over_pred": scale,
        "log_scale_err": float(np.log(scale)),
    }
    if layout.get("scale"):
        out["depth"]["scale_applied"] = layout["scale"].get("applied")

    # Structure: each visible true plane against the best predicted plane of any class (camera frame); the class
    # is scored apart from the geometry. A plane is visible where true 3D points (planar depth) lie on it.
    v_px, u_px = np.mgrid[0:H, 0:W]
    with np.errstate(invalid="ignore"):
        cam_points = np.stack([(u_px + 0.5 - K["cx"]) / K["fx"], (v_px + 0.5 - K["cy"]) / K["fy"], np.ones((H, W))], -1) * depth_true[..., None]
    background = (instance == 0) & ~window_mask & np.isfinite(depth_true)
    structure = []
    for plane in gt["structure"]:
        n_gt, d_gt = plane_in_cam(plane["point"], plane["normal"], R_gt, t_gt)
        with np.errstate(invalid="ignore"):
            visible = (background & (np.abs(np.nan_to_num(cam_points @ n_gt + d_gt, nan=1e9)) < 0.03)).mean()
        if visible < 0.02:
            continue
        best = None
        for pred in layout["structure"]:
            n_p, d_p = plane_in_cam(pred["center"], pred["normal"], R_pc, t_pc)
            err = (angle(n_p, n_gt), abs(d_p - d_gt), abs(scale * d_p - d_gt))
            if best is None or err[0] + 20 * err[1] < best[2] + 20 * best[3]:
                best = (pred["id"], pred["class"], *err)
        found = best is not None and best[2] < 10 and best[3] < PLANE_TOL_M
        found_aligned = best is not None and best[2] < 10 and best[4] < PLANE_TOL_M
        structure.append({"gt": plane["id"], "class": plane["class"], "visible_fraction": float(visible),
                          "found": found, "found_aligned": found_aligned,
                          "match": best[0] if best else None, "match_class": best[1] if best else None,
                          "class_correct": bool(found_aligned and best[1] == plane["class"]),
                          "normal_err_deg": best[2] if best else None,
                          "offset_err_m": best[3] if best else None, "offset_err_aligned_m": best[4] if best else None})
    out["structure"] = structure

    # Objects: every true object visible on TRUTH_PX pixels, matched by id to the predicted instances.
    instances = {}
    for obj in layout["objects"]:
        for inst in obj["instances"]:
            instances.setdefault(obj["object_id"], []).append(inst)
    pred_by_gt = {}
    objects = []
    false_positives = 0
    for obj in gt["objects"]:
        gt_mask = instance == obj["pass_index"]
        pixels = int(gt_mask.sum())
        if pixels < TRUTH_PX:
            continue
        # Truncated: part of the true box projects outside the image (the photo cannot show the whole object).
        c, s_ = np.cos(np.radians(obj["yaw_deg"])), np.sin(np.radians(obj["yaw_deg"]))
        sx, sy, sz = obj["size"]
        local = np.array([[x, y, z] for x in (-sx / 2, sx / 2) for y in (-sy / 2, sy / 2) for z in (-sz / 2, sz / 2)])
        corners = local @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]).T + np.array(obj["center"])
        corners_cam = to_cam(corners, R_gt, t_gt)
        if (corners_cam[:, 2] <= 0.05).any():
            truncated = True
        else:
            px = project(corners_cam, K)
            truncated = bool(((px < 0) | (px > [W, H])).any())
        candidates = instances.get(slug(obj["id"]), [])
        best, best_iou = None, 0.0
        for inst in candidates:
            mask = np.asarray(Image.open(os.path.join(REPO, inst["mask_file"])).convert("L").resize((W, H), Image.NEAREST)) > 127
            iou = (mask & gt_mask).sum() / max((mask | gt_mask).sum(), 1)
            if iou > best_iou:
                best, best_iou = inst, iou
        found = best is not None and best_iou >= 0.25
        # Every predicted instance of this object other than a correct match is a false positive.
        false_positives += len(candidates) - (1 if found else 0)
        bucket = next(label for lo, hi, label in SIZE_BUCKETS if lo <= pixels / (H * W) < hi)
        record = {"gt": obj["id"], "name": obj["name"], "class": obj.get("class", "furniture"), "pixel_fraction": pixels / (H * W),
                  "size_bucket": bucket, "truncated": truncated, "named": (pixels >= CONFIRM_PX), "found": found, "mask_iou": float(best_iou)}
        if best is not None:
            pred_center = to_cam(np.array(best["center"]), R_pc, t_pc)
            gt_center = to_cam(np.array(obj["center"]), R_gt, t_gt)
            record.update({
                "center_err_m": float(np.linalg.norm(pred_center - gt_center)),
                "center_err_aligned_m": float(np.linalg.norm(scale * pred_center - gt_center)),
                "height_rel_err_aligned": (scale * best["size"][1] - obj["size"][2]) / obj["size"][2],
                "height_err_m": best["size"][1] - obj["size"][2],
                "height_rel_err": (best["size"][1] - obj["size"][2]) / obj["size"][2],
                "gt_support": obj["support"],
                "pred_support": best["support"],
                "pred_truncated": best.get("truncated"),
            })
            pred_by_gt[best["id"]] = obj["id"]
        objects.append(record)
    for record in objects:
        if "pred_support" in record:
            pred = record["pred_support"]
            mapped = "floor" if pred in ("floor", "floor_estimate") else pred_by_gt.get(pred, pred)
            record["support_correct"] = mapped == record["gt_support"]
    out["objects"] = objects
    out["object_false_positives"] = false_positives

    # Lighting: decomposition, then light sources as pixels (light stage and the layout's final set) and as
    # 3D rectangles on the true window planes.
    light_json = latest(os.path.join(world, "output", "light"), "[0-9]*-light.json")
    lighting = layout.get("lighting", {})
    if light_json:
        light = json.load(open(light_json))
        albedo = np.asarray(Image.open(light["files"]["albedo"]).convert("RGB")).astype(np.float64) / 255
        albedo = np.where(albedo <= 0.04045, albedo / 12.92, ((albedo + 0.055) / 1.055) ** 2.4)
        surfaces = (~window_mask & np.isfinite(depth_true))[..., None].repeat(3, -1)
        ids = np.asarray(Image.open(light["files"]["emitters"]))
        stage = np.isin(ids, [e["id"] for e in light["emitters"] if e["kind"] == "light_source"])
        final = np.isin(ids, [r for e in lighting.get("emitters", []) for r in e.get("image_regions", [])])
        visible = window_mask.mean() > 0.005
        iou = lambda pred: float((pred & window_mask).sum() / max((pred | window_mask).sum(), 1)) if visible else None
        windows = []
        for window in gt["windows"]:
            center, u, v, n, size = window_rect(window, R_gt, t_gt)
            corners = np.array([center + a * size[0] / 2 * u + b * size[1] / 2 * v for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            if (corners[:, 2] <= 0.05).any():
                continue
            own = polygon_mask(project(corners, K), (H, W)) & window_mask
            if own.mean() < 0.005:
                continue
            best = None
            for e in lighting.get("emitters", []):
                pc = to_cam(np.array(e["corners"]), R_pc, t_pc)
                record = {"emitter": e["id"],
                          "rect_iou": rect_iou_on_plane(center, u, v, n, size, pc),
                          "rect_iou_aligned": rect_iou_on_plane(center, u, v, n, size, scale * pc),
                          "normal_err_deg": angle(R_pc @ np.array(e["normal"]), n),
                          "width_rel_err_aligned": (scale * e["size"][0] - size[0]) / size[0],
                          "height_rel_err_aligned": (scale * e["size"][1] - size[1]) / size[1],
                          "center_err_m": float(np.linalg.norm(to_cam(np.array(e["center"]), R_pc, t_pc) - center)),
                          "center_err_aligned_m": float(np.linalg.norm(scale * to_cam(np.array(e["center"]), R_pc, t_pc) - center))}
                key = (record["rect_iou_aligned"], -record["center_err_aligned_m"])
                if best is None or key > (best["rect_iou_aligned"], -best["center_err_aligned_m"]):
                    best = record
            windows.append({"gt": window["id"], "kind": window.get("kind", "panel"), "visible_fraction": float(own.mean()),
                            "clipped": bool(gt.get("tone", {}).get("curve", "clip") == "clip"),
                            "located": bool(best and best["rect_iou_aligned"] >= WINDOW_HIT_IOU), **(best or {})})
        out["light"] = {
            "albedo_si_rmse": si_rmse(albedo, arrays["albedo"].astype(np.float64), surfaces),
            "decomposition_r2": light["reconstruction_r2"],
            "windows_visible": bool(visible),
            "light_source_iou_stage": iou(stage),
            "light_source_iou_final": iou(final),
            "false_light_fraction_stage": float((stage & ~window_mask).mean()),
            "false_light_fraction_final": float((final & ~window_mask).mean()),
            "windows": windows,
            "dominant_confidence": lighting.get("dominant_light", {}).get("confidence"),
            "light_sources": len(lighting.get("emitters", [])),
            "tone": gt.get("tone", {}).get("curve"),
        }

    # Plate and environment against the empty render, on the furniture pixels (what the plate must invent).
    if "depth_empty" in arrays:
        furniture = np.isin(instance, [o["pass_index"] for o in gt["objects"] if o.get("class", "furniture") == "furniture"])
        plate_path = latest(os.path.join(world, "source"), "[0-9]*-*plate.png")
        if plate_path and furniture.any():
            plate = np.asarray(Image.open(plate_path).convert("RGB").resize((W, H))).astype(np.float64)
            empty = np.asarray(Image.open(os.path.join(view_dir, "empty.png")).convert("RGB")).astype(np.float64)
            source = np.asarray(Image.open(os.path.join(view_dir, "rgb.png")).convert("RGB")).astype(np.float64)
            mse = lambda a: float(np.mean((a[furniture] - empty[furniture]) ** 2))
            psnr = lambda a: float(10 * np.log10(255 ** 2 / max(mse(a), 1e-9)))
            out["plate"] = {"psnr_furniture_px": psnr(plate), "psnr_unedited_source": psnr(source), "furniture_fraction": float(furniture.mean())}
        env = latest(os.path.join(world, "output", "scene"), "[0-9]*-scene-environment.glb")
        if env and furniture.any():
            import trimesh

            mesh = trimesh.load(env, force="mesh")
            verts = to_cam(np.asarray(mesh.vertices), R_pc, t_pc)
            front = verts[:, 2] > 0.05
            px = np.rint(project(verts[front], K)).astype(int)
            inside = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
            z, px = verts[front][inside, 2], px[inside]
            truth = arrays["depth_empty"][px[:, 1], px[:, 0]].astype(np.float64)
            hidden = furniture[px[:, 1], px[:, 0]] & np.isfinite(truth)
            if hidden.sum() > 50:
                out["environment"] = {"absrel_hidden_metric": float(np.mean(np.abs(z[hidden] - truth[hidden]) / truth[hidden])),
                                      "absrel_hidden_aligned": float(np.mean(np.abs(scale * z[hidden] - truth[hidden]) / truth[hidden]))}
    return out


def summarize(results):
    def mean(values):
        values = [float(v) for v in values if v is not None]
        return round(float(np.mean(values)), 4) if values else None

    def median_abs(values):
        values = [abs(v) for v in values if v is not None]
        return round(float(np.median(values)), 4) if values else None

    scored = [r for r in results if "camera" in r]
    planes = [p for r in scored for p in r["structure"]]
    objects = [o for r in scored for o in r["objects"]]
    named = [o for o in objects if o["named"]]
    found = [o for o in named if o["found"]]
    light = [r["light"] for r in scored if "light" in r]
    windows = [w for l in light for w in l["windows"]]
    tp = sum(o["found"] for o in named)
    summary = {
        "views": len(results),
        "scored": len(scored),
        "camera": {k: median_abs([r["camera"][k] for r in scored]) for k in ("height_err_m", "pitch_err_deg", "roll_err_deg", "fov_err_deg")},
        "depth": {**{k: mean([r["depth"][k] for r in scored]) for k in ("absrel_metric", "absrel_scaled", "delta1_metric")},
                  "scale_err_median": median_abs([r["depth"]["scale_gt_over_pred"] - 1 for r in scored]),
                  "log_scale_bias": mean([r["depth"]["log_scale_err"] for r in scored])},
        "structure": {
            "floor_found_aligned": mean([p["found_aligned"] for p in planes if p["class"] == "floor"]),
            "wall_found_aligned": mean([p["found_aligned"] for p in planes if p["class"] == "wall"]),
            "ceiling_found_aligned": mean([p["found_aligned"] for p in planes if p["class"] == "ceiling"]),
            "wall_found": mean([p["found"] for p in planes if p["class"] == "wall"]),
            "class_correct": mean([p["class_correct"] for p in planes if p["found_aligned"]]),
            "normal_err_deg_median": median_abs([p["normal_err_deg"] for p in planes if p["found_aligned"]]),
            "offset_err_m_median": median_abs([p["offset_err_m"] for p in planes if p["found"]]),
            "offset_err_aligned_m_median": median_abs([p["offset_err_aligned_m"] for p in planes if p["found_aligned"]]),
        },
        "objects": {
            "recall": mean([o["found"] for o in named]),
            "precision": round(tp / max(tp + sum(r["object_false_positives"] for r in scored), 1), 4),
            "recall_by_size": {label: mean([o["found"] for o in named if o["size_bucket"] == label]) for _, _, label in SIZE_BUCKETS},
            "unnamed_small_objects": sum(not o["named"] for o in objects),
            "mask_iou_mean": mean([o["mask_iou"] for o in named]),
            "center_err_aligned_m_median": median_abs([o.get("center_err_aligned_m") for o in found]),
            "center_err_aligned_m_median_truncated": median_abs([o.get("center_err_aligned_m") for o in found if o["truncated"]]),
            "center_err_aligned_m_median_whole": median_abs([o.get("center_err_aligned_m") for o in found if not o["truncated"]]),
            "center_err_m_median": median_abs([o.get("center_err_m") for o in found]),
            "height_rel_err_aligned_median": median_abs([o.get("height_rel_err_aligned") for o in found]),
            "height_rel_err_aligned_median_whole": median_abs([o.get("height_rel_err_aligned") for o in found if not o["truncated"]]),
            "truncation_flag_accuracy": mean([o.get("pred_truncated") == o["truncated"] for o in found if o.get("pred_truncated") is not None]),
            "support_accuracy": mean([o.get("support_correct") for o in found]),
        },
        "light": {
            "albedo_si_rmse_mean": mean([l["albedo_si_rmse"] for l in light]),
            "decomposition_r2_mean": mean([l["decomposition_r2"] for l in light]),
            "views_with_windows": sum(l["windows_visible"] for l in light),
            "windows": len(windows),
            "light_source_iou_stage_mean": mean([l["light_source_iou_stage"] for l in light]),
            "light_source_iou_final_mean": mean([l["light_source_iou_final"] for l in light]),
            "false_light_fraction_final_mean": mean([l["false_light_fraction_final"] for l in light]),
            "window_located": mean([w["located"] for w in windows]),
            "window_located_clipped": mean([w["located"] for w in windows if w["clipped"]]),
            "window_located_rolloff": mean([w["located"] for w in windows if not w["clipped"]]),
            "window_rect_iou_aligned_mean": mean([w.get("rect_iou_aligned", 0.0) for w in windows]),
            "window_rect_iou_mean": mean([w.get("rect_iou", 0.0) for w in windows]),
            "window_normal_err_deg_median": median_abs([w.get("normal_err_deg") for w in windows]),
            "window_width_rel_err_aligned_median": median_abs([w.get("width_rel_err_aligned") for w in windows]),
            "window_center_err_aligned_m_median": median_abs([w.get("center_err_aligned_m") for w in windows]),
            "light_sources_per_view": mean([l.get("light_sources", 0) for l in light]),
        },
    }
    plates = [r["plate"] for r in scored if "plate" in r]
    if plates:
        summary["plate"] = {"psnr_furniture_px_mean": mean([p["psnr_furniture_px"] for p in plates]),
                            "psnr_unedited_source_mean": mean([p["psnr_unedited_source"] for p in plates]), "views": len(plates)}
    envs = [r["environment"] for r in scored if "environment" in r]
    if envs:
        summary["environment"] = {"absrel_hidden_metric_mean": mean([e["absrel_hidden_metric"] for e in envs]),
                                  "absrel_hidden_aligned_mean": mean([e["absrel_hidden_aligned"] for e in envs]), "views": len(envs)}
    return summary


def leaves(tree, prefix=()):
    for key, value in tree.items():
        if isinstance(value, dict):
            yield from leaves(value, prefix + (key,))
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and value is not None:
            yield prefix + (key,), value


def get(tree, path):
    for key in path:
        tree = tree.get(key) if isinstance(tree, dict) else None
    return tree


def bootstrap(results, draws=1000, seed=0):
    """95% interval of every summary number, resampling views (one view per room, so views are independent)."""
    scored = [r for r in results if "camera" in r]
    rng = np.random.default_rng(seed)
    samples = [summarize([scored[i] for i in rng.integers(0, len(scored), len(scored))]) for _ in range(draws)]
    ci = {}
    for path, _ in leaves(summarize(scored)):
        if path[-1] in ("views", "scored", "windows", "views_with_windows", "unnamed_small_objects"):
            continue
        values = [v for v in (get(s, path) for s in samples) if v is not None]
        if values:
            node = ci
            for key in path[:-1]:
                node = node.setdefault(key, {})
            node[path[-1]] = [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]
    return ci


def compare(a, b, draws=1000, seed=0):
    """Paired bootstrap of b - a on the views both scored. A metric whose interval contains 0 is a tie."""
    va = {r["view"]: r for r in a["views"] if "camera" in r}
    vb = {r["view"]: r for r in b["views"] if "camera" in r}
    common = sorted(set(va) & set(vb), key=lambda v: v.replace("-noisy", ""))
    ra, rb = [va[v] for v in common], [vb[v] for v in common]
    rng = np.random.default_rng(seed)
    base_a, base_b = summarize(ra), summarize(rb)
    diffs = {}
    for _ in range(draws):
        idx = rng.integers(0, len(common), len(common))
        sa, sb = summarize([ra[i] for i in idx]), summarize([rb[i] for i in idx])
        for path, _ in leaves(base_a):
            x, y = get(sa, path), get(sb, path)
            if x is not None and y is not None:
                diffs.setdefault(path, []).append(y - x)
    table = []
    for path, values in diffs.items():
        if path[-1] in ("views", "scored", "windows", "views_with_windows", "unnamed_small_objects"):
            continue
        lo, hi = np.percentile(values, [2.5, 97.5])
        table.append({"metric": ".".join(path), "a": get(base_a, path), "b": get(base_b, path),
                      "diff_ci95": [round(float(lo), 4), round(float(hi), 4)], "verdict": "tie" if lo <= 0 <= hi else ("b higher" if lo > 0 else "a higher")})
    return {"views": len(common), "a": a["name"], "b": b["name"], "metrics": table}


def log_test_read(name, summary):
    path = os.path.join(RESULTS, "test-reads.log")
    previous = sum(1 for _ in open(path)) if os.path.exists(path) else 0
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    with open(path, "a") as f:
        f.write(json.dumps({"time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "name": name,
                            "commit": commit, "objects_recall": summary["objects"]["recall"],
                            "window_located": summary["light"]["window_located"]}) + "\n")
    print(f"test split read #{previous + 1} (log: {os.path.relpath(path, REPO)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "score", "compare"])
    parser.add_argument("--split", default="val")
    parser.add_argument("--names", choices=["oracle", "noisy"], default="oracle")
    parser.add_argument("--name", default="current")
    parser.add_argument("--layout-index", type=int, help="score this layout index instead of the latest")
    parser.add_argument("--a", help="compare: result file name (without -<split>.json)")
    parser.add_argument("--b", help="compare: result file name (without -<split>.json)")
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if args.command == "prepare":
        prepare(args.split, args.names)
    elif args.command == "score":
        results = [score_view(v, args.names, args.layout_index) for v in views(args.split)]
        summary = summarize(results)
        report = {"name": args.name, "split": args.split, "names": args.names, "summary": summary,
                  "ci95": bootstrap(results, args.draws), "views": results}
        json.dump(report, open(os.path.join(RESULTS, f"{args.name}-{args.split}.json"), "w"), indent=2, default=float)
        if args.split == "test":
            log_test_read(args.name, summary)
        print(json.dumps(summary, indent=2))
    else:
        load = lambda n: json.load(open(os.path.join(RESULTS, f"{n}-{args.split}.json")))
        result = compare(load(args.a), load(args.b), args.draws)
        json.dump(result, open(os.path.join(RESULTS, f"compare-{args.a}-vs-{args.b}-{args.split}.json"), "w"), indent=2)
        for row in result["metrics"]:
            print(f"{row['metric']:60s} {row['a']!s:>9} {row['b']!s:>9} {row['diff_ci95']!s:>22} {row['verdict']}")


if __name__ == "__main__":
    main()
