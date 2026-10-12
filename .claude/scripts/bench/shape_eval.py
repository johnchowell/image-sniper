#!/usr/bin/env python3
"""Object shape completion against the true meshes.

  score    build every matched object's shape with primitive_scene.object_geometry and compare it with the true
           Poly Haven mesh, placed as rendered. Writes benchmark/results/shape-<name>-<mode>-<split>.json.
  compare  paired bootstrap (resampling views) of the difference between two scored files.

Modes:
  gt    true depth, true instance masks and true camera; the object box comes from the layout (mapped into the
        true frame with the view's depth scale). Shape completion is then the main error.
  pred  the pipeline as it runs: layout depth, layout masks; the shape is mapped into the true frame with the
        view's depth scale (true / predicted median depth).

The true surface is split into the part the camera sees (visible) and the rest (hidden: back sides, occluded
parts). Metrics per object (meters; tau = 5% of the true box diagonal):
  acc         mean distance from the shape to the true surface
  comp        mean distance from the true surface to the shape (comp_visible, comp_hidden)
  f           F-score at tau (precision, recall, recall_hidden)
  nc          normal consistency: mean |n . n_nearest| both ways (1 = every face has the true orientation)
  nc_near     the same over the points within tau of the other surface only: how flat and well oriented the
              surface is where the shape is right (the flatness of the faces, apart from completion errors)
Every test-split score is logged in benchmark/results/test-reads.log.
"""
import argparse
import datetime
import glob
import json
import math
import os
import shlex
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, ".claude", "scripts", "local"))
sys.path.insert(0, HERE)
import evaluate  # noqa: E402
import primitive_scene as ps  # noqa: E402

RESULTS = os.path.join(REPO, "benchmark", "results")
GLTF_TO_BLENDER = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
BLENDER_TO_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)  # Blender +Z up -> layout-like +Y up
POINTS = 20000
_MESHES = {}


def true_mesh(obj):
    """The true object mesh in the +Y-up true world frame, placed as render_scene.py placed it."""
    if obj["model_path"] not in _MESHES:
        mesh = trimesh.load(os.path.join(REPO, obj["model_path"]), force="mesh")
        V = np.asarray(mesh.vertices) @ GLTF_TO_BLENDER.T
        low, high = V.min(0), V.max(0)
        _MESHES[obj["model_path"]] = (V - np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]]), np.asarray(mesh.faces))
    V, F = _MESHES[obj["model_path"]]
    yaw = math.radians(obj["yaw_deg"])
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    V = V @ Rz.T + np.array([obj["center"][0], obj["center"][1], obj["bottom_z"]])
    return trimesh.Trimesh(vertices=V @ BLENDER_TO_YUP.T, faces=F, process=False)


def true_view(gt, depth):
    """A primitive_scene.View of the true camera in the +Y-up true frame."""
    M = np.array(gt["camera"]["matrix_world"])
    R = BLENDER_TO_YUP @ M[:3, :3] @ evaluate.BLENDER_TO_OPENCV
    t = BLENDER_TO_YUP @ M[:3, 3]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    W, H = gt["image_size"]
    fake = {"camera": {"matrix_world_from_camera_opencv": T.tolist(), "intrinsics_px": gt["camera"]["intrinsics_px"],
                       "image_size": [W, H], "position": t.tolist()}}
    return ps.View(fake, depth)


def surface_samples(mesh, n, seed=0):
    if len(mesh.faces) == 0 or mesh.area <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    pts, face = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts), np.asarray(mesh.face_normals[face])


def metrics(pred, true, visible_true, diag):
    (pp, pn), (tp, tn) = pred, true
    if len(pp) == 0:
        return {"empty": True}
    tau = 0.05 * diag
    d_pt, i_pt = cKDTree(tp).query(pp)
    d_tp, i_tp = cKDTree(pp).query(tp)
    hid = ~visible_true
    dot_p, dot_t = np.abs(np.einsum("ij,ij->i", pn, tn[i_pt])), np.abs(np.einsum("ij,ij->i", tn, pn[i_tp]))
    nc = (dot_p.mean() + dot_t.mean()) / 2
    near_p, near_t = d_pt < tau, d_tp < tau
    nc_near = (dot_p[near_p].mean() + dot_t[near_t].mean()) / 2 if near_p.any() and near_t.any() else None
    precision, recall = (d_pt < tau).mean(), (d_tp < tau).mean()
    return {"acc": float(d_pt.mean()), "comp": float(d_tp.mean()),
            "comp_visible": float(d_tp[~hid].mean()) if (~hid).any() else None,
            "comp_hidden": float(d_tp[hid].mean()) if hid.any() else None,
            "chamfer": float((d_pt.mean() + d_tp.mean()) / 2),
            "precision": float(precision), "recall": float(recall),
            "recall_hidden": float((d_tp[hid] < tau).mean()) if hid.any() else None,
            "f": float(2 * precision * recall / max(precision + recall, 1e-9)), "nc": float(nc),
            "nc_near": None if nc_near is None else float(nc_near)}


def score_view(job):
    view_dir, mode, extra = job
    args = ps.build_parser().parse_args(["--world", "-"] + shlex.split(extra))
    gt = json.load(open(os.path.join(view_dir, "gt.json")))
    arrays = np.load(os.path.join(view_dir, "gt.npz"))
    world = os.path.join(REPO, "worlds", evaluate.world_name(view_dir, "oracle"))
    found = evaluate.latest(os.path.join(world, "output", "layout"), "[0-9]*-layout.json")
    if not found:
        return {"view": os.path.basename(world), "status": "no_layout"}
    layout = json.load(open(found))
    cwd = os.getcwd()
    os.chdir(REPO)  # layout file paths are relative to the repository
    try:
        view_pred, instances, _, room_px = ps.scene_inputs(layout)
    finally:
        os.chdir(cwd)
    depth_true = arrays["depth"].astype(np.float64)
    depth_true[~np.isfinite(depth_true) | (depth_true > 50)] = np.nan
    instance = arrays["instance"]
    H, W = depth_true.shape
    vt = true_view(gt, depth_true)
    # Similarity layout frame -> true +Y-up frame: layout -> predicted OpenCV camera, depth scale, true camera.
    valid = np.isfinite(view_pred.depth) & np.isfinite(depth_true)
    scale = float(np.median(depth_true[valid] / view_pred.depth[valid]))
    R_lc = view_pred.R.T  # layout -> camera
    to_true = lambda P: (scale * ((np.asarray(P) - view_pred.t) @ R_lc.T)) @ vt.R.T + vt.t
    R_total = vt.R @ R_lc
    by_id = {evaluate.slug(o["id"]): o for o in gt["objects"]}
    rows = []
    masks = {o["pass_index"]: instance == o["pass_index"] for o in gt["objects"]}
    for obj, inst, under, exclude in ps.object_masks(instances, room_px):
        true_obj = by_id.get(obj.get("object_id"))
        if true_obj is None or true_obj.get("class", "furniture") != "furniture":
            continue
        gt_mask = masks[true_obj["pass_index"]]
        if gt_mask.sum() < evaluate.CONFIRM_PX:
            continue
        mask_pred = np.asarray(Image.open(os.path.join(REPO, inst["mask_file"])).convert("L").resize((W, H), Image.NEAREST)) > 127
        iou = (mask_pred & gt_mask).sum() / max((mask_pred | gt_mask).sum(), 1)
        if iou < 0.25:
            continue
        if mode == "gt":
            x_axis = R_total @ np.array([math.cos(math.radians(inst["yaw_deg"])), 0, -math.sin(math.radians(inst["yaw_deg"]))])
            yaw = math.degrees(math.atan2(-x_axis[2], x_axis[0]))
            inst_t = {**inst, "center": to_true(inst["center"]).tolist(), "size": (scale * np.array(inst["size"])).tolist(), "yaw_deg": yaw}
            supported_ids = [o["pass_index"] for o in gt["objects"] if o.get("support") == true_obj["id"]]
            mask_t = gt_mask.copy()
            for k in supported_ids:
                mask_t |= masks[k]
            exclude_t = (instance != true_obj["pass_index"]) & ~mask_t
            ys, xs = np.nonzero(gt_mask)
            inst_t["image_bbox_px"] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            mesh, *_, info = ps.object_geometry(obj, inst_t, mask_t, vt, args, exclude_t)
        else:
            mesh, *_, info = ps.object_geometry(obj, inst, under, view_pred, args, exclude)
            mesh = trimesh.Trimesh(vertices=to_true(mesh.vertices), faces=mesh.faces, process=False)
        tm = true_mesh(true_obj)
        tp, tn = surface_samples(tm, POINTS)
        u, v, z = vt.project(tp)
        inside = (z > 0.05) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        ui = np.clip(np.round(np.nan_to_num(u)).astype(int), 0, W - 1)
        vi = np.clip(np.round(np.nan_to_num(v)).astype(int), 0, H - 1)
        zobs = depth_true[vi, ui]
        visible = inside & (np.abs(z - np.nan_to_num(zobs, nan=-1)) < 0.01 + 0.01 * z) & gt_mask[vi, ui]
        diag = float(np.linalg.norm(tm.bounds[1] - tm.bounds[0]))
        m = metrics(surface_samples(mesh, POINTS, seed=1), (tp, tn), visible, diag)
        yaw_err = None
        if info.get("yaw_deg") is not None:
            y = math.radians(info["yaw_deg"])
            x_axis = np.array([math.cos(y), 0, -math.sin(y)]) if mode == "gt" else R_total @ np.array([math.cos(y), 0, -math.sin(y)])
            e = (math.degrees(math.atan2(-x_axis[2], x_axis[0])) - true_obj["yaw_deg"]) % 90
            yaw_err = round(min(e, 90 - e), 2)
        rows.append({"gt": true_obj["id"], "yaw_err_deg": yaw_err, "slug": true_obj["slug"], "name": obj["name"], "visible_fraction": round(float(visible.mean()), 4),
                     "pixels": int(gt_mask.sum()), "diag_m": round(diag, 4), "faces": int(len(mesh.faces)), "primitive": info.get("primitive"), **m})
    return {"view": os.path.basename(world), "scale": scale, "objects": rows}


KEYS = ["yaw_err_deg", "acc", "comp", "comp_visible", "comp_hidden", "chamfer", "precision", "recall", "recall_hidden", "f", "nc", "nc_near", "faces"]


def summarize(views):
    rows = [r for v in views for r in v.get("objects", []) if not r.get("empty")]
    out = {"objects": len(rows), "empty": sum(r.get("empty", False) for v in views for r in v.get("objects", []))}
    for k in KEYS:
        values = [r[k] for r in rows if r.get(k) is not None]
        out[k] = round(float(np.mean(values)), 5) if values else None
    return out


def bootstrap(views, draws=1000, seed=0):
    rng = np.random.default_rng(seed)
    samples = [summarize([views[i] for i in rng.integers(0, len(views), len(views))]) for _ in range(draws)]
    return {k: [round(float(np.percentile([s[k] for s in samples if s[k] is not None], q)), 5) for q in (2.5, 97.5)] for k in KEYS}


def compare(a, b, draws=1000, seed=0):
    """Paired bootstrap of b - a on the objects both scored, resampling views."""
    def table(result):
        return {v["view"]: {r["gt"]: r for r in v.get("objects", []) if not r.get("empty")} for v in result["views"]}
    ta, tb = table(a), table(b)
    pairs = {v: sorted(set(ta[v]) & set(tb[v])) for v in sorted(set(ta) & set(tb))}
    views = [v for v in pairs if pairs[v]]
    def mean(vs, side, k):
        values = [side[v][g][k] for v in vs for g in pairs[v] if side[v][g].get(k) is not None]
        return float(np.mean(values)) if values else None
    rng = np.random.default_rng(seed)
    out = []
    for k in KEYS:
        base_a, base_b = mean(views, ta, k), mean(views, tb, k)
        if base_a is None or base_b is None:
            continue
        diffs = []
        for _ in range(draws):
            vs = [views[i] for i in rng.integers(0, len(views), len(views))]
            x, y = mean(vs, ta, k), mean(vs, tb, k)
            if x is not None and y is not None:
                diffs.append(y - x)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out.append({"metric": k, "a": round(base_a, 5), "b": round(base_b, 5), "diff_ci95": [round(float(lo), 5), round(float(hi), 5)],
                    "verdict": "tie" if lo <= 0 <= hi else ("b higher" if lo > 0 else "b lower")})
    return {"a": a["name"], "b": b["name"], "views": len(views), "objects": sum(len(pairs[v]) for v in views), "metrics": out}


def surface_ceiling(split, models, voxels=80):
    """What the surface step can reach with a perfect occupancy: each true mesh voxelized in its box frame (voxels
    per longest side), then each surface method, scored against the true mesh."""
    from scipy import ndimage
    seen, rows = set(), []
    for v in evaluate.views(split):
        for o in json.load(open(os.path.join(v, "gt.json")))["objects"]:
            if o.get("class", "furniture") != "furniture" or o["model_path"] in seen or len(seen) >= models:
                continue
            seen.add(o["model_path"])
            tm = true_mesh(o)
            yaw = math.radians(o["yaw_deg"])
            x_axis = BLENDER_TO_YUP @ np.array([math.cos(yaw), math.sin(yaw), 0])
            axes = np.stack([x_axis, [0, 1, 0], np.cross(x_axis, [0, 1, 0])], 1)
            local = tm.vertices @ axes
            lo = local.min(0)
            vox = float((local.max(0) - lo).max()) / voxels
            n = np.ceil((local.max(0) - lo) / vox).astype(int) + 1
            idx = np.clip(np.floor((trimesh.sample.sample_surface(tm, 300000, seed=0)[0] @ axes - lo) / vox).astype(int), 0, n - 1)
            occ = np.zeros(n, bool)
            occ[tuple(idx.T)] = True
            occ = ndimage.binary_fill_holes(ndimage.binary_closing(np.pad(occ, 1))[1:-1, 1:-1, 1:-1] | occ)
            tp, tn = surface_samples(tm, POINTS)
            diag = float(np.linalg.norm(tm.bounds[1] - tm.bounds[0]))
            surfaces = {"smooth": lambda: ps.smooth_surface(occ, lo, vox, False, 10), "voxels": lambda: ps.voxel_surface(occ, lo, vox, False),
                        "planar": lambda: ps.planar_surface(occ, lo, vox, False, lo[1], 12.0, 10)}
            for name, make in surfaces.items():
                Vl, F = make()
                m = metrics(surface_samples(trimesh.Trimesh(Vl @ axes.T, F, process=False), POINTS, seed=1), (tp, tn), np.ones(len(tp), bool), diag)
                rows.append({"model": os.path.basename(o["model_path"]), "surface": name, "faces": len(F), **m})
    summary = {name: {k: round(float(np.mean([r[k] for r in rows if r["surface"] == name])), 5) for k in ("nc", "chamfer", "f", "faces")}
               for name in ("smooth", "voxels", "planar")}
    return {"split": split, "models": len(seen), "voxels": voxels, "summary": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--split", required=True)
    s.add_argument("--mode", choices=["gt", "pred"], default="gt")
    s.add_argument("--name", required=True)
    s.add_argument("--args", default="", help="primitive_scene.py options of the variant (quoted)")
    s.add_argument("--limit", type=int)
    s.add_argument("--workers", type=int, default=os.cpu_count())
    g = sub.add_parser("ceiling", help="surface methods on the true shapes (perfect occupancy)")
    g.add_argument("--split", default="train")
    g.add_argument("--models", type=int, default=40)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = parser.parse_args()
    if args.command == "ceiling":
        result = surface_ceiling(args.split, args.models)
        json.dump(result, open(os.path.join(RESULTS, f"shape-surface-ceiling-{args.split}.json"), "w"), indent=1)
        print(json.dumps(result["summary"]))
        return
    if args.command == "compare":
        result = compare(json.load(open(args.a)), json.load(open(args.b)))
        print(json.dumps(result, indent=1))
        return
    view_dirs = evaluate.views(args.split)[: args.limit]
    jobs = [(v, args.mode, args.args) for v in view_dirs]
    with Pool(args.workers) as pool:
        views = pool.map(score_view, jobs, chunksize=1)
    summary = summarize(views)
    out = {"name": args.name, "split": args.split, "mode": args.mode, "args": args.args, "summary": summary,
           "ci95": bootstrap([v for v in views if "objects" in v]), "views": views}
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"shape-{args.name}-{args.mode}-{args.split}.json")
    json.dump(out, open(path, "w"), indent=1)
    if args.split == "test":
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        with open(os.path.join(RESULTS, "test-reads.log"), "a") as f:
            f.write(json.dumps({"time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                                "name": f"shape-{args.name}-{args.mode}", "commit": commit, "summary": summary}) + "\n")
    print(json.dumps({"file": os.path.relpath(path, REPO), **summary}))


if __name__ == "__main__":
    main()
