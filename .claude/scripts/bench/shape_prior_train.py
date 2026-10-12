#!/usr/bin/env python3
"""Trains the shape prior (local/shape_prior.py) on simulated single views of ModelNet40 indoor shapes.

  generate  simulated views: a shape (benchmark/shapes/modelnet40-64.npz) at a random size and yaw on a floor or a
            table top, a wall behind it, other boxes in front and beside it, a random camera (some views cut the
            object at the image border), a depth map with noise, edge smearing, a coarse depth grid and a
            heavy-tailed depth offset of the object, a mask with ragged edges, and a box estimate with errors in
            center, size and yaw. The evidence of the cube around the estimated box (shape_prior.evidence, the
            same code the pipeline runs) is the input; the true occupancy of the cube is the target. ModelNet's train shapes make the training set, its test shapes
            the validation set; no benchmark asset is a ModelNet shape.
  train     fits the 3D U-Net (binary cross entropy) and keeps the weights with the lowest validation loss in
            .claude/models/shape-prior.pt.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, ".claude", "scripts", "local"))
import primitive_scene as ps  # noqa: E402
import shape_prior as sp  # noqa: E402

SHAPES = os.path.join(REPO, "benchmark", "shapes")
SMALL = {"bottle", "bowl", "cup", "flower_pot", "keyboard", "laptop", "radio", "vase", "xbox", "monitor", "lamp", "plant", "glass_box"}
W, H = 640, 480
_DATA = {}


class SimView:
    def __init__(self, R, t, fx, cx, cy, depth):
        self.R, self.t = R, t
        self.fx = self.fy = fx
        self.cx, self.cy = cx, cy
        self.W, self.H = W, H
        self.depth = depth
        self.position = t

    def observed(self, u, v):
        ui = np.clip(np.round(u).astype(int), 0, self.W - 1)
        vi = np.clip(np.round(v).astype(int), 0, self.H - 1)
        z = self.depth[vi, ui]
        cam = np.stack([(u - self.cx) / self.fx * z, (v - self.cy) / self.fy * z, z], -1)
        return cam @ self.R.T + self.t, z

    def project(self, P):
        cam = (np.asarray(P) - self.t) @ self.R
        z = cam[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.fx * cam[..., 0] / z + self.cx, self.fy * cam[..., 1] / z + self.cy, z


def rot_y(a):
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def look_at(position, target, roll):
    f = target - position
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 1.0, 0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)  # image down
    R = np.stack([r, d, f], 1)  # columns: camera x (right), y (down), z (forward) in the world
    c, s = math.cos(roll), math.sin(roll)
    return R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ray_boxes(origin, dirs, boxes):
    """Distance along each ray to the nearest of the oriented boxes (center, half size, rotation); inf if none."""
    best = np.full(dirs.shape[0], np.inf)
    for c, h, R in boxes:
        o = (origin - c) @ R
        d = dirs @ R
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-h - o) / d, (h - o) / d
        tmin = np.nanmax(np.minimum(t1, t2), 1)
        tmax = np.nanmin(np.maximum(t1, t2), 1)
        hit = (tmax >= tmin) & (tmin > 0)
        best = np.where(hit, np.minimum(best, tmin), best)
    return best


def sample(job):
    split, seed = job
    rng = np.random.default_rng(seed)
    if "occ" not in _DATA:
        d = np.load(os.path.join(SHAPES, "modelnet40-64.npz"))
        _DATA.update({k: d[k] for k in ("occ", "extent", "category", "split")})
        _DATA["res"] = int(d["res"])
    ids = np.nonzero(_DATA["split"] == split)[0]
    i = int(rng.choice(ids))
    res = _DATA["res"]
    occ64 = np.unpackbits(_DATA["occ"][i]).reshape(res, res, res).astype(bool)
    extent = _DATA["extent"][i]
    small = _DATA["category"][i] in SMALL
    longest = math.exp(rng.uniform(math.log(0.12), math.log(0.7))) if small else math.exp(rng.uniform(math.log(0.4), math.log(2.4)))
    size_true = extent * longest
    # Support: the floor, or a table top for small things.
    on_table = small and rng.random() < 0.6
    base = rng.uniform(0.4, 0.95) if on_table else 0.0
    yaw = rng.uniform(-math.pi, math.pi)
    R_obj = rot_y(yaw)
    pos = np.array([0.0, base + size_true[1] / 2, 0.0])  # shape center
    # Camera.
    dist = float(np.clip(longest * rng.uniform(1.0, 4.5) + rng.uniform(0.3, 1.2), 0.5, 8.0))
    az = rng.uniform(-math.pi, math.pi)
    cam_h = rng.uniform(0.5, 1.9)
    cam = np.array([dist * math.sin(az), cam_h, dist * math.cos(az)])
    fx = rng.uniform(380, 650)
    target = pos + rng.normal(0, 0.15, 3) * size_true
    R = look_at(cam, target, rng.normal(0, 0.02))
    if rng.random() < 0.25:  # the object at the image border: part of it is cut off
        side = rng.choice(4)
        shift = rng.uniform(0.3, 0.55)
        cx, cy = W / 2 + (shift * W if side == 0 else -shift * W if side == 1 else 0), H / 2 + (shift * H if side == 2 else -shift * H if side == 3 else 0)
    else:
        cx, cy = W / 2 + rng.normal(0, 30), H / 2 + rng.normal(0, 30)
    # Estimated box: the true box with errors (the layout's fit), then the cube around it.
    est_yaw = yaw + rng.normal(0, math.radians(5)) + rng.integers(4) * math.pi / 2
    axes = rot_y(est_yaw)
    k90 = int(round((est_yaw - yaw) / (math.pi / 2))) % 2
    size_est_true = size_true[[2, 1, 0]] if k90 else size_true
    size_est = size_est_true * np.exp(rng.normal(0, 0.15, 3))
    center_est = pos + rng.normal(0, 0.06, 3) * size_true
    if rng.random() < 0.15:  # the layout box is somewhere else (a failed fit)
        a = rng.uniform(-math.pi, math.pi)
        center_est = center_est + np.array([math.cos(a), 0, math.sin(a)]) * rng.uniform(0.3, 1.2)
    supported = rng.random() < 0.9
    if supported:
        center_est[1] = base + size_est[1] / 2
    # Scene boxes: occluders between the camera and the object, neighbors, a table top under small things.
    vox = longest / res
    to_cam = cam - pos
    boxes_other, boxes_room = [], []
    for _ in range(rng.choice([0, 0, 1, 2])):
        t = rng.uniform(0.25, 0.8)
        c = pos + t * to_cam + np.cross(to_cam / np.linalg.norm(to_cam), [0, 1, 0]) * rng.normal(0, 0.5) * longest
        h = np.abs(rng.uniform(0.1, 0.6, 3)) * longest / 2 + 0.02
        c[1] = max(c[1], h[1]) if rng.random() < 0.7 else h[1]
        boxes_other.append((c, h, rot_y(rng.uniform(-math.pi, math.pi))))
    for _ in range(rng.choice([0, 1, 2])):
        ang = rng.uniform(-math.pi, math.pi)
        h = rng.uniform(0.1, 0.6, 3)
        c = pos + np.array([math.cos(ang), 0, math.sin(ang)]) * (longest / 2 + np.max(h) + rng.uniform(0.05, 0.5))
        c[1] = h[1]
        boxes_other.append((c, h, rot_y(rng.uniform(-math.pi, math.pi))))
    if on_table:
        top = np.array([rng.uniform(0.6, 1.6), 0.02, rng.uniform(0.5, 1.0)])
        c = pos.copy()
        c[1] = base - 0.02
        c[[0, 2]] += rng.normal(0, 0.15, 2)
        boxes_other.append((c, top / 2 + np.array([longest / 2, 0, longest / 2]), rot_y(rng.uniform(-math.pi, math.pi))))
    # Render the window around the object and the layout box (generously: the cube comes later).
    r_true = 0.8 * longest + 0.3
    corners = np.concatenate([pos + np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * r_true,
                              center_est + np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * (0.8 * size_est.max())])
    view = SimView(R, cam, fx, cx, cy, None)
    u, v, z = view.project(corners)
    if (z <= 0.05).any():
        x0, y0, x1, y1 = 0, 0, W, H
    else:
        x0, y0 = max(0, int(u.min()) - 4), max(0, int(v.min()) - 4)
        x1, y1 = min(W, int(u.max()) + 5), min(H, int(v.max()) + 5)
    if x1 <= x0 or y1 <= y0:
        return None
    vv, uu = np.mgrid[y0:y1, x0:x1]
    dirs = np.stack([(uu - cx) / fx, (vv - cy) / fx, np.ones(uu.shape)], -1).reshape(-1, 3) @ R.T  # z = 1 per step
    zbuf = np.full(dirs.shape[0], np.inf)
    ident = np.zeros(dirs.shape[0], np.int8)  # 0 room, 1 object, 2 other
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = -cam[1] / dirs[:, 1]
    zbuf = np.where(t_floor > 0, t_floor, np.inf)
    if rng.random() < 0.75:  # a wall behind the object
        n = -to_cam / np.linalg.norm(to_cam)
        n = rot_y(rng.normal(0, 0.4)) @ np.array([n[0], 0, n[2]]) / max(np.linalg.norm(n[[0, 2]]), 1e-6)
        p0 = pos + n * (longest / 2 + rng.uniform(0.0, 1.2))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_wall = ((p0 - cam) @ n) / (dirs @ n)
        zbuf = np.where((t_wall > 0) & (t_wall < zbuf), t_wall, zbuf)
    t_other = ray_boxes(cam, dirs, boxes_other)
    ident = np.where(t_other < zbuf, 2, ident)
    zbuf = np.minimum(zbuf, t_other)
    # The object: splats of its shell voxels.
    from scipy import ndimage
    shell = occ64 & ~ndimage.binary_erosion(occ64)
    ijk = np.argwhere(shell)
    Pn = ((ijk + 0.5) / res - 0.5) * longest  # normalized frame: longest side 1, centered
    Pw = pos + Pn @ R_obj.T
    pu, pv, pz = view.project(Pw)
    ok = (pz > 0.05) & (pu >= x0 - 20) & (pu < x1 + 20) & (pv >= y0 - 20) & (pv < y1 + 20)
    pu, pv, pz = pu[ok], pv[ok], pz[ok] - vox / 2
    obj = np.full((y1 - y0, x1 - x0), np.inf)
    if len(pz):
        half = np.clip(np.ceil(0.6 * vox * fx / pz), 1, 12).astype(int)
        for hh in np.unique(half):
            sel = half == hh
            for dy in range(-hh, hh + 1):
                for dx in range(-hh, hh + 1):
                    x = np.round(pu[sel]).astype(int) + dx - x0
                    y = np.round(pv[sel]).astype(int) + dy - y0
                    good = (x >= 0) & (x < x1 - x0) & (y >= 0) & (y < y1 - y0)
                    np.minimum.at(obj, (y[good], x[good]), pz[sel][good])
    obj = obj.reshape(-1)
    ident = np.where(obj < zbuf, 1, ident)
    zbuf = np.minimum(zbuf, obj).reshape(y1 - y0, x1 - x0)
    ident = ident.reshape(y1 - y0, x1 - x0)
    # Depth errors: per-pixel noise, a smooth bias, smearing across depth edges, a coarse grid.
    z = zbuf.copy()
    finite = np.isfinite(z)
    z[~finite] = 50.0
    gy, gx = np.mgrid[0:z.shape[0], 0:z.shape[1]] / 100.0
    bias = 1 + rng.normal(0, 0.015) * gx + rng.normal(0, 0.015) * gy + rng.normal(0, 0.01)
    z = z * bias * (1 + rng.normal(0, rng.uniform(0.002, 0.012), z.shape))
    # The object's own depth offset against the room: heavy-tailed, as monocular depth on the benchmark's train
    # split (per-object log bias |b| median 0.019, p90 0.118).
    z = np.where(ident == 1, z * math.exp(rng.normal(0, 0.15 if rng.random() < 0.2 else 0.02)), z)
    if rng.random() < 0.6:
        k = int(rng.choice([3, 5]))
        edge = (cv2.dilate(z.astype(np.float32), np.ones((3, 3))) - cv2.erode(z.astype(np.float32), np.ones((3, 3)))) > 0.05 * z
        z = np.where(edge, cv2.blur(z.astype(np.float32), (k, k)), z)
    step = int(rng.choice([1, 2, 4]))
    if step > 1:
        z = cv2.resize(z[::step, ::step].astype(np.float32), (z.shape[1], z.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = (ident == 1).astype(np.uint8)
    r = int(rng.choice([0, 0, 1, 2, 3, 5, 8]))
    if r:
        mask = (cv2.dilate if rng.random() < 0.5 else cv2.erode)(mask, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    depth = np.full((H, W), np.nan, np.float32)
    depth[y0:y1, x0:x1] = np.where(finite, z, np.nan)
    m_full = np.zeros((H, W), bool)
    m_full[y0:y1, x0:x1] = mask > 0
    ex_full = np.zeros((H, W), bool)
    ex_full[y0:y1, x0:x1] = (ident != 1) & (mask == 0)
    view.depth = depth
    core = cv2.erode(m_full.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    center_est, size_est = ps.evidence_box(center_est, size_est, axes, core, view, supported)
    lo, L = sp.cube_frame(center_est, size_est, axes, supported)
    P, local = sp.grid_points(center_est, axes, lo, L, sp.GRID)
    surface, empty, outside, behind, _ = sp.evidence(P.reshape(-1, 3), m_full, ex_full, view, 0.01)
    shape = (sp.GRID,) * 3
    view_dir = (center_est - cam) @ axes
    # Target: the true occupancy at the cube's cells (4 sub-samples per cell, majority).
    sub = []
    for off in np.array([[-1, -1, -1], [1, 1, 1], [-1, 1, 1], [1, -1, -1]]) * 0.25 * L / sp.GRID:
        q = ((P.reshape(-1, 3) + off @ axes.T - pos) @ R_obj) / longest + 0.5
        idx = np.floor(q * res).astype(int)
        good = np.all((idx >= 0) & (idx < res), 1)
        val = np.zeros(len(q), bool)
        val[good] = occ64[tuple(idx[good].T)]
        sub.append(val)
    target = (np.sum(sub, 0) >= 2).reshape(shape)
    if not surface.any():
        return None
    bits = np.packbits(np.stack([surface, empty, outside]).reshape(3, -1), axis=1)
    behind8 = np.round(np.clip(behind / L, 0, 1) * 255).astype(np.uint8)
    return bits, behind8, (view_dir / np.linalg.norm(view_dir)).astype(np.float32), np.packbits(target.reshape(-1))


def unpack(bits, behind8, view_dir):
    """Network input (B, CHANNELS, n, n, n) from stored samples, through shape_prior.features."""
    n = sp.GRID
    out = []
    for b, h, d in zip(bits, behind8, view_dir):
        s, e, o = np.unpackbits(b, axis=1).reshape(3, n, n, n).astype(bool)
        out.append(sp.features(s, e, o, h.reshape(n, n, n).astype(np.float32) / 255, 1.0, d))
    return np.stack(out)


def generate(args):
    seeds = [(args.split, args.seed * 10_000_000 + k) for k in range(args.count * 2)]
    rows = []
    t0 = time.time()
    with Pool(args.workers) as pool:
        for out in pool.imap(sample, seeds, chunksize=16):
            if out is None:
                continue
            rows.append(out)
            if len(rows) % 2000 == 0:
                print(len(rows), f"{time.time() - t0:.0f}s", flush=True)
            if len(rows) >= args.count:
                pool.terminate()
                break
    bits, behind, view_dir, target = (np.stack(c) for c in zip(*rows))
    path = os.path.join(SHAPES, f"prior-{args.split}-{args.seed}.npz")
    np.savez(path, bits=bits, behind=behind, view_dir=view_dir, target=target)
    print(json.dumps({"file": os.path.relpath(path, REPO), "samples": len(rows), "seconds": round(time.time() - t0)}))


def train(args):
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    data = [np.load(os.path.join(SHAPES, f)) for f in args.train]
    D = {k: np.concatenate([d[k] for d in data]) for k in ("bits", "behind", "view_dir", "target")}
    V = dict(np.load(os.path.join(SHAPES, args.val)))
    n = sp.GRID

    def batch(S, idx, augment):
        x = unpack(S["bits"][idx], S["behind"][idx], S["view_dir"][idx])
        y = np.unpackbits(S["target"][idx], axis=1).reshape(len(idx), 1, n, n, n).astype(np.float32)
        if augment:  # mirror x and/or z: the scene mirrored is another valid scene (view direction channel flips)
            for b in range(len(idx)):
                for axis, ch in ((0, 5), (2, 7)):
                    if rng.random() < 0.5:
                        x[b] = np.flip(x[b], axis + 1)
                        x[b, ch] = -x[b, ch]
                        y[b] = np.flip(y[b], axis + 1)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))

    model = sp.network()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = int(args.minutes * 60)
    sched_t0 = time.time()
    best, step, log = float("inf"), 0, []
    out_path = os.path.abspath(sp.WEIGHTS)
    best_path = os.path.join(SHAPES, "shape-prior-best.pt")  # checkpoints stay out of the repository until the end
    val_idx = np.arange(min(len(V["target"]), 512))

    def validate():
        model.eval()
        losses, ious = [], {th: [] for th in (0.2, 0.3, 0.5)}
        with torch.no_grad():
            for k in range(0, len(val_idx), 16):
                x, y = batch(V, val_idx[k:k + 16], False)
                logit = model(x)
                losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(logit, y)) * len(x))
                unknown = (x[:, 0:1] < 0.5) & (x[:, 1:2] < 0.5)  # cells the view does not decide
                for th in ious:
                    p = torch.sigmoid(logit) > th
                    inter = ((p & (y > 0.5)) & unknown).sum((1, 2, 3, 4)).float()
                    union = ((p | (y > 0.5)) & unknown).sum((1, 2, 3, 4)).float().clamp(min=1)
                    ious[th] += (inter / union).tolist()
        model.train()
        return sum(losses) / len(val_idx), {f"iou_unknown_{th}": round(float(np.mean(v)), 4) for th, v in ious.items()}

    while time.time() - sched_t0 < total:
        frac = (time.time() - sched_t0) / total
        for g in opt.param_groups:
            g["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * min(frac, 1)))
        idx = rng.integers(0, len(D["target"]), args.batch)
        x, y = batch(D, idx, True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 250 == 0:
            vl, viou = validate()
            log.append({"step": step, "minutes": round((time.time() - sched_t0) / 60, 1), "train_loss": round(loss.item(), 4), "val_loss": round(vl, 4), **viou})
            print(json.dumps(log[-1]), flush=True)
            if vl < best:
                best = vl
                torch.save(model.state_dict(), best_path)
    vl, viou = validate()
    log.append({"step": step, "val_loss": round(vl, 4), **viou, "final": True})
    if vl < best:
        torch.save(model.state_dict(), best_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.copyfile(best_path, out_path)
    json.dump({"args": vars(args), "samples": int(len(D["target"])), "log": log}, open(os.path.join(REPO, "benchmark", "results", "shape-prior-train-log.json"), "w"), indent=1)
    print(json.dumps({"weights": os.path.relpath(out_path, REPO), "best_val_loss": round(min(best, vl), 4), "steps": step}))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--split", choices=["train", "test"], default="train")
    g.add_argument("--count", type=int, default=1000)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--workers", type=int, default=os.cpu_count())
    t = sub.add_parser("train")
    t.add_argument("--train", nargs="+", required=True)
    t.add_argument("--val", required=True)
    t.add_argument("--threads", type=int, default=os.cpu_count())
    t.add_argument("--minutes", type=float, default=90)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()
    generate(args) if args.command == "generate" else train(args)


if __name__ == "__main__":
    main()
