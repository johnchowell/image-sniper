#!/usr/bin/env python3
"""ModelNet40 indoor categories as solid voxel grids for the shape prior (shape_prior_train.py).

Each mesh is centered on its bounding box and scaled so its longest side is 1; +Y is up (ModelNet's +Z). The
solid is the voxels of densely sampled surface points, closed once and with inner cavities filled (open meshes
stay shells, which is right for thin parts such as table tops). Writes benchmark/shapes/modelnet40-<res>.npz:
occ (packed bits, res^3 per shape), extent (normalized box size), category, split (ModelNet's train/test).
"""
import argparse
import os
import zipfile
from multiprocessing import Pool

import numpy as np
import trimesh
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SHAPES = os.path.join(REPO, "benchmark", "shapes")
INDOOR = ["bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "chair", "cup", "desk", "dresser", "flower_pot", "glass_box",
          "keyboard", "lamp", "laptop", "mantel", "monitor", "night_stand", "piano", "plant", "radio", "range_hood", "sink", "sofa",
          "stool", "table", "toilet", "tv_stand", "vase", "wardrobe", "xbox"]
ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)


def read_off(text):
    tokens = text.split()
    if tokens[0] == "OFF":
        tokens = tokens[1:]
    elif tokens[0].startswith("OFF"):  # some files join the header and the counts: "OFF490 518 0"
        tokens = [tokens[0][3:]] + tokens[1:]
    nv, nf = int(tokens[0]), int(tokens[1])
    V = np.array(tokens[3:3 + 3 * nv], dtype=np.float64).reshape(nv, 3)
    rest = np.array(tokens[3 + 3 * nv:], dtype=np.int64)
    faces, i = [], 0
    for _ in range(nf):
        k = rest[i]
        poly = rest[i + 1:i + 1 + k]
        faces += [[poly[0], poly[j], poly[j + 1]] for j in range(1, k - 1)]
        i += k + 1
    return V, np.array(faces, dtype=np.int64).reshape(-1, 3)


def voxelize(args):
    name, res = args
    with zipfile.ZipFile(os.path.join(SHAPES, "ModelNet40.zip")) as z:
        V, F = read_off(z.read(name).decode("ascii", "ignore"))
    V = V @ ZUP_TO_YUP.T
    low, high = V.min(0), V.max(0)
    side = float((high - low).max())
    if side <= 0 or len(F) == 0:
        return None
    V = (V - (low + high) / 2) / side
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    if mesh.area <= 0:
        return None
    n = int(min(400000, max(60000, mesh.area * res * res * 6)))
    pts = trimesh.sample.sample_surface(mesh, n, seed=0)[0]
    idx = np.clip(np.floor((pts + 0.5) * res).astype(int), 0, res - 1)
    occ = np.zeros((res, res, res), bool)
    occ[tuple(idx.T)] = True
    occ = ndimage.binary_closing(np.pad(occ, 1)).astype(bool)[1:-1, 1:-1, 1:-1] | occ
    occ = ndimage.binary_fill_holes(occ)
    parts = name.split("/")
    return np.packbits(occ.reshape(-1)), (high - low) / side, parts[1], parts[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    with zipfile.ZipFile(os.path.join(SHAPES, "ModelNet40.zip")) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".off") and n.split("/")[1] in INDOOR)[: args.limit]
    with Pool(args.workers) as pool:
        out = [r for r in pool.imap(voxelize, [(n, args.res) for n in names], chunksize=8) if r is not None]
    occ, extent, category, split = zip(*out)
    path = os.path.join(SHAPES, f"modelnet40-{args.res}.npz")
    np.savez_compressed(path, occ=np.stack(occ), extent=np.stack(extent).astype(np.float32), category=np.array(category), split=np.array(split), res=args.res)
    print({"file": os.path.relpath(path, REPO), "shapes": len(occ), "skipped": len(names) - len(occ)})


if __name__ == "__main__":
    main()
