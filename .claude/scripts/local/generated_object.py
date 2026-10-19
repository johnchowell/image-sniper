"""Generated objects in the primitive scene: the latest image-to-3D asset of an object (object_mesh.py: the cut-out
of the object's mask, TripoSR, canonical frame) registered to what the view shows.

Registration: yaw, scale and position of the asset (its base on the support when the object stands on one) that
explain the view's evidence: the observed surface points of the object lie on the asset's surface, and no part
of the asset lies where the camera sees past it (shape_prior.evidence) or outside the cube that the shape prior
completes (shape_prior.cube_frame: the domain that the evidence box allows). The cost is the mean distance of the
observed points to the asset plus a penalty per meter for the fraction of the asset's surface in either region.
"""
import math
import os
import re

import numpy as np
import trimesh
from scipy import optimize
from scipy.spatial import cKDTree

import shape_prior

INDEXED_GLB = re.compile(r"^(\d+)-(.+)\.glb$")


def load_generated(mask_file, object_id):
    """The latest canonical asset of an object (meters, +Y up, front +Z, base at y=0), or None."""
    world_dir = mask_file.split(os.sep + "output" + os.sep)[0]
    directory = os.path.join(world_dir, "output", object_id)
    best = None
    for name in os.listdir(directory) if os.path.isdir(directory) else []:
        m = INDEXED_GLB.match(name)
        if m and m.group(2) == object_id and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), os.path.join(directory, name))
    if best is None:
        return None
    mesh = trimesh.load(best[1], force="mesh", process=False)
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices, float), faces=np.asarray(mesh.faces), process=False), best[1]


def rotation_y(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def register(mesh, center, size, axes, core, mask, exclude, view, supported, depth_noise, free_weight=0.15, samples=3000, seed=0, stretch=False):
    """Similarity transform (yaw about +Y, uniform scale, translation) of a canonical asset onto the view's
    evidence. Returns the registered mesh (layout frame) and a report. stretch: also a scale along the asset's +Z
    (the generation view axis), the one extent a single-view model does not see and can collapse (a wire basket
    comes out as a 3 cm card)."""
    rng = np.random.default_rng(seed)
    Pc, _ = trimesh.sample.sample_surface(mesh, samples, seed=seed)
    Pc = np.asarray(Pc, float)
    vs, us = np.nonzero(core)
    if len(us) < 30:
        return None, {"registration": "no observed pixels"}
    pick = rng.choice(len(us), min(len(us), 2000), replace=False)
    obs, _ = view.observed(us[pick].astype(float), vs[pick].astype(float))
    obs = obs[np.isfinite(obs).all(1)]
    if len(obs) < 30:
        return None, {"registration": "no observed depth"}
    height = float(Pc[:, 1].max() - Pc[:, 1].min())
    base_y = center[1] - size[1] / 2
    s_box = size[1] / max(height, 1e-6)
    cube_lo, cube_side = shape_prior.cube_frame(center, size, axes, supported)

    def violation(P):
        local = (P - center) @ axes
        beyond = np.any((local < cube_lo) | (local > cube_lo + cube_side), axis=1)
        _, empty, _, _, _ = shape_prior.evidence(P, mask, exclude, view, depth_noise)
        return empty | beyond

    def place(theta, s, t, sz=1.0):
        return (s * Pc * [1.0, 1.0, sz]) @ rotation_y(theta).T + t

    def cost(theta, s, t, sz=1.0):
        P = place(theta, s, t, sz)
        d, _ = cKDTree(P).query(obs, distance_upper_bound=0.15)
        acc = float(np.mean(np.minimum(d, 0.15)))
        return acc + free_weight * float(violation(P).mean())

    def align_translation(theta, s, t, sz=1.0, iterations=6):
        for _ in range(iterations):
            P = place(theta, s, t, sz)
            _, idx = cKDTree(P).query(obs)
            step = np.median(obs - P[idx], axis=0)
            if supported:
                step[1] = 0.0
            t = t + step
        return t

    def start(theta, s, sz=1.0):
        t = np.array([center[0], base_y if supported else center[1] - s * height / 2, center[2]], float)
        return align_translation(theta, s, t, sz)

    depth = float(np.ptp(Pc[:, 2]))
    width = float(max(np.ptp(Pc[:, 0]), height))
    # Only a collapsed asset (depth under a quarter of its width) is stretched: on a normal asset the extra freedom
    # fits depth noise (train: 66 objects better, 57 worse). Candidates up to a box-like depth at most.
    stretch = stretch and depth < 0.25 * width
    stretches = [1.0] + ([f for f in (1.5, 2.5, 4.0, 8.0) if f * depth <= 1.2 * width] if stretch else [])
    candidates = []
    for theta in np.radians(np.arange(0, 360, 15)):
        for f in (0.8, 0.9, 1.0, 1.12, 1.25):
            for sz in stretches:
                t = start(theta, s_box * f, sz)
                candidates.append((cost(theta, s_box * f, t, sz), theta, s_box * f, t, sz))
    candidates.sort(key=lambda c: c[0])
    best = candidates[0]
    for c0, theta0, s0, t0, sz0 in candidates[:3]:
        free_y = not supported
        k_sz = 5 if free_y else 4

        def objective(x):
            theta, log_s = x[0], x[1]
            t = np.array([x[2], x[4] if free_y else t0[1], x[3]])
            sz = sz0 * math.exp(x[k_sz]) if stretch else 1.0
            return cost(theta, s0 * math.exp(log_s), t, sz)

        x0 = [theta0, 0.0, t0[0], t0[2]] + ([t0[1]] if free_y else []) + ([0.0] if stretch else [])
        steps = [math.radians(10), 0.08, 0.05, 0.05] + ([0.05] if free_y else []) + ([0.15] if stretch else [])
        simplex = np.array([x0] + [np.array(x0) + np.eye(len(x0))[k] * steps[k] for k in range(len(x0))])
        res = optimize.minimize(objective, x0, method="Nelder-Mead", options={"initial_simplex": simplex, "maxfev": 250 + 50 * stretch, "xatol": 1e-3, "fatol": 1e-5})
        if res.fun < best[0]:
            x = res.x
            best = (float(res.fun), x[0], s0 * math.exp(x[1]), np.array([x[2], x[4] if free_y else t0[1], x[3]]), sz0 * math.exp(x[k_sz]) if stretch else 1.0)
    c, theta, s, t, sz = best
    V = (s * np.asarray(mesh.vertices) * [1.0, 1.0, sz]) @ rotation_y(theta).T + t
    out = trimesh.Trimesh(vertices=V, faces=mesh.faces, process=False)
    P = place(theta, s, t, sz)
    d, _ = cKDTree(P).query(obs)
    _, empty, _, _, _ = shape_prior.evidence(P, mask, exclude, view, depth_noise)
    local = (P - center) @ axes
    beyond = np.any((local < cube_lo) | (local > cube_lo + cube_side), axis=1)
    return out, {"registration_cost": round(c, 4), "asset_yaw_deg": round(math.degrees(theta) % 360, 2), "scale_vs_box_height": round(s / s_box, 3), "depth_stretch": round(sz, 3),
                 "observed_dist_m_median": round(float(np.median(d)), 4), "free_space_fraction": round(float(empty.mean()), 4),
                 "beyond_cube_fraction": round(float(beyond.mean()), 4)}


def solid(mesh, lo, vox, n, center, axes):
    """Occupancy of a registered mesh in a cube grid (corner lo in the local box frame, n cells of size vox):
    surface cells from dense surface samples, closed shells filled."""
    from scipy import ndimage
    count = int(min(2_000_000, max(50_000, 4 * mesh.area / (0.5 * vox) ** 2)))
    P, _ = trimesh.sample.sample_surface(mesh, count, seed=0)
    local = (np.asarray(P) - center) @ axes
    idx = np.floor((local - lo) / vox).astype(int)
    keep = np.all((idx >= 0) & (idx < n), axis=1)
    occ = np.zeros((n, n, n), bool)
    occ[tuple(idx[keep].T)] = True
    return ndimage.binary_fill_holes(occ)
