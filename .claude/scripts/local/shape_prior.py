"""Shape completion for objects seen from one view: an evidence volume, a learned occupancy prior, flat surfaces.

Evidence: each voxel of a cube around the object's box is projected into the view. It is surface (on the
object's mask at the observed depth), empty (the camera sees past it, or another surface is at its depth), or
unknown (behind what the camera sees, or outside the image).

Prior: a 3D U-Net (trained by .claude/scripts/bench/shape_prior_train.py on simulated views of ModelNet40 indoor
shapes, none of them in the benchmark) maps the evidence to occupancy, so hidden legs, back sides and the parts
behind other objects follow the shapes furniture has.

Flat surfaces: the mesh's faces are grouped into planar regions (normals within a few degrees, snapped to the
object's box axes when close), each region gets a fitted plane, and every vertex moves onto the planes of its
faces (corners and edges where planes meet).
"""
import math
import os

import numpy as np

GRID = 32
PAD = 1.3  # cube side / longest box side
CHANNELS = 8
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "shape-prior.pt")


def cube_frame(center, size, axes, supported):
    """Cube around the box: side PAD * longest side, on the support when the object stands on one. Returns the
    cube's corner (local box coordinates) and side."""
    L = PAD * float(np.max(size))
    lo = np.array([-L / 2, -L / 2, -L / 2])
    if supported:
        lo[1] = -size[1] / 2 - 0.02 * L
    return lo, L


def grid_points(center, axes, lo, L, n):
    vox = L / n
    ax = [lo[k] + (np.arange(n) + 0.5) * vox for k in range(3)]
    local = np.stack(np.meshgrid(*ax, indexing="ij"), -1)
    return center + local @ axes.T, local


def evidence(P, mask, exclude, view, depth_noise):
    """Per point: surface, empty, outside (no pixel), how far behind the observed depth an unknown point lies (m),
    and the signed depth offset to the observed surface (nan without depth)."""
    u, v, z = view.project(P)
    inside = (z > 0.05) & (u >= 0) & (u <= view.W - 1) & (v >= 0) & (v <= view.H - 1)
    ui = np.clip(np.round(np.nan_to_num(u)).astype(int), 0, view.W - 1)
    vi = np.clip(np.round(np.nan_to_num(v)).astype(int), 0, view.H - 1)
    zobs = view.depth[vi, ui]
    known = inside & np.isfinite(zobs)
    zobs = np.nan_to_num(zobs)
    tol = 0.02 + 2 * depth_noise * zobs
    in_mask = mask[vi, ui] & inside
    surface = known & in_mask & (np.abs(z - zobs) <= tol)
    free = known & (z < zobs - tol)
    other = known & ~in_mask & (exclude[vi, ui] | (np.abs(z - zobs) <= tol))  # someone else's surface
    empty = (free | (other & (z <= zobs + tol))) & ~surface
    dz = np.where(known, z - zobs, np.nan)
    behind = np.where(known & ~surface & ~empty, dz, 0.0)
    return surface, empty, ~inside, behind, dz


def features(surface, empty, outside, behind, L, view_dir_local):
    """Network input (CHANNELS, n, n, n) from the cube's evidence."""
    n = surface.shape[0]
    height = np.broadcast_to(((np.arange(n) + 0.5) / n)[None, :, None], surface.shape)
    d = np.asarray(view_dir_local, np.float32) / max(np.linalg.norm(view_dir_local), 1e-9)
    chans = [surface, empty, outside, np.clip(behind / L, 0, 1), height] + [np.full(surface.shape, d[k]) for k in range(3)]
    return np.stack(chans).astype(np.float32)


def network():
    import torch
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.GroupNorm(4, o), nn.SiLU(), nn.Conv3d(o, o, 3, padding=1), nn.GroupNorm(4, o), nn.SiLU())

    class UNet3D(nn.Module):
        def __init__(self, c=(16, 32, 64, 96)):
            super().__init__()
            self.enc = nn.ModuleList([block(CHANNELS, c[0]), block(c[0], c[1]), block(c[1], c[2]), block(c[2], c[3])])
            self.dec = nn.ModuleList([block(c[3] + c[2], c[2]), block(c[2] + c[1], c[1]), block(c[1] + c[0], c[0])])
            self.out = nn.Conv3d(c[0], 1, 1)

        def forward(self, x):
            skips = []
            for i, b in enumerate(self.enc):
                x = b(x if i == 0 else nn.functional.max_pool3d(x, 2))
                skips.append(x)
            for i, b in enumerate(self.dec):
                skip = skips[-2 - i]
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
                x = b(torch.cat([x, skip], 1))
            return self.out(x)

    return UNet3D()


_MODELS = {}


def load_model(path=None):
    path = os.path.abspath(path or WEIGHTS)
    if path not in _MODELS:
        import torch
        model = network()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        _MODELS[path] = model
    return _MODELS[path]


def predict(feat, weights=None):
    import torch
    with torch.no_grad():
        logits = load_model(weights)(torch.from_numpy(feat)[None])[0, 0].numpy()
    return 1 / (1 + np.exp(-logits))


def upsample(prob, n_fine):
    """Trilinear resample of a cube grid to n_fine cells per side (cell centers aligned)."""
    from scipy import ndimage
    n = prob.shape[0]
    c = (np.arange(n_fine) + 0.5) * n / n_fine - 0.5
    coords = np.stack(np.meshgrid(c, c, c, indexing="ij"))
    return ndimage.map_coordinates(prob, coords, order=1, mode="nearest")


def planar_snap(V, F, axes, angle_deg=12.0, axis_deg=10.0, min_area=None, dist_tol=None, weight=0.02):
    """Moves the vertices onto fitted planes of the mesh's planar regions. axes: the object's box axes (columns);
    region normals within axis_deg of one are snapped to it. Returns new vertices."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    if len(F) == 0:
        return V
    normals, areas, centers = mesh.face_normals, mesh.area_faces, mesh.triangles_center
    edge = float(np.median(mesh.edges_unique_length)) if len(mesh.edges_unique_length) else 0.01
    min_area = (4 * edge) ** 2 if min_area is None else min_area
    dist_tol = 1.5 * edge if dist_tol is None else dist_tol
    adj = mesh.face_adjacency
    neighbors = [[] for _ in range(len(F))]
    for a, b in adj:
        neighbors[a].append(b)
        neighbors[b].append(a)
    cos_t = math.cos(math.radians(angle_deg))
    region = -np.ones(len(F), int)
    planes = []
    for seed in np.argsort(-areas):
        if region[seed] >= 0:
            continue
        rid = len(planes)
        n_sum = normals[seed] * areas[seed]
        n = normals[seed]
        d = n @ centers[seed]
        stack, members = [seed], [seed]
        region[seed] = rid
        while stack:
            f = stack.pop()
            for g in neighbors[f]:
                if region[g] < 0 and normals[g] @ n > cos_t and abs(n @ centers[g] - d) < dist_tol:
                    region[g] = rid
                    members.append(g)
                    stack.append(g)
                    n_sum = n_sum + normals[g] * areas[g]
                    n = n_sum / np.linalg.norm(n_sum)
                    w = areas[members]
                    d = float(n @ (centers[members] * w[:, None]).sum(0) / w.sum())
        planes.append((n, d, float(areas[members].sum())))
    # Plane fit per large region: normal snapped to a box axis when close, offset from the area-weighted centroid.
    cos_a = math.cos(math.radians(axis_deg))
    fitted = []
    for rid, (n, d, area) in enumerate(planes):
        if area < min_area:
            fitted.append(None)
            continue
        dots = axes.T @ n
        k = int(np.argmax(np.abs(dots)))
        if abs(dots[k]) > cos_a:
            n = axes[:, k] * np.sign(dots[k])
        members = region == rid
        w = areas[members]
        d = float(n @ (centers[members] * w[:, None]).sum(0) / w.sum())
        fitted.append((n, d))
    # Each vertex: least squares onto the planes of its faces, pulled weakly to where it was.
    A = np.repeat(np.eye(3)[None] * weight, len(V), 0)
    b = V * weight
    has = np.array([fitted[r] is not None for r in region])
    if has.any():
        Nf = np.array([fitted[r][0] for r in region[has]])
        Df = np.array([fitted[r][1] for r in region[has]])
        for corner in range(3):
            np.add.at(A, F[has, corner], np.einsum("fi,fj->fij", Nf, Nf) / 3)
            np.add.at(b, F[has, corner], Nf * Df[:, None] / 3)
    return np.linalg.solve(A, b[..., None])[..., 0]


def complete(center, size, axes, mask, exclude, view, supported, depth_noise, n_fine, threshold, weights=None, hint=None, hint_mode="union"):
    """Occupancy of a cube around the box (n_fine cells per side) from the evidence and the learned prior.
    A cell is full where the prior says so and the view does not see past it; cells on the observed surface (within
    a cell of the measured depth) are full whatever the prior says, so thin parts the view shows stay. hint: a
    function (corner, cell size, cells per side) -> occupancy of another shape hypothesis (a generated asset);
    hint_mode says how it joins the prior: replace it, union, or intersect. Returns the occupancy, the cube's
    corner (local box frame), the cell size and the prior's probabilities."""
    lo, L = cube_frame(center, size, axes, supported)
    n = GRID
    P, _ = grid_points(center, axes, lo, L, n)
    s, e, o, b, _ = evidence(P.reshape(-1, 3), mask, exclude, view, depth_noise)
    shape = (n,) * 3
    feat = features(s.reshape(shape), e.reshape(shape), o.reshape(shape), b.reshape(shape), L, (center - view.position) @ axes)
    prob = predict(feat, weights)
    Pf, _ = grid_points(center, axes, lo, L, n_fine)
    vox = L / n_fine
    sf, ef, _, _, dz = evidence(Pf.reshape(-1, 3), mask, exclude, view, depth_noise)
    shape_f = (n_fine,) * 3
    sf, ef, dz = sf.reshape(shape_f), ef.reshape(shape_f), dz.reshape(shape_f)
    pf = upsample(prob, n_fine)
    exact = sf & (np.abs(np.nan_to_num(dz, nan=1e9)) <= vox)
    hidden = pf > threshold
    if hint is not None:
        g = hint(lo, vox, n_fine)
        hidden = {"replace": g, "union": hidden | g, "intersect": hidden & g}[hint_mode]
    occ = exact | (~ef & hidden)
    return occ, lo, vox, prob
