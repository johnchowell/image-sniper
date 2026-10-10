#!/usr/bin/env python3
"""Primitive scene: the layout as textured primitives with PBR maps, an alternative to the TripoSR object meshes.

Room shell: the farthest wall in each direction, the floor and the ceiling, clipped against each other into a
closed room; sides the photo never shows close it with the median wall color. Other structure planes (cabinet
fronts, desk tops, shelves) are panels. Objects are boxes (cylinders for round names). Every surface is a grid
displaced to the measured depth where it is seen (relief), with maps sampled by projecting each texel into the
photo:
  baseColor   lighting-free albedo (of the clean plate for structure, of the photo for objects)
  roughness   from the light stage's non-diffuse share (specular floors and glass read glossy)
  normal      from the depth relief against the primitive (structure)
  emissive    window and lamp regions of the light stage
  alpha       object masks (open shapes such as legs stay open)
Writes worlds/<world>/output/scene/N-primitive-scene.glb (+ request metadata). Prints a JSON summary.
"""
import argparse
import json
import math
import os
import re

import cv2
import numpy as np
import trimesh
from PIL import Image

from common import latest, load_layout, next_index, world_path, write_request

ROUND = re.compile(r"vase|pot\b|carafe|bottle|cup|mug|\bcans?\b|jar|bucket|round|urn|barrel|stool", re.I)
LUMA = np.array([0.2126, 0.7152, 0.0722])


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


class View:
    """The source camera: projects layout-frame points to pixels and gives the observed depth per pixel."""

    def __init__(self, layout, depth_full):
        M = np.array(layout["camera"]["matrix_world_from_camera_opencv"])
        self.R, self.t = M[:3, :3], M[:3, 3]
        K = layout["camera"]["intrinsics_px"]
        self.fx, self.fy, self.cx, self.cy = K["fx"], K["fy"], K["cx"], K["cy"]
        self.W, self.H = layout["camera"]["image_size"]
        self.depth = depth_full
        self.position = np.array(layout["camera"]["position"])

    def project(self, P):
        cam = (np.asarray(P) - self.t) @ self.R
        z = cam[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * cam[..., 0] / z + self.cx
            v = self.fy * cam[..., 1] / z + self.cy
        return u, v, z

    def observed(self, u, v):
        """Observed point (layout frame) along each pixel's ray, from the depth map (nan where none)."""
        ui = np.clip(np.round(u).astype(int), 0, self.W - 1)
        vi = np.clip(np.round(v).astype(int), 0, self.H - 1)
        z = self.depth[vi, ui]
        cam = np.stack([(u - self.cx) / self.fx * z, (v - self.cy) / self.fy * z, z], -1)
        return cam @ self.R.T + self.t, z


def sample(image, u, v):
    """Bilinear sample of an HxWxC float image at pixel coordinates."""
    return cv2.remap(image.astype(np.float32), u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def fill(texture, known, fallback):
    """Fills unknown texels: inpainting near known ones, the fallback color far away."""
    if known.all():
        return texture
    if not known.any():
        texture[:] = fallback
        return texture
    dist = cv2.distanceTransform((~known).astype(np.uint8), cv2.DIST_L2, 3)
    out = texture.copy()
    img8 = np.clip(texture * 255, 0, 255).astype(np.uint8)
    inpainted = cv2.inpaint(img8, (~known).astype(np.uint8), 5, cv2.INPAINT_TELEA).astype(np.float32) / 255
    weight = np.clip(1 - dist / 40.0, 0, 1)[..., None]  # past about 40 texels, the plane's own median color
    out[~known] = (weight * inpainted + (1 - weight) * fallback)[~known]
    return out


def height_to_normal(height, texel_m):
    """Tangent-space normal map (OpenGL convention, +Y up in texture) from a height field in meters."""
    gy, gx = np.gradient(height, texel_m)
    n = np.stack([-gx, gy, np.ones_like(height)], -1)  # texture v grows downward in image rows
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return (n * 0.5 + 0.5).astype(np.float32)


def read_linear16(light, key, size):
    """A 16-bit RGB linear map of the light stage, in linear units. PIL cannot read 16-bit RGB (it truncates to 8
    bits), so OpenCV reads it."""
    raw = cv2.imread(light["files"][key], cv2.IMREAD_UNCHANGED)
    assert raw is not None and raw.dtype == np.uint16, f"{light['files'][key]}: expected 16-bit RGB"
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535 * light["encoding"][key]["value = sample / 65535 *"]
    return cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)


def image_of(array, alpha=None):
    rgb = np.clip(array * 255, 0, 255).astype(np.uint8)
    if alpha is not None:
        return Image.fromarray(np.dstack([rgb, (alpha * 255).astype(np.uint8)]), "RGBA")
    return Image.fromarray(rgb, "RGB")


class Maps:
    """Per-pixel source maps from the photo, the plate and their light estimates."""

    def __init__(self, world, layout, view):
        W, H = view.W, view.H
        self.view = view
        source = layout["source_image"]
        plate = latest(world_path(world, "source"), "plate", ".png")
        self.plate_path = plate[1] if plate else source
        light_dir = world_path(world, "output", "light")
        estimates = []
        for name in sorted(os.listdir(light_dir)):
            if name.endswith("-light.json") and name[0].isdigit():
                estimates.append(json.load(open(os.path.join(light_dir, name))))
        by_source = {os.path.normpath(e["source_image"]): e for e in estimates}  # latest index wins
        self.photo_light = by_source.get(os.path.normpath(source))
        self.plate_light = by_source.get(os.path.normpath(self.plate_path)) or self.photo_light
        load = lambda p, mode="RGB": np.asarray(Image.open(p).convert(mode).resize((W, H), Image.BICUBIC)).astype(np.float32) / 255
        self.photo = load(source)
        self.plate = load(self.plate_path)
        # Albedo with the photo's detail: the photo divided by the light stage's (smooth) diffuse shading. The
        # model's own albedo is computed at 768 px and loses texture detail.
        self.photo_albedo = self.detail_albedo(self.photo, self.photo_light)
        self.plate_albedo = self.detail_albedo(self.plate, self.plate_light)
        self.photo_rough = self.roughness(self.photo_light)
        self.plate_rough = self.roughness(self.plate_light)
        final = [r for e in layout.get("lighting", {}).get("emitters", []) for r in e.get("image_regions", [])]
        self.plate_emit = self.emission(self.photo_light, self.photo, final)  # the layout's emitters are regions of the photo's estimate
        self.photo_emit = np.zeros_like(self.photo)

    def detail_albedo(self, image, light):
        W, H = self.view.W, self.view.H
        if not light:
            return image
        shading = read_linear16(light, "shading", (W, H))
        shading = cv2.GaussianBlur(shading, (0, 0), 2)
        linear = srgb_to_linear(image)
        albedo = linear / np.maximum(shading, 0.02)
        # Keep the model's albedo level (exposure of the division is arbitrary): match median luminance.
        model = srgb_to_linear(np.asarray(Image.open(light["files"]["albedo"]).convert("RGB").resize((W, H))).astype(np.float32) / 255)
        albedo *= np.median(model @ LUMA) / max(np.median(albedo @ LUMA), 1e-6)
        return linear_to_srgb(np.clip(albedo, 0, 1)).astype(np.float32)

    def roughness(self, light):
        """Glossy where the light stage puts non-diffuse light (reflections), matte elsewhere."""
        W, H = self.view.W, self.view.H
        if not light:
            return np.full((H, W), 0.8, np.float32)
        shading, residual = read_linear16(light, "shading", (W, H)), read_linear16(light, "residual", (W, H))
        albedo = srgb_to_linear(np.asarray(Image.open(light["files"]["albedo"]).convert("RGB").resize((W, H))).astype(np.float32) / 255)
        # Gloss against the scene's light level, not as a share of the pixel: on a dark surface a tiny residual is
        # a large share, which made dark matte wood read as glossy.
        level = np.median((albedo * shading + residual) @ LUMA)
        gloss = np.clip((residual @ LUMA) / max(0.5 * level, 1e-6), 0, 1)
        gloss = cv2.GaussianBlur(gloss.astype(np.float32), (0, 0), 3)
        return np.clip(0.9 - 0.7 * gloss, 0.15, 0.9).astype(np.float32)

    def emission(self, light, image, regions):
        W, H = self.view.W, self.view.H
        if not light or not regions:
            return np.zeros((H, W, 3), np.float32)
        ids = np.asarray(Image.open(light["files"]["emitters"]).resize((W, H), Image.NEAREST))
        mask = np.isin(ids, regions).astype(np.float32)
        return image * cv2.GaussianBlur(mask, (0, 0), 1.5)[..., None]


def plane_grid(origin, u_axis, v_axis, size_u, size_v, spacing):
    nu, nv = max(2, int(math.ceil(size_u / spacing)) + 1), max(2, int(math.ceil(size_v / spacing)) + 1)
    a, b = np.meshgrid(np.linspace(0, size_u, nu), np.linspace(0, size_v, nv))
    P = origin + a[..., None] * u_axis + b[..., None] * v_axis
    faces = []
    for j in range(nv - 1):
        for i in range(nu - 1):
            k = j * nu + i
            faces += [[k, k + 1, k + nu + 1], [k, k + nu + 1, k + nu]]
    uv = np.stack([a / size_u, b / size_v], -1)  # trimesh UVs: origin bottom-left (flipped to glTF on export)
    return P, np.array(faces), uv


def relief(height, z, args):
    """Relief that the depth can carry: monocular depth ripples by about depth_noise * z, which a normal map turns
    into false streaks under grazing light. Height below 3 sigma of that noise is set to 0, larger height keeps its
    excess (soft threshold, no step at the edge)."""
    if args.no_relief:
        return np.zeros(np.shape(height), np.float32)
    floor = 3 * args.depth_noise * np.nan_to_num(z)
    height = np.nan_to_num(height)
    return (np.sign(height) * np.maximum(np.abs(height) - floor, 0)).astype(np.float32)


def texel_for(view, origin, u_axis, v_axis, normal, size_u, size_v, valid_px, args):
    """Texel size that keeps the photo's detail: the footprint of one photo pixel on the plane at the near end (10th
    percentile) of the seen part. The footprint is z / f across the view direction (only the other direction is
    stretched by the view angle, and a square texel must resolve both). A fixed size blurs near surfaces."""
    a, b = np.meshgrid(np.arange(0.025, size_u, 0.05), np.arange(0.025, size_v, 0.05))
    P = origin + a[..., None] * u_axis + b[..., None] * v_axis
    u, v, z = view.project(P)
    inside = (z > 0.05) & (u >= 0) & (u < view.W - 1) & (v >= 0) & (v < view.H - 1)
    if not inside.any():
        return args.texel_m
    ok = valid_px[np.clip(np.round(np.where(inside, v, 0)).astype(int), 0, view.H - 1), np.clip(np.round(np.where(inside, u, 0)).astype(int), 0, view.W - 1)]
    pick = inside & ok if (inside & ok).any() else inside
    footprint = z[pick] / view.fx
    return float(np.clip(np.percentile(footprint, 10), args.texel_min_m, args.texel_m))


def textured_surface(name, maps, view, origin, u_axis, v_axis, normal, size_u, size_v, valid_px, args, structure=True, fallback=None, cutout=False, emissive=False):
    """A planar primitive with relief and PBR maps. valid_px: pixels whose depth belongs to this surface."""
    texel = max(texel_for(view, origin, u_axis, v_axis, normal, size_u, size_v, valid_px, args), max(size_u, size_v) / args.max_texture)
    tw, th = max(4, int(round(size_u / texel))), max(4, int(round(size_v / texel)))
    a, b = np.meshgrid((np.arange(tw) + 0.5) / tw * size_u, (np.arange(th)[::-1] + 0.5) / th * size_v)
    P = origin + a[..., None] * u_axis + b[..., None] * v_axis
    u, v, z = view.project(P)
    inside = (z > 0.05) & (u >= 0) & (u < view.W - 1) & (v >= 0) & (v < view.H - 1)
    uu, vv = np.where(inside, u, 0), np.where(inside, v, 0)
    obs, obs_z = view.observed(uu, vv)
    height = (obs - P) @ normal  # observed surface relative to the primitive, along its normal
    ok_px = valid_px[np.clip(np.round(vv).astype(int), 0, view.H - 1), np.clip(np.round(uu).astype(int), 0, view.W - 1)]
    seen = inside & ok_px & (np.abs(height) < args.relief_m)
    albedo_src = maps.plate_albedo if structure else maps.photo_albedo
    rough_src = maps.plate_rough if structure else maps.photo_rough
    emit_src = maps.plate_emit if structure else maps.photo_emit
    color = sample(albedo_src, uu, vv)
    emit = np.where(seen[..., None] & emissive, sample(emit_src, uu, vv), 0).astype(np.float32)
    if emissive and seen.any():
        # Glass behind things in front of it (hanging plants) glows like the glass around it.
        emit = fill(emit, seen, np.zeros(3, np.float32)).astype(np.float32)
    lit = emit.max(-1) > 0.02
    if fallback is None:
        wall = seen & ~lit  # the surface's own color: glass texels are not wall
        fallback = np.median(color[wall], 0) if wall.any() else np.array([0.6, 0.6, 0.6])
    color = fill(np.where(seen[..., None], color, 0), seen, fallback)
    rough = np.where(seen, sample(rough_src, uu, vv), 0.8).astype(np.float32)
    # Windows are glass: no diffuse color of their own; the photo's light comes from the emission.
    glow = np.clip(emit.max(-1) * 4, 0, 1)[..., None]
    color = color * (1 - glow) + 0.04 * glow
    h = relief(np.where(seen, height, 0), np.where(seen, obs_z, 0), args)
    h = cv2.GaussianBlur(h, (0, 0), args.relief_sigma_m / texel)
    normal_map = height_to_normal(h, texel)

    # Geometry: a coarser grid displaced by the relief where seen.
    G, faces, uv = plane_grid(origin, u_axis, v_axis, size_u, size_v, args.grid_m)
    gu, gv, gz = view.project(G)
    gin = (gz > 0.05) & (gu >= 0) & (gu < view.W - 1) & (gv >= 0) & (gv < view.H - 1)
    gobs, gobs_z = view.observed(np.where(gin, gu, 0), np.where(gin, gv, 0))
    gh = (gobs - G) @ normal
    gok = valid_px[np.clip(np.round(np.where(gin, gv, 0)).astype(int), 0, view.H - 1), np.clip(np.round(np.where(gin, gu, 0)).astype(int), 0, view.W - 1)]
    keep = gin & gok & (np.abs(gh) < args.relief_m)
    gh = relief(np.where(keep, gh, 0), np.where(keep, gobs_z, 0), args)
    gh = cv2.GaussianBlur(gh.astype(np.float32), (0, 0), 0.7)
    G = G + gh[..., None] * normal
    mesh = trimesh.Trimesh(vertices=G.reshape(-1, 3), faces=faces, process=False)
    if np.dot(mesh.face_normals.mean(0), normal) < 0:
        mesh.faces = mesh.faces[:, ::-1]
    # A panel (a furniture face) exists only where the camera saw it.
    alpha = cv2.dilate(seen.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32) if cutout else None
    material = trimesh.visual.material.PBRMaterial(
        name=name, baseColorTexture=image_of(color, alpha), alphaMode="MASK" if cutout else None, alphaCutoff=0.5 if cutout else None,
        metallicRoughnessTexture=image_of(np.dstack([np.zeros_like(rough), rough, np.zeros_like(rough)])),
        normalTexture=image_of(normal_map), metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True,
        emissiveTexture=image_of(emit) if emit.max() > 0.02 else None, emissiveFactor=[1.0, 1.0, 1.0] if emit.max() > 0.02 else None)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv.reshape(-1, 2), material=material)
    return mesh, {"texture_px": [tw, th], "seen_fraction": round(float(seen.mean()), 3)}


def clip_polygon(poly, point, normal):
    """Sutherland-Hodgman: keeps the side of the line (2D) where (x - point) . normal >= 0."""
    out = []
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        da, db = (a - point) @ normal, (b - point) @ normal
        if da >= 0:
            out.append(a)
        if da * db < 0:
            out.append(a + (b - a) * (da / (da - db)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--texel-m", type=float, default=0.01, help="largest texel (surfaces seen from far or not at all)")
    parser.add_argument("--texel-min-m", type=float, default=0.002, help="smallest texel")
    parser.add_argument("--max-texture", type=int, default=4096)
    parser.add_argument("--grid-m", type=float, default=0.04, help="geometry grid spacing for relief")
    parser.add_argument("--relief-m", type=float, default=0.12, help="largest relief accepted from depth")
    parser.add_argument("--relief-sigma-m", type=float, default=0.01, help="smoothing of the relief before the normal map")
    parser.add_argument("--no-relief", action="store_true", help="flat primitives: no displacement, flat normal maps")
    parser.add_argument("--depth-noise", type=float, default=0.01, help="relative noise of the depth (sigma / depth)")
    parser.add_argument("--voxel-m", type=float, default=0.01, help="smallest voxel of the object carving")
    parser.add_argument("--voxels", type=int, default=80, help="voxels along an object's longest side (sets the voxel size)")
    parser.add_argument("--uv-grid-m", type=float, default=0.1, help="largest triangle edge of object meshes (projective UV error stays under a pixel)")
    parser.add_argument("--clutter-min-px", type=int, default=400, help="smallest unexplained part that becomes a primitive")
    parser.add_argument("--clutter-depth-m", type=float, default=0.4, help="largest depth assumed for a clutter part")
    args = parser.parse_args()

    layout_index, layout_path, layout = load_layout(args.world)
    W, H = layout["camera"]["image_size"]
    step = layout["depth"]["grid_step_px"]
    depth_grid = np.asarray(Image.open(layout["files"]["depth_png"])).astype(np.float32) / 1000
    depth = cv2.resize(depth_grid, (W, H), interpolation=cv2.INTER_NEAREST)
    depth[depth <= 0] = np.nan
    view = View(layout, depth)
    maps = Maps(args.world, layout, view)

    # Object masks (listed objects): their depth does not belong to structure.
    object_px = np.zeros((H, W), bool)
    instances = []
    for obj in layout["objects"]:
        for inst in obj["instances"]:
            mask = np.asarray(Image.open(inst["mask_file"]).convert("L").resize((W, H), Image.NEAREST)) > 127
            object_px |= mask
            instances.append((obj, inst, mask))
    structure_px = ~object_px
    # Room surfaces (floor, walls, ceiling) from the layout's label map: never part of an object.
    labels = np.asarray(Image.open(layout["labels"]["file"]).convert("RGB").resize((W, H), Image.NEAREST)).astype(int)
    room_colors = {tuple(e["color"]) for e in layout["labels"]["palette"] if e["class"] in ("floor", "wall", "ceiling")}
    room_px = np.zeros((H, W), bool)
    for color in room_colors:
        room_px |= np.all(labels == color, axis=-1)

    scene = trimesh.Scene()
    report = {"world": args.world, "layout_index": layout_index, "surfaces": [], "objects": []}
    planes = layout["structure"]
    floor = next((p for p in planes if p["class"] == "floor"), None)
    ceiling = next((p for p in planes if p["class"] == "ceiling"), None)
    walls = [p for p in planes if p["class"] == "wall"]
    lit = {e.get("on_surface") for e in layout.get("lighting", {}).get("emitters", [])}  # surfaces holding a light source
    # Shell walls: no other wall of the same orientation behind them (a cabinet front sits in front of the wall).
    shell = []
    for w in walls:
        n = np.array(w["normal"])
        behind = [o for o in walls if o is not w and np.dot(o["normal"], n) > math.cos(math.radians(15))
                  and (np.array(o["center"]) - np.array(w["center"])) @ n < -0.15]
        (shell if not behind else []).append(w)
    panels = [p for p in planes if p not in shell and p is not floor and p is not ceiling]

    # Room polygon on the floor (x, z): a box around everything, clipped by each shell wall's half-plane.
    pts = [np.array(p["center"])[[0, 2]] for p in planes] + [view.position[[0, 2]]]
    lo, hi = np.min(pts, 0) - 1.5, np.max(pts, 0) + 1.5
    poly = [np.array([lo[0], lo[1]]), np.array([hi[0], lo[1]]), np.array([hi[0], hi[1]]), np.array([lo[0], hi[1]])]
    owner = {}
    for w in shell:
        n2 = np.array(w["normal"])[[0, 2]]
        n2 /= np.linalg.norm(n2)
        if (view.position[[0, 2]] - np.array(w["center"])[[0, 2]]) @ n2 < 0:
            n2 = -n2
        poly = clip_polygon(poly, np.array(w["center"])[[0, 2]], n2)
    room_h = ceiling["center"][1] if ceiling else max([2.5] + [p["center"][1] + p["size"][1] / 2 for p in walls])
    wall_colors = []
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        mid = (a + b) / 2
        match = min(shell, key=lambda w: abs((mid - np.array(w["center"])[[0, 2]]) @ (np.array(w["normal"])[[0, 2]] / np.linalg.norm(np.array(w["normal"])[[0, 2]]))), default=None)
        dist = abs((mid - np.array(match["center"])[[0, 2]]) @ (np.array(match["normal"])[[0, 2]] / np.linalg.norm(np.array(match["normal"])[[0, 2]]))) if match else 1e9
        owner[k] = match["id"] if dist < 0.05 else None
    # Shell walls first (they set the fallback color for unseen walls).
    edges = sorted(range(len(poly)), key=lambda k: owner[k] is None)
    for k in edges:
        a, b = poly[k], poly[(k + 1) % len(poly)]
        length = float(np.linalg.norm(b - a))
        if length < 0.05:
            continue
        u_axis = np.array([b[0] - a[0], 0, b[1] - a[1]]) / length
        inward = np.array([-u_axis[2], 0, u_axis[0]])
        if (view.position - np.array([a[0], 0, a[1]])) @ inward < 0:
            inward = -inward
        fallback = np.median(wall_colors, 0) if (owner[k] is None and wall_colors) else None
        mesh, info = textured_surface(f"room-wall-{k}", maps, view, np.array([a[0], 0, a[1]]), u_axis, np.array([0, 1.0, 0]), inward,
                                      length, room_h, structure_px, args, fallback=fallback, emissive=owner[k] in lit)
        if owner[k]:
            wall_colors.append(np.asarray(mesh.visual.material.baseColorTexture.convert("RGB")).reshape(-1, 3).mean(0) / 255)
        scene.add_geometry(mesh, geom_name=f"room-wall-{k}")
        report["surfaces"].append({"name": f"room-wall-{k}", "plane": owner[k], **info})
    # Floor and ceiling over the room polygon's bounding rectangle (texels outside the room are never seen).
    xs, zs = [p[0] for p in poly], [p[1] for p in poly]
    for name, y, normal in (("room-floor", 0.0, np.array([0, 1.0, 0])), ("room-ceiling", room_h, np.array([0, -1.0, 0]))):
        origin = np.array([min(xs), y, max(zs)])
        mesh, info = textured_surface(name, maps, view, origin, np.array([1.0, 0, 0]), np.array([0, 0, -1.0]), normal,
                                      max(xs) - min(xs), max(zs) - min(zs), structure_px, args,
                                      fallback=None if name == "room-floor" else np.array([0.85, 0.85, 0.83]))
        scene.add_geometry(mesh, geom_name=name)
        report["surfaces"].append({"name": name, **info})
    # Panels: the other planes at their measured extents.
    for p in panels:
        c, u_axis, v_axis = np.array(p["center"]), np.array(p["u_axis"]), np.array(p["v_axis"])
        n = np.array(p["normal"])
        if (view.position - c) @ n < 0:
            n = -n
        origin = c - u_axis * p["size"][0] / 2 - v_axis * p["size"][1] / 2
        mesh, info = textured_surface(f"panel-{p['id']}", maps, view, origin, u_axis, v_axis, n, p["size"][0], p["size"][1], structure_px, args, cutout=True,
                                      emissive=p["id"] in lit)
        if info["seen_fraction"] < 0.15:  # mostly covered by a listed object: that object's own primitive shows it
            report["surfaces"].append({"name": p["id"], "class": p["class"], "skipped": "covered by objects", **info})
            continue
        scene.add_geometry(mesh, geom_name=f"panel-{p['id']}")
        report["surfaces"].append({"name": p["id"], "class": p["class"], **info})

    # Objects: box or cylinder primitives, camera-facing parts displaced to the observed depth, textured by
    # projection into the photo's albedo with the mask as alpha.
    for obj, inst, mask in instances:
        under = mask.copy()
        others = np.zeros_like(mask)
        for _, other, other_mask in instances:
            if other.get("support") == inst["id"]:
                under |= other_mask
            elif other is not inst:
                others |= other_mask
        mesh, info = object_primitive(obj, inst, under, maps, view, args, exclude=room_px | (others & ~under))
        scene.add_geometry(mesh, geom_name=inst["id"])
        report["objects"].append({"id": inst["id"], **info})

    # Clutter: observed surfaces that no plane and no listed object explains (hanging plants, things on desks
    # and shelves). Each connected part becomes a carved primitive too, so it does not stay painted on the plane
    # behind it.
    for k, (mask, inst) in enumerate(clutter_parts(layout, view, object_px, args)):
        mesh, info = object_primitive({"name": "clutter"}, inst, mask, maps, view, args, exclude=object_px)
        if info["faces_seen"] < 0.1:  # the view shows almost none of the box: no evidence for a solid there
            report["objects"].append({"id": inst["id"], "clutter": True, "skipped": "faces mostly unseen", **info})
            continue
        scene.add_geometry(mesh, geom_name=inst["id"])
        report["objects"].append({"id": inst["id"], "clutter": True, "pixels": int(mask.sum()), **info})

    scene_dir = world_path(args.world, "output", "scene")
    n = next_index(scene_dir)
    out = os.path.join(scene_dir, f"{n}-primitive-scene.glb")
    scene.export(out, include_normals=True)
    write_request(os.path.join(scene_dir, f".{n}-primitive-scene-request.json"), {
        "kind": "primitive-scene", "provider": "local/primitives", "endpoint": "local/primitives", "index": n, "status": "completed",
        "input_files": [layout_path, layout["source_image"], maps.plate_path], "output_files": [out],
        "input": {k: getattr(args, k) for k in ("texel_m", "texel_min_m", "max_texture", "grid_m", "relief_m", "relief_sigma_m", "no_relief", "depth_noise", "voxel_m", "voxels", "uv_grid_m", "clutter_min_px", "clutter_depth_m")},
        "maps": {"structure_albedo": maps.plate_light["files"]["albedo"] if maps.plate_light else None,
                 "object_albedo": maps.photo_light["files"]["albedo"] if maps.photo_light else None},
        "result": report})
    print(json.dumps({"scene": out, "surfaces": len(report["surfaces"]), "objects": len(report["objects"]), "bytes": os.path.getsize(out)}))


def greedy_rects(M):
    """Covers the True cells of a 2D array with rectangles (greedy: widest run first, then grown down)."""
    M = M.copy()
    rects = []
    for i, j in zip(*np.nonzero(M)):
        if not M[i, j]:
            continue
        j1 = j
        while j1 + 1 < M.shape[1] and M[i, j1 + 1]:
            j1 += 1
        i1 = i
        while i1 + 1 < M.shape[0] and M[i1 + 1, j:j1 + 1].all():
            i1 += 1
        M[i:i1 + 1, j:j1 + 1] = False
        rects.append((i, j, i1 + 1, j1 + 1))
    return rects


def voxel_surface(occ, lo, vox, drop_bottom):
    """Boundary of a voxel occupancy as merged quads (greedy meshing): a union of box primitives without inner
    faces. Local coordinates; lo is the corner of voxel (0, 0, 0). drop_bottom removes the down faces of the lowest
    layer (they lie on the support and are never seen)."""
    V, F = [], []
    padded = np.pad(occ, 1)
    for d in range(3):
        a_ax, b_ax = [k for k in range(3) if k != d]
        for sgn in (-1, 1):
            neighbor = np.roll(padded, -sgn, axis=d)[1:-1, 1:-1, 1:-1]
            face = occ & ~neighbor
            for c in range(occ.shape[d]):
                if d == 1 and sgn < 0 and c == 0 and drop_bottom:
                    continue
                M = np.take(face, c, axis=d)
                for i0, j0, i1, j1 in greedy_rects(M):
                    corners = []
                    for ia, jb in ((i0, j0), (i1, j0), (i1, j1), (i0, j1)):
                        q = np.zeros(3)
                        q[d] = lo[d] + (c + (sgn > 0)) * vox
                        q[a_ax], q[b_ax] = lo[a_ax] + ia * vox, lo[b_ax] + jb * vox
                        corners.append(q)
                    k = len(V)
                    V += corners
                    tri = [[k, k + 1, k + 2], [k, k + 2, k + 3]]
                    n = np.cross(corners[1] - corners[0], corners[2] - corners[0])
                    if n[d] * sgn < 0:
                        tri = [t[::-1] for t in tri]
                    F += tri
    return np.array(V, float).reshape(-1, 3), np.array(F, int).reshape(-1, 3)


def carve(center, size, axes, mask, exclude, view, args):
    """Occupancy of the object inside its box from one view (space carving with an extrusion prior).

    A voxel is empty when the camera sees past it (observed depth farther than the voxel), or when its pixel shows
    another surface at its depth (room or another object). Voxels on the object's observed surface are full.
    Occluded voxels (behind the object or behind something else) are kept where the front silhouette, extruded
    along the box axis closest to the view direction, says the object is; a silhouette cell is empty when the
    camera sees through its extrusion line (the gap under a desk between its legs)."""
    vox = max(args.voxel_m, float(size.max()) / args.voxels)
    n = np.maximum(1, np.round(size / vox).astype(int))
    lo = -n * vox / 2
    grid = np.stack(np.meshgrid(*[lo[k] + (np.arange(n[k]) + 0.5) * vox for k in range(3)], indexing="ij"), -1)
    P = center + grid @ axes.T
    u, v, z = view.project(P)
    inside = (z > 0.05) & (u >= 0) & (u <= view.W - 1) & (v >= 0) & (v <= view.H - 1)
    ui = np.clip(np.round(np.nan_to_num(u)).astype(int), 0, view.W - 1)
    vi = np.clip(np.round(np.nan_to_num(v)).astype(int), 0, view.H - 1)
    zobs = view.depth[vi, ui]
    known = inside & np.isfinite(zobs)
    tol = 0.02 + 2 * args.depth_noise * np.nan_to_num(zobs)
    in_mask = mask[vi, ui] & inside
    surface = known & in_mask & (np.abs(z - zobs) <= tol)
    free = known & (z < zobs - tol)
    other = known & ~in_mask & (exclude[vi, ui] | (np.abs(z - zobs) <= tol))  # someone else's surface
    empty = free | (other & (z <= zobs + tol))
    # Extrusion axis: the horizontal box axis closest to the view direction.
    view_dir = center - view.position
    d = 0 if abs(axes[:, 0] @ view_dir) > abs(axes[:, 2] @ view_dir) else 2
    seen_through = (free & ~surface).sum(axis=d) >= 2
    silhouette = surface.any(axis=d) | ~seen_through
    silhouette = cv2.morphologyEx(silhouette.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    silhouette = cv2.blur(silhouette.astype(np.float32), (5, 5), borderType=cv2.BORDER_REPLICATE) > 0.5  # majority: straight edges
    occ = (surface | (~empty & np.expand_dims(silhouette, d))) & ~(empty & ~surface)
    # Depth noise makes ragged faces: a 3x3x3 majority vote smooths them to the primitive's planes.
    from scipy import ndimage
    occ = ndimage.uniform_filter(occ.astype(np.float32), 3, mode="nearest") > 0.5
    # Floating crumbs (single voxels from depth noise) go: keep the parts connected to the largest one.
    labels, count = ndimage.label(occ)
    if count > 1:
        sizes = ndimage.sum(occ, labels, range(1, count + 1))
        occ = np.isin(labels, 1 + np.nonzero(sizes >= max(8, 0.02 * sizes.max()))[0])
    return occ, lo, vox


def textured_object(mesh, inst, mask, maps, view, args, alpha_mask, supported):
    """Projective texture on the faces the camera saw (photo albedo crop, alpha from the mask), the object's median
    albedo on hidden faces (back, bottom, occluded), so nothing is pasted twice."""
    mesh = mesh.subdivide_to_size(args.uv_grid_m, max_iter=8)  # projective UVs are affine per triangle: keep them small
    mesh.unmerge_vertices()
    x0, y0, x1, y1 = inst["image_bbox_px"]
    pad = 8
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(view.W, x1 + pad), min(view.H, y1 + pad)
    w, h = x1 - x0, y1 - y0
    crop = maps.photo_albedo[y0:y1, x0:x1]
    rough = maps.photo_rough[y0:y1, x0:x1]
    seen_px = mask[y0:y1, x0:x1]
    base = np.median(crop[seen_px], 0) if seen_px.any() else np.array([0.5, 0.5, 0.5])
    base_rough = float(np.median(rough[seen_px])) if seen_px.any() else 0.8
    # Atlas: the crop, then a strip of the median color for hidden faces.
    strip = 8
    color = np.concatenate([crop, np.broadcast_to(base, (h, strip, 3))], 1)
    alpha = np.concatenate([cv2.dilate(alpha_mask[y0:y1, x0:x1].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32), np.ones((h, strip), np.float32)], 1)
    rough = np.concatenate([rough, np.full((h, strip), base_rough, np.float32)], 1)
    wa = w + strip
    # Per-face visibility: facing the camera, and the observed depth at the face's pixel is the face itself.
    C = mesh.triangles_center
    u, v, z = view.project(C)
    inside = (z > 0.05) & (u >= 0) & (u <= view.W - 1) & (v >= 0) & (v <= view.H - 1)
    ui = np.clip(np.round(np.nan_to_num(u)).astype(int), 0, view.W - 1)
    vi = np.clip(np.round(np.nan_to_num(v)).astype(int), 0, view.H - 1)
    zobs = view.depth[vi, ui]
    facing = np.einsum("ij,ij->i", mesh.face_normals, view.position - C) > 0
    tol = 0.03 + 3 * args.depth_noise * np.nan_to_num(zobs, nan=0)
    visible = inside & facing & ((np.abs(z - zobs) <= tol) | ~np.isfinite(zobs)) & alpha_mask[vi, ui]
    V = np.array(mesh.vertices)
    vu, vv, _ = view.project(V)
    uv = np.stack([(np.clip(vu, x0, x1 - 1) - x0 + 0.5) / wa, 1 - (np.clip(vv, y0, y1 - 1) - y0 + 0.5) / h], 1)
    hidden_uv = np.array([(w + strip / 2) / wa, 0.5])
    # A hidden face takes the color of the nearest visible face of the object (the back of a black leg stays
    # black); the object's median color only when the view shows none of it.
    order = mesh.faces.reshape(-1)
    face_uv = np.full((len(C), 2), hidden_uv)
    seen_face = visible & np.isfinite(u) & np.isfinite(v) & (alpha_mask[vi, ui])
    face_uv[seen_face] = np.stack([(u[seen_face] - x0 + 0.5) / wa, 1 - (v[seen_face] - y0 + 0.5) / h], 1)
    hidden = ~visible
    if seen_face.any() and hidden.any():
        from scipy.spatial import cKDTree
        _, nearest = cKDTree(C[seen_face]).query(C[hidden])
        face_uv[hidden] = face_uv[seen_face][nearest]
    face_hidden = np.repeat(hidden, 3)
    uv[order[face_hidden]] = np.repeat(face_uv[hidden], 3, axis=0)
    uv = np.where(np.isfinite(uv), uv, hidden_uv)
    material = trimesh.visual.material.PBRMaterial(
        name=inst["id"], baseColorTexture=image_of(color, alpha), alphaMode="MASK", alphaCutoff=0.5, doubleSided=False,
        metallicRoughnessTexture=image_of(np.dstack([np.zeros_like(rough), rough, np.zeros_like(rough)])), metallicFactor=0.0, roughnessFactor=1.0)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return mesh, {"faces_seen": round(float(visible.mean()), 3)}


def clutter_parts(layout, view, object_px, args):
    """Pixels whose observed point is off every structure plane (inside its extent) by more than the relief limit
    and the depth noise, outside the listed objects; connected parts of at least clutter_min_px pixels. Each part
    gets a box in the room's frame: its observed extent, as deep as it is wide (to clutter_depth_m) away from the
    camera, since one view shows only its front."""
    H, W = object_px.shape
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    P, z = view.observed(uu, vv)
    best = np.full((H, W), np.inf)
    for p in layout["structure"]:
        c, n, u, v = (np.array(p[k]) for k in ("center", "normal", "u_axis", "v_axis"))
        rel = P - c
        within = (np.abs(rel @ u) <= p["size"][0] / 2 + 0.3) & (np.abs(rel @ v) <= p["size"][1] / 2 + 0.3)
        if p["class"] in ("floor", "ceiling"):
            within[:] = True
        best = np.where(within, np.minimum(best, np.abs(rel @ n)), best)
    off = np.isfinite(z) & ~object_px & (best > np.maximum(args.relief_m, 3 * args.depth_noise * np.nan_to_num(z)))
    # Parts split at depth edges (a jump larger than the noise between neighbors): image adjacency alone joins a
    # plant to the shelf far behind it.
    zf = np.nan_to_num(z)
    jump = np.maximum(np.abs(np.diff(zf, axis=0, prepend=zf[:1])), np.abs(np.diff(zf, axis=1, prepend=zf[:, :1])))
    jump = np.maximum(jump, np.maximum(np.abs(np.diff(zf, axis=0, append=zf[-1:])), np.abs(np.diff(zf, axis=1, append=zf[:, -1:]))))
    off &= jump < 0.02 + 3 * args.depth_noise * zf
    off = cv2.morphologyEx(off.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(off, connectivity=4)
    walls = [p for p in layout["structure"] if p["class"] == "wall"]
    yaw0 = math.degrees(math.atan2(walls[0]["normal"][0], walls[0]["normal"][2])) if walls else 0.0  # box z along the main wall's normal
    parts = []
    for k in np.argsort(-stats[:, 4]):
        if k == 0 or stats[k, 4] < args.clutter_min_px:
            continue
        mask = labels == k
        pts = P[mask]
        yaw = math.radians(yaw0)
        axes = np.stack([[math.cos(yaw), 0, -math.sin(yaw)], [0, 1.0, 0], [math.sin(yaw), 0, math.cos(yaw)]], 1)
        local = pts @ axes
        lo, hi = np.percentile(local, 2, 0), np.percentile(local, 98, 0)
        width = max(hi[0] - lo[0], 0.03)
        to_cam = view.position @ axes
        depth = max(hi[2] - lo[2], min(width, args.clutter_depth_m))
        if to_cam[2] > (lo[2] + hi[2]) / 2:  # extend away from the camera
            lo[2] = hi[2] - depth
        else:
            hi[2] = lo[2] + depth
        lo[1] = max(lo[1], 0.0)
        size = np.maximum(hi - lo, 0.03)
        x, y, w, h = stats[k, :4]
        parts.append((mask, {"id": f"clutter-{len(parts) + 1}", "center": (axes @ ((lo + hi) / 2)).tolist(), "size": size.tolist(),
                             "yaw_deg": yaw0, "image_bbox_px": [int(x), int(y), int(x + w), int(y + h)], "support": "none_detected"}))
    return parts


def object_primitive(obj, inst, mask, maps, view, args, exclude=None):
    """An object as box primitives: a tight box from its observed points, carved to the shape the view shows
    (round names: one cylinder). Returns the mesh and a report."""
    center, size = np.array(inst["center"], float), np.array(inst["size"], float)
    yaw = math.radians(inst["yaw_deg"])
    axes = np.stack([[math.cos(yaw), 0, -math.sin(yaw)], [0, 1.0, 0], [math.sin(yaw), 0, math.cos(yaw)]], 1)  # columns: box x, y, z
    # Tight fit: each box face that faces the camera moves to the observed surface (a robust percentile of the
    # object's points along that axis); faces turned away keep the layout's extent (their side is unseen).
    H, W = mask.shape
    core = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    vs, us = np.nonzero(core)
    if len(us) > 50:
        pick = np.random.default_rng(0).choice(len(us), min(len(us), 20000), replace=False)
        obs, _ = view.observed(us[pick].astype(np.float64), vs[pick].astype(np.float64))
        local = (obs[np.isfinite(obs).all(1)] - center) @ axes
        local = local[np.all(np.abs(local) <= size / 2 + 0.1, axis=1)]
        to_cam = (view.position - center) @ axes
        lo, hi = -size / 2, size / 2
        if len(local) > 50:
            for k in (0, 2):  # the vertical extent stays the layout's (snapped to the support)
                if to_cam[k] > 0:
                    hi[k] = min(hi[k], np.percentile(local[:, k], 98))
                else:
                    lo[k] = max(lo[k], np.percentile(local[:, k], 2))
            if hi[0] - lo[0] > 0.02 and hi[2] - lo[2] > 0.02:
                center = center + axes @ ((lo + hi) / 2)
                size = hi - lo
    # Evidence beyond the mask: pixels whose observed surface lies inside the fitted box along the camera ray
    # (masks miss parts, such as a desk top behind papers), unless the label map says floor, wall or ceiling.
    exclude = np.zeros_like(mask) if exclude is None else exclude
    x0, y0, x1, y1 = inst["image_bbox_px"]
    pad = 40
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(view.W, x1 + pad), min(view.H, y1 + pad)
    vv, uu = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    dirs = np.stack([(uu - view.cx) / view.fx, (vv - view.cy) / view.fy, np.ones_like(uu)], -1) @ view.R.T  # z = 1 per step
    o = (view.position - center) @ axes
    dl = dirs @ axes
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-size / 2 - o) / dl, (size / 2 - o) / dl
    tmin, tmax = np.nanmax(np.minimum(t1, t2), -1), np.nanmin(np.maximum(t1, t2), -1)
    zobs = view.depth[y0:y1, x0:x1]
    tol = 0.03 + 0.02 * np.nan_to_num(zobs)
    inside_box = (tmax >= tmin) & (tmin > 0) & (zobs >= tmin - tol) & (zobs <= tmax + tol)
    evidence = np.zeros_like(mask)
    evidence[y0:y1, x0:x1] = inside_box & ~exclude[y0:y1, x0:x1]
    evidence = cv2.morphologyEx(evidence.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    mask = mask | evidence
    inst = {**inst, "image_bbox_px": [int(x0), int(y0), int(x1), int(y1)]}
    supported = inst.get("support") not in (None, "none_detected")
    # The surface continues under what stands on it (a laptop on a desk): close the mask's gaps for the alpha.
    alpha_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
    if ROUND.search(obj["name"]):
        radius = max(size[0], size[2]) / 2
        mesh = trimesh.creation.cylinder(radius=radius, height=size[1], sections=48)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))  # z axis -> y axis
        if supported:  # the cap on the support is never seen
            mesh.update_faces(mesh.face_normals[:, 1] > -0.9)
        mesh = trimesh.Trimesh(vertices=mesh.vertices + center, faces=mesh.faces, process=False)
        info = {"primitive": "cylinder"}
    else:
        occ, lo, vox = carve(center, size, axes, mask, exclude, view, args)
        Vl, F = voxel_surface(occ, lo, vox, drop_bottom=supported)
        mesh = trimesh.Trimesh(vertices=Vl @ axes.T + center, faces=F, process=False)
        info = {"primitive": "carved boxes", "voxel_m": round(vox, 4), "voxels": int(occ.sum()), "occupancy": round(float(occ.mean()), 3),
                "quads": len(F) // 2}
    mesh, tex = textured_object(mesh, inst, mask, maps, view, args, alpha_mask, supported)
    return mesh, {**info, **tex}


if __name__ == "__main__":
    main()
