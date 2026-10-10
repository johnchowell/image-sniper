#!/usr/bin/env python3
"""Renders a scene GLB in Blender Cycles from the layout's source camera and two orbit views, matches exposure
to the photo, and writes a comparison sheet (photo | render | absolute difference / orbit left | orbit right)
plus PSNR. Run with the bpy runtime: .venv-render/bin/python .claude/scripts/local/render_check.py --world W.

Lights come from the layout: each light-source emitter becomes an area light shining into the room, emissive
textures glow, the world gives the ambient color of the shading's spherical-harmonic band 0, and a sun follows
the dominant light when its confidence is not low.
Writes next to the scene: N-<scene>-check.png and N-<scene>-check.json.
"""
import argparse
import glob
import json
import math
import os
import struct
import zlib

import bpy
import numpy as np
from mathutils import Matrix, Vector

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
G = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))  # layout (+Y up) -> Blender (+Z up), as the glTF importer
CV_TO_BLENDER_CAMERA = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))


def to_blender(p):
    return G @ Vector(p)


def write_png(path, rgb8):
    h, w, _ = rgb8.shape
    raw = b"".join(b"\x00" + rgb8[y].tobytes() for y in range(h))
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def load_rgb(path, size):
    image = bpy.data.images.load(os.path.abspath(path))
    image.scale(*size)
    data = np.empty(size[0] * size[1] * 4, np.float32)
    image.pixels.foreach_get(data)
    bpy.data.images.remove(image)
    return np.flipud(data.reshape(size[1], size[0], 4))[..., :3]  # sRGB values 0-1 (8-bit sources load unconverted)


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def render_linear(path, size):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(path)
    data = np.empty(size[0] * size[1] * 4, np.float32)
    image.pixels.foreach_get(data)
    bpy.data.images.remove(image)
    os.remove(path)
    return np.flipud(data.reshape(size[1], size[0], 4))[..., :3]


def look_at(eye, target):
    forward = (target - eye).normalized()
    return Matrix.Translation(eye) @ forward.to_track_quat("-Z", "Y").to_matrix().to_4x4()


def main():
    import sys
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--scene", help="scene GLB (default: latest N-primitive-scene.glb)")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--scale", type=float, default=0.5, help="render size relative to the photo")
    parser.add_argument("--ambient", type=float, default=1.0, help="scale of the world ambient (the shading's SH band 0 already contains bounced light the path tracer adds again)")
    parser.add_argument("--fill", type=float, default=-1.0, help="power (W) of an unseen light along the estimated dominant light direction (light from outside the frame); -1 fits it to the photo, 0 leaves it out")
    parser.add_argument("--window-strength", type=float, default=12.0, help="emission multiplier for window textures")
    parser.add_argument("--light-power", type=float, default=0.0, help="extra area lights per layout emitter, W per m^2 (0: windows light the room through their emission)")
    parser.add_argument("--light-basis", type=int, default=1, help="1: fit ceiling and side lights too; 0: window and dominant fill only")
    parser.add_argument("--fit-scale", type=float, default=0.25, help="render size of the light basis relative to the photo")
    parser.add_argument("--fit-samples", type=int, default=16)
    parser.add_argument("--no-emit", default="", help="comma-separated material names whose emission is turned off (experiments)")
    parser.add_argument("--tag", default="", help="suffix for the output files (experiments)")
    args = parser.parse_args(argv)

    scene_dir = os.path.join("worlds", args.world, "output", "scene")
    glb = args.scene or max(glob.glob(os.path.join(scene_dir, "[0-9]*-primitive-scene.glb")), key=lambda p: int(os.path.basename(p).split("-")[0]))
    layouts = glob.glob(os.path.join("worlds", args.world, "output", "layout", "[0-9]*-layout.json"))
    layout = json.load(open(max(layouts, key=lambda p: int(os.path.basename(p).split("-")[0]))))
    W0, H0 = layout["camera"]["image_size"]
    size = (int(W0 * args.scale), int(H0 * args.scale))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module="io_scene_gltf2")
    bpy.ops.import_scene.gltf(filepath=os.path.abspath(glb))
    scene = bpy.context.scene
    off = {name for name in args.no_emit.split(",") if name}
    emissive_nodes = []
    for mat in bpy.data.materials:  # window textures glow brighter than 1
        if not mat.use_nodes or mat.name in off:
            for node in mat.node_tree.nodes if mat.use_nodes else []:
                if node.type == "BSDF_PRINCIPLED" and node.inputs["Emission Color"].is_linked:
                    node.inputs["Emission Strength"].default_value = 0.0
            continue
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED" and node.inputs["Emission Color"].is_linked:
                emissive_nodes.append(node.inputs["Emission Strength"])
            if node.type == "EMISSION" and node.inputs["Color"].is_linked:
                emissive_nodes.append(node.inputs["Strength"])

    lighting = layout.get("lighting", {})
    for e in lighting.get("emitters", []) if args.light_power > 0 else []:
        bpy.ops.object.light_add(type="AREA")
        light = bpy.context.active_object
        light.data.shape = "RECTANGLE"
        light.data.size, light.data.size_y = e["size"]
        light.data.energy = args.light_power * e["size"][0] * e["size"][1]
        n, u = to_blender(e["normal"]), to_blender(e["u_axis"])
        center = to_blender(e["center"]) + n * 0.02
        z = -n  # area lights emit along local -Z: local -Z = n (into the room)
        x = u.normalized()
        y = z.cross(x)
        light.matrix_world = Matrix.Translation(center) @ Matrix((x, y, z)).transposed().to_4x4()
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    sh = lighting.get("irradiance_sh", {}).get("coefficients_rgb")
    ambient = [max(0.0, c) * 0.886 * args.ambient for c in sh[0]] if sh else [0.3 * args.ambient] * 3  # Y00 term -> mean irradiance / pi scale
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (*ambient, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    scene.world = world
    dominant = lighting.get("dominant_light")
    # Light basis for the fit: the window emission, plus area lights the camera cannot see where light from outside
    # the frame can come from (the dominant direction, four ceiling quadrants, the four sides of the room).
    basis = {"window": None}  # name -> light object (None: the emissive textures)

    def area(name, center, facing, size_x, size_y):
        bpy.ops.object.light_add(type="AREA")
        light = bpy.context.active_object
        light.name = name
        light.data.shape = "RECTANGLE"
        light.data.size, light.data.size_y = max(size_x, 0.2), max(size_y, 0.2)
        light.data.energy = 0.0
        light.matrix_world = Matrix.Translation(center) @ (-facing).normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
        light.visible_camera = False
        light.visible_glossy = False
        basis[name] = light

    source_cam = G @ Matrix(layout["camera"]["matrix_world_from_camera_opencv"])
    if dominant and args.fill != 0:
        middle = source_cam.translation + (source_cam.to_3x3() @ Vector((0, 0, 1))) * 2.0
        d = to_blender(dominant["direction"]).normalized()
        area("fill", middle + d * 2.5, -d, 2.0, 2.0)
    floor_obj = bpy.data.objects.get("room-floor")
    ceiling_obj = bpy.data.objects.get("room-ceiling")
    if args.light_basis and floor_obj is not None:
        corners = [floor_obj.matrix_world @ Vector(c) for c in floor_obj.bound_box]
        x0, x1 = min(c.x for c in corners), max(c.x for c in corners)
        y0, y1 = min(c.y for c in corners), max(c.y for c in corners)
        top = max((ceiling_obj.matrix_world @ Vector(c)).z for c in ceiling_obj.bound_box) if ceiling_obj else 2.6
        for i, (fx, fy) in enumerate(((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))):
            area(f"ceiling-{i + 1}", Vector((x0 + fx * (x1 - x0), y0 + fy * (y1 - y0), top - 0.05)), Vector((0, 0, -1)), 0.4 * (x1 - x0), 0.4 * (y1 - y0))
        mid_x, mid_y, h = (x0 + x1) / 2, (y0 + y1) / 2, min(1.4, top / 2)
        area("side-x0", Vector((x0 + 0.1, mid_y, h)), Vector((1, 0, 0)), 0.8 * (y1 - y0), 1.6)
        area("side-x1", Vector((x1 - 0.1, mid_y, h)), Vector((-1, 0, 0)), 0.8 * (y1 - y0), 1.6)
        area("side-y0", Vector((mid_x, y0 + 0.1, h)), Vector((0, 1, 0)), 0.8 * (x1 - x0), 1.6)
        area("side-y1", Vector((mid_x, y1 - 0.1, h)), Vector((0, -1, 0)), 0.8 * (x1 - x0), 1.6)
    if dominant and dominant.get("confidence") in ("medium", "high"):
        bpy.ops.object.light_add(type="SUN")
        sun = bpy.context.active_object
        sun.data.energy = 2.0
        d = to_blender(dominant["direction"])
        sun.matrix_world = Matrix.Translation(d * 5) @ (-d).to_track_quat("-Z", "Y").to_matrix().to_4x4()

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.view_settings.view_transform = "Standard"
    cam_data = bpy.data.cameras.new("camera")
    cam = bpy.data.objects.new("camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    K = layout["camera"]["intrinsics_px"]
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.lens = K["fx"] / W0 * cam_data.sensor_width
    cam_data.shift_x = (W0 / 2 - K["cx"]) / W0
    cam_data.shift_y = (K["cy"] - H0 / 2) / W0
    cam_data.clip_end = 100
    source = G @ Matrix(layout["camera"]["matrix_world_from_camera_opencv"]) @ CV_TO_BLENDER_CAMERA
    cam.matrix_world = source

    UNIT_W = 100.0

    def set_lights(weights):
        for name, light in basis.items():
            w = weights.get(name, 0.0)
            if light is None:
                for socket in emissive_nodes:
                    socket.default_value = w * args.window_strength
            else:
                light.data.energy = w * UNIT_W

    tmp = os.path.abspath(os.path.join(scene_dir, ".render-check.exr"))
    photo = load_rgb(layout["source_image"], size)
    photo_lin = srgb_to_linear(photo)
    pl_full = photo_lin @ LUMA
    # Inverse lighting. Rendering is linear in the light powers: render each basis light alone (small, few
    # samples), then fit non-negative powers w and the camera response, photo = a * (sum w_k B_k)^g, on mid-tone
    # luminance, alternating a least-squares solve for w (photo mapped back through the response) and a log-space
    # line fit for a and g. The window must stay clipped where the photo is clipped.
    fit_size = (max(32, int(W0 * args.fit_scale)), max(32, int(H0 * args.fit_scale)))
    scene.cycles.samples = args.fit_samples
    scene.render.resolution_x, scene.render.resolution_y = fit_size
    names = list(basis)
    B = []
    for name in names:
        set_lights({name: 1.0})
        B.append(render_linear(tmp, fit_size) @ LUMA)
    B = np.stack(B, -1).reshape(-1, len(names))
    small = load_rgb(layout["source_image"], fit_size)
    pl = (srgb_to_linear(small) @ LUMA).reshape(-1)
    clipped = small.max(-1).reshape(-1) >= 0.97
    mid = ~clipped & (pl > 0.002)
    fixed = {}
    if args.fill > 0 and "fill" in basis:
        fixed["fill"] = args.fill / UNIT_W

    def nnls(A, t, iters=3000):
        Gm, h = A.T @ A, A.T @ t
        w = np.zeros(A.shape[1])
        diag = np.maximum(np.diag(Gm), 1e-12)
        for _ in range(iters):
            for k in range(len(w)):
                w[k] = max(0.0, w[k] - (Gm[k] @ w - h[k]) / diag[k])
        return w

    g, log_a = 1.0, 0.0
    weights = {n: 0.0 for n in names}
    weights["window"] = 1.0
    free = [k for k, n in enumerate(names) if n not in fixed]
    for _ in range(6):
        target = np.power(pl[mid] / np.exp(log_a), 1 / g)
        known = sum(B[mid, names.index(n)] * v for n, v in fixed.items()) if fixed else 0.0
        w = nnls(B[mid][:, free], target - known)
        for k, v in zip(free, w):
            weights[names[k]] = float(v)
        for n, v in fixed.items():
            weights[n] = v
        R = B @ np.array([weights[n] for n in names])
        ok = mid & (R > 1e-6)
        g, log_a = np.polyfit(np.log(R[ok]), np.log(pl[ok]), 1)
        # Clipped window pixels must render at least white after the response.
        win = clipped & (weights["window"] * B[:, 0] > 0.5 * R)
        if win.sum() > 20 and weights["window"] > 0:
            need = np.exp(-log_a / g)  # render luminance that maps to 1
            have = np.median(R[win])
            if have < need:
                weights["window"] *= need / have
                fixed["window"] = weights["window"]
                free = [k for k, n in enumerate(names) if n not in fixed]
    set_lights(weights)
    scene.cycles.samples = args.samples
    scene.render.resolution_x, scene.render.resolution_y = size
    render = render_linear(tmp, size)
    rl = render @ LUMA
    ok = (photo.max(-1) < 0.97) & (pl_full > 0.002) & (rl > 1e-5)
    g, log_a = np.polyfit(np.log(rl[ok]), np.log(pl_full[ok]), 1)
    exposure = float(np.median(photo_lin @ LUMA) / max(np.median(render @ LUMA), 1e-6))
    tone = lambda img: img * (np.exp(log_a) * np.power(np.maximum(img @ LUMA, 1e-6), g) / np.maximum(img @ LUMA, 1e-6))[..., None]  # on luminance: keeps color
    toned = tone(render)
    render_srgb = linear_to_srgb(toned)
    psnr_exposure = 10 * math.log10(1 / max(float(np.mean((linear_to_srgb(render * exposure) - photo) ** 2)), 1e-10))
    diff = np.abs(render_srgb - photo).mean(-1)
    mse = float(np.mean((render_srgb - photo) ** 2))
    psnr = 10 * math.log10(1 / max(mse, 1e-10))

    # Orbit views: around a point 3 m ahead of the source camera at mid height, 35 degrees each side, raised.
    eye = source.translation
    forward = -(source.to_3x3() @ Vector((0, 0, 1)))
    pivot = eye + Vector((forward.x, forward.y, 0)).normalized() * 3.0
    pivot.z = 1.0
    orbits = []
    for angle in (-35, 35):
        rel = Matrix.Rotation(math.radians(angle), 4, "Z") @ (eye - pivot)
        cam.matrix_world = look_at(pivot + rel + Vector((0, 0, 0.7)), pivot)
        orbits.append(linear_to_srgb(tone(render_linear(tmp, size))))

    top = np.hstack([photo, render_srgb, np.stack([np.clip(diff * 3, 0, 1)] * 3, -1)])
    bottom = np.hstack([orbits[0], orbits[1], np.zeros_like(photo)])
    sheet = (np.vstack([top, bottom]) * 255).round().astype(np.uint8)
    stem = os.path.splitext(glb)[0] + (f"-{args.tag}" if args.tag else "")
    write_png(stem + "-check.png", sheet)
    write_png(stem + "-render.png", (render_srgb * 255).round().astype(np.uint8))
    json.dump({"scene": glb, "psnr_db": round(psnr, 2), "psnr_exposure_only_db": round(psnr_exposure, 2), "tone_gamma": round(float(g), 3), "mean_abs_err": round(float(diff.mean()), 4), "exposure": exposure, "light_weights_w": {n: round(v * (args.window_strength if n == "window" else UNIT_W), 2) for n, v in weights.items()},
               "samples": args.samples, "size": size, "ambient": args.ambient, "fill": args.fill, "window_strength": args.window_strength, "light_power": args.light_power},
              open(stem + "-check.json", "w"), indent=2)
    # The fitted light rig in the layout frame (for viewers): area lights with power, the window emission strength.
    to_layout = lambda v: [round(v.x, 4), round(v.z, 4), round(-v.y, 4)]
    rig = {"frame": "layout (+Y up)", "window_emission_strength": round(weights.get("window", 0.0) * args.window_strength, 4),
           "exposure_note": "photo_linear = a * render_linear^g", "tone": {"a": round(float(np.exp(log_a)), 6), "g": round(float(g), 4)}, "area_lights": []}
    for name, light in basis.items():
        if light is None or weights.get(name, 0.0) <= 0:
            continue
        facing = light.matrix_world.to_3x3() @ Vector((0, 0, -1))
        rig["area_lights"].append({"name": name, "center": to_layout(light.matrix_world.translation), "normal": to_layout(facing.normalized()),
                                   "up": to_layout((light.matrix_world.to_3x3() @ Vector((0, 1, 0))).normalized()),
                                   "size": [round(light.data.size, 3), round(light.data.size_y if light.data.shape == "RECTANGLE" else light.data.size, 3)],
                                   "power_w": round(weights[name] * UNIT_W, 2)})
    json.dump(rig, open(stem + "-lights.json", "w"), indent=2)
    print(json.dumps({"sheet": stem + "-check.png", "psnr_db": round(psnr, 2), "psnr_exposure_only_db": round(psnr_exposure, 2), "tone_gamma": round(float(g), 3), "mean_abs_err": round(float(diff.mean()), 4),
                      "light_weights_w": {n: round(v * (args.window_strength if n == "window" else UNIT_W), 1) for n, v in weights.items()}}))


if __name__ == "__main__":
    main()
