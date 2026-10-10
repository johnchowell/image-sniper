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
    parser.add_argument("--fill", type=float, default=0.0, help="power (W) of an unseen light along the estimated dominant light direction (light from outside the frame)")
    parser.add_argument("--window-strength", type=float, default=12.0, help="emission multiplier for window textures")
    parser.add_argument("--light-power", type=float, default=0.0, help="extra area lights per layout emitter, W per m^2 (0: windows light the room through their emission)")
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
    for mat in bpy.data.materials:  # window textures glow brighter than 1
        if not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED" and node.inputs["Emission Color"].is_linked:
                node.inputs["Emission Strength"].default_value = args.window_strength
            if node.type == "EMISSION" and node.inputs["Color"].is_linked:
                node.inputs["Strength"].default_value = args.window_strength

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
    if dominant and args.fill > 0:
        # Light the photo receives from outside the frame (an unseen window or lamp), placed where the shading's
        # dominant direction points, 2.5 m from the room's middle at camera height, facing back into the room.
        source_cam = G @ Matrix(layout["camera"]["matrix_world_from_camera_opencv"])
        middle = source_cam.translation + (source_cam.to_3x3() @ Vector((0, 0, 1))) * 2.0
        d = to_blender(dominant["direction"]).normalized()
        bpy.ops.object.light_add(type="AREA")
        fill = bpy.context.active_object
        fill.data.shape = "SQUARE"
        fill.data.size = 2.0
        fill.data.energy = args.fill
        fill.matrix_world = Matrix.Translation(middle + d * 2.5) @ (-d).to_track_quat("-Z", "Y").to_matrix().to_4x4()
        fill.visible_camera = False
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

    tmp = os.path.abspath(os.path.join(scene_dir, ".render-check.exr"))
    render = render_linear(tmp, size)
    photo = load_rgb(layout["source_image"], size)
    photo_lin = srgb_to_linear(photo)
    exposure = float(np.median(photo_lin @ LUMA) / max(np.median(render @ LUMA), 1e-6))
    # The photo's camera response is unknown: fit photo = a * render^g (linear light, mid-tones only) so the
    # comparison measures the model, not the tone curve. Both PSNRs are reported.
    rl, pl = render @ LUMA, photo_lin @ LUMA
    mid = (photo.max(-1) < 0.97) & (pl > 0.002) & (rl > 1e-5)
    g, log_a = np.polyfit(np.log(rl[mid]), np.log(pl[mid]), 1)
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
    stem = os.path.splitext(glb)[0]
    write_png(stem + "-check.png", sheet)
    write_png(stem + "-render.png", (render_srgb * 255).round().astype(np.uint8))
    json.dump({"scene": glb, "psnr_db": round(psnr, 2), "psnr_exposure_only_db": round(psnr_exposure, 2), "tone_gamma": round(float(g), 3), "mean_abs_err": round(float(diff.mean()), 4), "exposure": exposure,
               "samples": args.samples, "size": size, "ambient": args.ambient, "fill": args.fill, "window_strength": args.window_strength, "light_power": args.light_power},
              open(stem + "-check.json", "w"), indent=2)
    print(json.dumps({"sheet": stem + "-check.png", "psnr_db": round(psnr, 2), "psnr_exposure_only_db": round(psnr_exposure, 2), "tone_gamma": round(float(g), 3), "mean_abs_err": round(float(diff.mean()), 4)}))


if __name__ == "__main__":
    main()
