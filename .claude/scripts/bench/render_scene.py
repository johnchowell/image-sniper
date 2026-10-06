#!/usr/bin/env python3
"""Renders one procedural room with known ground truth (Blender Cycles, run with the bpy venv).

Room: textured floor and walls, white ceiling, 1-2 windows (emissive panels that overexpose and light the room),
optional hidden ceiling fill light, furniture on the floor without overlaps, small props on table tops.
Per view writes rgb.png, gt.npz (planar depth, world normals, albedo, diffuse shading, non-diffuse light,
emission, instance ids), and gt.json (camera, room planes, object boxes with support, windows, lights).
Blender world frame: Z up, meters.
"""
import argparse
import json
import math
import os
import random
import re
import struct
import zlib

import bpy
import numpy as np
from mathutils import Matrix, Vector

SKIP = re.compile(r"wall|ceiling|chandelier|fan|hose|mounted|sconce", re.I)
SMALL_MAX = 0.55  # props up to this size go on table tops
WIDTH, HEIGHT = 640, 480


def clean_name(slug):
    words = re.sub(r"[_\-]+", " ", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", slug)).split()
    return " ".join(w.lower() for w in words if not w.isdigit())


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for module in ("io_scene_gltf2", "cycles"):
        try:
            bpy.ops.preferences.addon_enable(module=module)
        except Exception:
            pass


def textured_material(name, maps, scale_m, tint=None):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = nodes["Principled BSDF"]
    coords = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1 / scale_m, 1 / scale_m, 1 / scale_m)
    links.new(coords.outputs["Object"], mapping.inputs["Vector"])
    if maps and maps.get("Diffuse"):
        image = nodes.new("ShaderNodeTexImage")
        image.image = bpy.data.images.load(os.path.abspath(maps["Diffuse"]))
        image.projection = "BOX"
        links.new(mapping.outputs["Vector"], image.inputs["Vector"])
        links.new(image.outputs["Color"], bsdf.inputs["Base Color"])
    elif tint:
        bsdf.inputs["Base Color"].default_value = (*tint, 1)
    bsdf.inputs["Roughness"].default_value = 0.85
    return material


def add_plane(name, size, location, rotation, material):
    bpy.ops.mesh.primitive_plane_add(size=1, location=location, rotation=rotation)
    plane = bpy.context.active_object
    plane.name = name
    plane.scale = (size[0], size[1], 1)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    plane.data.materials.append(material)
    return plane


def import_model(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.abspath(path))
    new = [o for o in bpy.data.objects if o not in before]
    root = bpy.data.objects.new("root", None)
    bpy.context.scene.collection.objects.link(root)
    for obj in new:
        if obj.parent is None:
            obj.parent = root
    bpy.context.view_layer.update()
    meshes = [o for o in new if o.type == "MESH"]
    corners = np.array([o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box])
    low, high = corners.min(0), corners.max(0)
    # Pivot at the bottom center of the bounding box.
    offset = Vector(((low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]))
    for obj in new:
        if obj.parent is root:
            obj.matrix_world = Matrix.Translation(-offset) @ obj.matrix_world
    bpy.context.view_layer.update()
    return root, meshes, high - low


def footprint(center, size, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    half = np.array([[size[0] / 2, size[1] / 2], [-size[0] / 2, size[1] / 2]])
    reach = np.abs(half @ np.array([[c, s], [-s, c]])).max(0)
    return np.array([center[0] - reach[0], center[1] - reach[1], center[0] + reach[0], center[1] + reach[1]])


def overlaps(a, b, margin=0.1):
    return not (a[2] + margin < b[0] or b[2] + margin < a[0] or a[3] + margin < b[1] or b[3] + margin < a[1])


def write_png(path, rgb8):
    height, width, _ = rgb8.shape
    raw = b"".join(b"\x00" + rgb8[y].tobytes() for y in range(height))
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def read_exr(path, channels):
    image = bpy.data.images.load(path)
    w, h = image.size
    data = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(data)
    bpy.data.images.remove(image)
    return np.flipud(data.reshape(h, w, 4))[..., :channels]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", default="benchmark/assets/manifest.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--objects", type=int, nargs=2, default=[5, 8], metavar=("MIN", "MAX"), help="furniture and props tried")
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 2], help="window counts to choose from")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    manifest = json.load(open(args.assets))
    reset()
    scene = bpy.context.scene

    # Room.
    W, D, H = rng.uniform(3.6, 6.0), rng.uniform(4.0, 7.0), rng.uniform(2.5, 3.1)
    floors = [t for t in manifest["textures"] if t["role"] == "floor"]
    walls = [t for t in manifest["textures"] if t["role"] == "wall"]
    # Textures map at their published real-world tile size (fallback 2 m), so scale cues match real rooms.
    floor_tex = rng.choice(floors) if floors else {}
    wall_tex = rng.choice(walls) if walls else {}
    floor_mat = textured_material("floor", floor_tex.get("maps"), floor_tex.get("tile_m") or 2.0, (0.4, 0.3, 0.2))
    wall_mat = textured_material("wall", wall_tex.get("maps"), wall_tex.get("tile_m") or 2.0, (0.8, 0.78, 0.74))
    ceiling_mat = textured_material("ceiling", None, 1, (0.85, 0.85, 0.85))
    structure = [
        {"id": "floor", "class": "floor", "point": [0, 0, 0], "normal": [0, 0, 1], "size": [W, D],
         "obj": add_plane("floor", (W, D), (0, 0, 0), (0, 0, 0), floor_mat)},
        {"id": "ceiling", "class": "ceiling", "point": [0, 0, H], "normal": [0, 0, -1], "size": [W, D],
         "obj": add_plane("ceiling", (W, D), (0, 0, H), (math.pi, 0, 0), ceiling_mat)},
    ]
    for wall_id, (location, rotation, normal, size) in {
        "wall-back": ((0, D / 2, H / 2), (math.pi / 2, 0, math.pi), (0, -1, 0), (W, H)),
        "wall-front": ((0, -D / 2, H / 2), (math.pi / 2, 0, 0), (0, 1, 0), (W, H)),
        "wall-left": ((-W / 2, 0, H / 2), (math.pi / 2, 0, -math.pi / 2), (1, 0, 0), (D, H)),
        "wall-right": ((W / 2, 0, H / 2), (math.pi / 2, 0, math.pi / 2), (-1, 0, 0), (D, H)),
    }.items():
        structure.append({"id": wall_id, "class": "wall", "point": list(location), "normal": list(normal), "size": list(size),
                          "obj": add_plane(wall_id, size, location, rotation, wall_mat)})

    # Windows: emissive panels on the back or side walls (they light the room and overexpose like daylight).
    emission = bpy.data.materials.new("window")
    emission.use_nodes = True
    nodes = emission.node_tree.nodes
    nodes.remove(nodes["Principled BSDF"])
    emit = nodes.new("ShaderNodeEmission")
    color = (1.0, 0.97, 0.93) if rng.random() < 0.5 else (0.92, 0.96, 1.0)
    emit.inputs["Color"].default_value = (*color, 1)
    emit.inputs["Strength"].default_value = rng.uniform(10, 25)
    emission.node_tree.links.new(emit.outputs["Emission"], nodes["Material Output"].inputs["Surface"])
    windows = []
    for wall in rng.sample([s for s in structure if s["id"] in ("wall-back", "wall-left", "wall-right")], rng.choice(args.windows)):
        width = rng.uniform(1.0, min(2.4, wall["size"][0] - 1.0))
        height = rng.uniform(1.1, min(1.7, H - 1.2))
        along = rng.uniform(-(wall["size"][0] - width) / 2 + 0.3, (wall["size"][0] - width) / 2 - 0.3)
        normal = Vector(wall["normal"])
        tangent = Vector((0, 0, 1)).cross(normal).normalized()
        center = Vector(wall["point"]) + tangent * along + Vector((0, 0, 0.85 + height / 2 - H / 2)) + normal * 0.01
        rotation = Matrix((tangent, Vector((0, 0, 1)), normal)).transposed().to_euler()
        panel = add_plane(f"window-{len(windows) + 1}", (width, height), center, rotation, emission)
        windows.append({"id": panel.name, "wall": wall["id"], "center": list(center), "normal": list(normal),
                        "size": [width, height], "color_linear_rgb": list(color), "strength": emit.inputs["Strength"].default_value})
    lights = []
    if rng.random() < 0.5:
        bpy.ops.object.light_add(type="AREA", location=(rng.uniform(-W / 4, W / 4), rng.uniform(-D / 4, D / 4), H - 0.05))
        fill = bpy.context.active_object
        fill.data.size = 1.0
        fill.data.energy = rng.uniform(80, 200)
        fill.visible_camera = False
        lights.append({"id": "ceiling-fill", "type": "area", "center": list(fill.location), "energy": fill.data.energy})
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.02, 0.02, 0.02, 1)
    scene.world = world

    # Furniture on the floor, props on table tops.
    models = [m for m in manifest["models"] if not SKIP.search(m["slug"])]
    rng.shuffle(models)
    placed, boxes = [], []
    pass_index = 0
    for model in models[: rng.randint(*args.objects)]:
        root, meshes, size = import_model(model["path"])
        if max(size) > 2.6 or max(size) < 0.08:
            continue
        small = max(size) <= SMALL_MAX
        tables = [p for p in placed if "table" in p["name"] and p["support"] == "floor"]
        support, base_z, region = "floor", 0.0, (-W / 2 + 0.3, -D / 2 + 1.6, W / 2 - 0.3, D / 2 - 0.3)
        if small and tables and rng.random() < 0.8:
            table = rng.choice(tables)
            support, base_z = table["id"], table["top_z"]
            region = (table["footprint"][0] + 0.1, table["footprint"][1] + 0.1, table["footprint"][2] - 0.1, table["footprint"][3] - 0.1)
        for _ in range(40):
            yaw = rng.uniform(-math.pi, math.pi)
            x, y = rng.uniform(region[0], region[2]), rng.uniform(region[1], region[3])
            fp = footprint((x, y), size, yaw)
            same_level = [b for b in boxes if b["support"] == support]
            inside = fp[0] >= region[0] - 0.15 and fp[2] <= region[2] + 0.15 and fp[1] >= region[1] - 0.15 and fp[3] <= region[3] + 0.15
            if inside and not any(overlaps(fp, b["footprint"], 0.05 if support != "floor" else 0.15) for b in same_level):
                break
        else:
            for obj in [root] + list(root.children_recursive):
                bpy.data.objects.remove(obj, do_unlink=True)
            continue
        root.location = (x, y, base_z)
        root.rotation_euler = (0, 0, yaw)
        pass_index += 1
        for mesh in meshes:
            mesh.pass_index = pass_index
        bpy.context.view_layer.update()
        corners = np.array([m.matrix_world @ Vector(c) for m in meshes for c in m.bound_box])
        record = {"id": f"{model['slug']}-{pass_index}", "slug": model["slug"], "name": clean_name(model["slug"]),
                  "pass_index": pass_index, "support": support, "center": [x, y, base_z + size[2] / 2],
                  "size": [float(size[0]), float(size[1]), float(size[2])], "yaw_deg": math.degrees(yaw),
                  "bottom_z": base_z, "top_z": float(corners[:, 2].max()), "footprint": fp.tolist(),
                  "model_path": model["path"], "license": "CC0", "page": model["page"]}
        placed.append(record)
        boxes.append(record)

    # Render settings and passes.
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = WIDTH, HEIGHT
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    layer = scene.view_layers[0]
    for attr in ("use_pass_z", "use_pass_normal", "use_pass_diffuse_color", "use_pass_diffuse_direct", "use_pass_diffuse_indirect",
                 "use_pass_glossy_color", "use_pass_glossy_direct", "use_pass_glossy_indirect", "use_pass_transmission_color",
                 "use_pass_transmission_direct", "use_pass_transmission_indirect", "use_pass_emit", "use_pass_object_index"):
        setattr(layer, attr, True)
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    render_layers = tree.nodes.new("CompositorNodeRLayers")
    output = tree.nodes.new("CompositorNodeOutputFile")
    output.format.file_format = "OPEN_EXR"
    output.format.color_depth = "32"
    output.file_slots.clear()
    passes = ["Image", "Depth", "Normal", "DiffCol", "DiffDir", "DiffInd", "GlossCol", "GlossDir", "GlossInd",
              "TransCol", "TransDir", "TransInd", "Emit", "IndexOB"]
    for name in passes:
        output.file_slots.new(f"{name}_")
        tree.links.new(render_layers.outputs[name], output.inputs[f"{name}_"])
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(render_layers.outputs["Image"], composite.inputs["Image"])

    camera_data = bpy.data.cameras.new("camera")
    camera = bpy.data.objects.new("camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    targets = np.array([p["center"] for p in placed]) if placed else np.array([[0, 0, 0.5]])
    for view in range(args.views):
        fov = math.radians(rng.uniform(50, 68))
        camera_data.sensor_fit = "HORIZONTAL"
        camera_data.lens = camera_data.sensor_width / (2 * math.tan(fov / 2))
        eye = Vector((rng.uniform(-W / 2 + 0.6, W / 2 - 0.6), -D / 2 + rng.uniform(0.3, 0.9), rng.uniform(1.15, 1.7)))
        target = Vector(targets.mean(0)) + Vector((rng.uniform(-0.4, 0.4), rng.uniform(-0.2, 0.4), rng.uniform(-0.2, 0.2)))
        forward = (target - eye).normalized()
        roll = math.radians(rng.uniform(-2.5, 2.5))
        camera.matrix_world = Matrix.Translation(eye) @ (forward.to_track_quat("-Z", "Y").to_matrix().to_4x4()) @ Matrix.Rotation(roll, 4, "Z")
        bpy.context.view_layer.update()

        view_dir = os.path.join(args.out, f"view-{view}")
        os.makedirs(view_dir, exist_ok=True)
        output.base_path = view_dir
        scene.frame_set(1)
        bpy.ops.render.render(write_still=False)

        frame = lambda name: os.path.join(view_dir, f"{name}_0001.exr")
        exr = {name: read_exr(frame(name), 1 if name in ("Depth", "IndexOB") else 3) for name in passes}
        for name in passes:
            os.remove(frame(name))
        # Exposure in post: median luminance of non-emitting pixels at photographic middle gray (0.18).
        luma = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        emitting = (exr["Emit"] @ luma) > 1e-4
        exposure = float(0.18 / max(np.median((exr["Image"] @ luma)[~emitting]), 1e-6))
        write_png(os.path.join(view_dir, "rgb.png"), np.round(linear_to_srgb(exr["Image"] * exposure) * 255).astype(np.uint8))
        diffuse = (exr["DiffDir"] + exr["DiffInd"]) * exposure
        residual = (exr["GlossCol"] * (exr["GlossDir"] + exr["GlossInd"]) + exr["TransCol"] * (exr["TransDir"] + exr["TransInd"]) + exr["Emit"]) * exposure
        np.savez_compressed(os.path.join(view_dir, "gt.npz"), depth=exr["Depth"][..., 0], normal_world=exr["Normal"],
                            albedo=exr["DiffCol"], shading=diffuse, residual=residual, emission=exr["Emit"],
                            instance=np.rint(exr["IndexOB"][..., 0]).astype(np.int16))

        matrix = camera.matrix_world
        fx = WIDTH / (2 * math.tan(fov / 2))
        cam_forward = -(matrix.to_3x3() @ Vector((0, 0, 1)))
        cam_right = matrix.to_3x3() @ Vector((1, 0, 0))
        cam_up = matrix.to_3x3() @ Vector((0, 1, 0))
        gt = {
            "schema_version": 1,
            "frame": "blender world: meters, +Z up",
            "image": "rgb.png",
            "image_size": [WIDTH, HEIGHT],
            "depth": "planar depth along the camera axis (meters), Cycles Z pass",
            "decomposition": "linear rgb = albedo * shading + residual (residual = glossy + transmission + emission)",
            "exposure": exposure,
            "clipped_fraction": float((exr["Image"] * exposure >= 0.98).all(-1).mean()),
            "camera": {
                "position": list(matrix.translation),
                "matrix_world": [list(row) for row in matrix],
                "intrinsics_px": {"fx": fx, "fy": fx, "cx": WIDTH / 2, "cy": HEIGHT / 2},
                "fov_x_deg": math.degrees(fov),
                "height_m": matrix.translation.z,
                "pitch_deg": math.degrees(math.asin(max(-1, min(1, cam_forward.z)))),
                "roll_deg": math.degrees(math.atan2(cam_right.z, cam_up.z)),
            },
            "room": {"size": [W, D, H]},
            "structure": [{k: v for k, v in s.items() if k != "obj"} for s in structure],
            "objects": [{k: v for k, v in p.items() if k != "footprint"} for p in placed],
            "windows": windows,
            "lights": lights,
            "seed": args.seed,
            "view": view,
        }
        with open(os.path.join(view_dir, "gt.json"), "w") as f:
            json.dump(gt, f, indent=2)
        print(json.dumps({"view": view_dir, "objects": len(placed), "windows": len(windows)}), flush=True)


if __name__ == "__main__":
    main()
