#!/usr/bin/env python3
"""Renders one procedural room with known ground truth (Blender Cycles, run with the bpy venv).

Room: textured floor and walls with baseboards, white ceiling, a door on some walls, 1-2 windows. A window is
either an opening in the wall with an outdoor HDRI behind it (daylight enters through a Cycles portal) or a
frosted emissive panel; both have a frame and sometimes mullions. Furniture stands on the floor without
overlaps; small props go on table tops. Exposure, tone curve (hard clip or a phone-like highlight roll-off)
and sensor noise vary per view.

Per view writes rgb.png and empty.png (the same view with the furniture hidden: plate truth), gt.npz (planar
depth, depth of the empty room, world normals, albedo, diffuse shading, non-diffuse light, emission, window
mask, instance ids), and gt.json (camera, room planes, objects with support, doors, windows, lights, exposure,
tone, noise). Blender world frame: Z up, meters.
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
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

SKIP = re.compile(r"wall|ceiling|chandelier|fan|hose|mounted|sconce", re.I)
SMALL_MAX = 0.55  # props up to this size go on table tops
WIDTH, HEIGHT = 640, 480
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
FAR = 1e3  # depth beyond this is the sky seen through an opening


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


def textured_material(name, maps, scale_m, tint=None, roughness=0.85):
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
    bsdf.inputs["Roughness"].default_value = roughness
    return material


def add_quads(name, quads, normal, material, pass_index=0):
    """One mesh object at the world origin (so Object texture coordinates are world coordinates and textures run
    on across pieces). Each quad is wound so its normal is `normal`."""
    verts, faces = [], []
    for quad in quads:
        q = [Vector(c) for c in quad]
        if (q[1] - q[0]).cross(q[3] - q[0]).dot(Vector(normal)) < 0:
            q = q[::-1]
        faces.append(tuple(range(len(verts), len(verts) + 4)))
        verts += [tuple(c) for c in q]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    obj.pass_index = pass_index
    return obj


def add_box(name, center, size, axes, material, pass_index=0):
    """Axis-aligned cube scaled to `size` along the columns of `axes` (3x3 world rotation)."""
    bpy.ops.mesh.primitive_cube_add(size=1)
    box = bpy.context.active_object
    box.name = name
    box.matrix_world = Matrix.Translation(Vector(center)) @ Matrix(axes).to_4x4() @ Matrix.Diagonal((*size, 1))
    box.data.materials.append(material)
    box.pass_index = pass_index
    return box


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


def develop(linear, exposure, tone, noise, rng):
    """Camera model: exposure, sensor noise (shot + read, in linear units), tone curve, sRGB, 8 bits.
    tone "clip" saturates highlights like a single exposure; "rolloff" (extended Reinhard on luminance with
    white point tone["white"]) keeps window detail like phone HDR."""
    x = linear * exposure
    sigma = np.sqrt(noise["shot"] * np.clip(x, 0, None) + noise["read"] ** 2)
    x = x + rng.standard_normal(x.shape).astype(np.float32) * sigma
    if tone["curve"] == "rolloff":
        lum = np.clip(x @ LUMA, 1e-6, None)[..., None]
        x = x * (1 + lum / tone["white"] ** 2) / (1 + lum)
    return np.round(linear_to_srgb(x) * 255).astype(np.uint8)


def read_exr(path, channels):
    image = bpy.data.images.load(path)
    w, h = image.size
    data = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(data)
    bpy.data.images.remove(image)
    return np.flipud(data.reshape(h, w, 4))[..., :channels]


def wall_frame(wall, H):
    """Wall basis: u along the wall (tangent), v up from the floor; to_world(u, v, w) with w toward the room."""
    n = Vector(wall["normal"])
    t = Vector((0, 0, 1)).cross(n).normalized()
    origin = Vector(wall["point"]) - Vector((0, 0, H / 2))
    return lambda u, v, w=0.0: origin + t * u + Vector((0, 0, v)) + n * w, t, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", default="benchmark/assets/manifest.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--views", type=int, default=1)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--objects", type=int, nargs=2, default=[5, 8], metavar=("MIN", "MAX"), help="furniture and props tried")
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 2], help="window counts to choose from")
    parser.add_argument("--no-empty", action="store_true", help="skip the furniture-free render (plate truth)")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    manifest = json.load(open(args.assets))
    reset()
    scene = bpy.context.scene

    # Room.
    W, D, H = rng.uniform(3.6, 6.0), rng.uniform(4.0, 7.0), rng.uniform(2.4, 3.1)
    floors = [t for t in manifest["textures"] if t["role"] == "floor"]
    walls = [t for t in manifest["textures"] if t["role"] == "wall"]
    # Textures map at their published real-world tile size (fallback 2 m), so scale cues match real rooms.
    floor_tex = rng.choice(floors) if floors else {}
    wall_tex = rng.choice(walls) if walls else {}
    floor_mat = textured_material("floor", floor_tex.get("maps"), floor_tex.get("tile_m") or 2.0, (0.4, 0.3, 0.2))
    wall_mat = textured_material("wall", wall_tex.get("maps"), wall_tex.get("tile_m") or 2.0, (0.8, 0.78, 0.74))
    ceiling_mat = textured_material("ceiling", None, 1, (0.85, 0.85, 0.85))
    trim_color = rng.choice([(0.85, 0.84, 0.8), (0.95, 0.95, 0.93), (0.35, 0.24, 0.16), (0.2, 0.2, 0.22)])
    trim_mat = textured_material("trim", None, 1, trim_color, roughness=0.5)
    structure = [
        {"id": "floor", "class": "floor", "point": [0, 0, 0], "normal": [0, 0, 1], "size": [W, D]},
        {"id": "ceiling", "class": "ceiling", "point": [0, 0, H], "normal": [0, 0, -1], "size": [W, D]},
    ]
    add_quads("floor", [[(-W / 2, -D / 2, 0), (W / 2, -D / 2, 0), (W / 2, D / 2, 0), (-W / 2, D / 2, 0)]], (0, 0, 1), floor_mat)
    add_quads("ceiling", [[(-W / 2, -D / 2, H), (W / 2, -D / 2, H), (W / 2, D / 2, H), (-W / 2, D / 2, H)]], (0, 0, -1), ceiling_mat)
    for wall_id, (point, normal, length) in {
        "wall-back": ((0, D / 2, H / 2), (0, -1, 0), W), "wall-front": ((0, -D / 2, H / 2), (0, 1, 0), W),
        "wall-left": ((-W / 2, 0, H / 2), (1, 0, 0), D), "wall-right": ((W / 2, 0, H / 2), (-1, 0, 0), D),
    }.items():
        structure.append({"id": wall_id, "class": "wall", "point": list(point), "normal": list(normal), "size": [length, H]})
    by_id = {s["id"]: s for s in structure}

    # Windows: at most one per wall, on the walls the camera faces. Openings show the HDRI; panels glow.
    hdris = manifest.get("hdris", [])
    windows, openings = [], {}
    window_walls = rng.sample(["wall-back", "wall-left", "wall-right"], rng.choice(args.windows))
    for wall_id in window_walls:
        wall = by_id[wall_id]
        length = wall["size"][0]
        width = rng.uniform(0.9, min(2.4, length - 1.2))
        height = rng.uniform(1.0, min(1.7, H - 1.15))
        sill = rng.uniform(0.75, min(1.0, H - height - 0.2))
        along = rng.uniform(-(length - width) / 2 + 0.4, (length - width) / 2 - 0.4)
        kind = "opening" if hdris and rng.random() < 0.6 else "panel"
        mullions = [m for m in ("vertical", "horizontal") if rng.random() < (0.5 if m == "vertical" else 0.3)]
        openings[wall_id] = {"u0": along - width / 2, "u1": along + width / 2, "v0": sill, "v1": sill + height, "kind": kind, "mullions": mullions}

    # Door: on a visible wall without a window (or beside the window when every wall has one).
    doors = []
    door_walls = [w for w in ("wall-back", "wall-left", "wall-right") if w not in openings] or list(openings)
    if rng.random() < 0.7:
        wall_id = rng.choice(door_walls)
        length = by_id[wall_id]["size"][0]
        dw, dh = rng.uniform(0.78, 0.92), rng.uniform(2.0, min(2.12, H - 0.15))
        lo, hi = -length / 2 + 0.25 + dw / 2, length / 2 - 0.25 - dw / 2
        if wall_id in openings:  # keep 0.3 m clear of the window
            o = openings[wall_id]
            spans = [(lo, o["u0"] - 0.3 - dw / 2), (o["u1"] + 0.3 + dw / 2, hi)]
            spans = [s for s in spans if s[1] > s[0]]
        else:
            spans = [(lo, hi)] if hi > lo else []
        if spans:
            a, b = rng.choice(spans)
            doors.append({"wall": wall_id, "u": rng.uniform(a, b), "width": dw, "height": dh})

    # Walls: one mesh each, cut around its opening; openings get a 0.15 m reveal (wall thickness).
    for wall_id in ("wall-back", "wall-front", "wall-left", "wall-right"):
        wall = by_id[wall_id]
        to_world, t, n = wall_frame(wall, H)
        L = wall["size"][0]
        rect = lambda u0, u1, v0, v1, w=0.0: [to_world(u0, v0, w), to_world(u1, v0, w), to_world(u1, v1, w), to_world(u0, v1, w)]
        o = openings.get(wall_id)
        if o and o["kind"] == "opening":
            quads = [rect(-L / 2, o["u0"], 0, H), rect(o["u1"], L / 2, 0, H), rect(o["u0"], o["u1"], 0, o["v0"]), rect(o["u0"], o["u1"], o["v1"], H)]
            add_quads(wall_id, quads, tuple(n), wall_mat)
            r = -0.15
            add_quads(f"{wall_id}-reveal-0", [[to_world(o["u0"], o["v0"]), to_world(o["u0"], o["v1"]), to_world(o["u0"], o["v1"], r), to_world(o["u0"], o["v0"], r)]], tuple(t), wall_mat)
            add_quads(f"{wall_id}-reveal-1", [[to_world(o["u1"], o["v0"]), to_world(o["u1"], o["v1"]), to_world(o["u1"], o["v1"], r), to_world(o["u1"], o["v0"], r)]], tuple(-t), wall_mat)
            add_quads(f"{wall_id}-sill", [[to_world(o["u0"], o["v0"]), to_world(o["u1"], o["v0"]), to_world(o["u1"], o["v0"], r), to_world(o["u0"], o["v0"], r)]], (0, 0, 1), wall_mat)
            add_quads(f"{wall_id}-head", [[to_world(o["u0"], o["v1"]), to_world(o["u1"], o["v1"]), to_world(o["u1"], o["v1"], r), to_world(o["u0"], o["v1"], r)]], (0, 0, -1), wall_mat)
        else:
            add_quads(wall_id, [rect(-L / 2, L / 2, 0, H)], tuple(n), wall_mat)
        # Baseboard along the full wall.
        bh = rng.uniform(0.07, 0.12)
        add_box(f"{wall_id}-baseboard", to_world(0, bh / 2, 0.008), (L, bh, 0.016), Matrix((t, Vector((0, 0, 1)), n)).transposed(), trim_mat)

    # Window frames, mullions, panels, portals.
    emission = bpy.data.materials.new("window-panel")
    emission.use_nodes = True
    nodes = emission.node_tree.nodes
    nodes.remove(nodes["Principled BSDF"])
    emit = nodes.new("ShaderNodeEmission")
    panel_color = (1.0, 0.97, 0.93) if rng.random() < 0.5 else (0.92, 0.96, 1.0)
    emit.inputs["Color"].default_value = (*panel_color, 1)
    emit.inputs["Strength"].default_value = rng.uniform(10, 25)
    emission.node_tree.links.new(emit.outputs["Emission"], nodes["Material Output"].inputs["Surface"])
    for wall_id, o in openings.items():
        wall = by_id[wall_id]
        to_world, t, n = wall_frame(wall, H)
        axes = Matrix((t, Vector((0, 0, 1)), n)).transposed()
        fw, fd = 0.05, 0.06
        depth_w = -0.07 if o["kind"] == "opening" else 0.012
        uc, vc = (o["u0"] + o["u1"]) / 2, (o["v0"] + o["v1"]) / 2
        width, height = o["u1"] - o["u0"], o["v1"] - o["v0"]
        bars = [(to_world(o["u0"] + fw / 2, vc, depth_w), (fw, height, fd)), (to_world(o["u1"] - fw / 2, vc, depth_w), (fw, height, fd)),
                (to_world(uc, o["v0"] + fw / 2, depth_w), (width, fw, fd)), (to_world(uc, o["v1"] - fw / 2, depth_w), (width, fw, fd))]
        if "vertical" in o["mullions"]:
            bars.append((to_world(uc, vc, depth_w), (0.04, height, fd * 0.8)))
        if "horizontal" in o["mullions"]:
            bars.append((to_world(uc, vc, depth_w), (width, 0.04, fd * 0.8)))
        for k, (center, size) in enumerate(bars):
            add_box(f"{wall_id}-frame-{k}", center, size, axes, trim_mat)
        center = to_world(uc, vc, -0.1 if o["kind"] == "opening" else 0.005)
        if o["kind"] == "panel":
            add_quads(f"window-{len(windows) + 1}", [[to_world(o["u0"], o["v0"], 0.005), to_world(o["u1"], o["v0"], 0.005),
                                                     to_world(o["u1"], o["v1"], 0.005), to_world(o["u0"], o["v1"], 0.005)]], tuple(n), emission)
        else:
            bpy.ops.object.light_add(type="AREA")
            portal = bpy.context.active_object
            portal.data.shape = "RECTANGLE"
            portal.data.size, portal.data.size_y = width, height
            # Area lights emit along local -Z: local -Z points into the room (basis (-t, up, -n) is right-handed).
            portal.matrix_world = Matrix.Translation(to_world(uc, vc, -0.14)) @ Matrix((-t, Vector((0, 0, 1)), -n)).transposed().to_4x4()
            try:
                portal.data.cycles.is_portal = True
            except AttributeError:
                portal.data.is_portal = True
        windows.append({"id": f"window-{len(windows) + 1}", "wall": wall_id, "kind": o["kind"], "center": list(to_world(uc, vc, 0)),
                        "normal": list(n), "size": [width, height], "sill_m": o["v0"], "mullions": o["mullions"],
                        "color_linear_rgb": list(panel_color) if o["kind"] == "panel" else None,
                        "strength": emit.inputs["Strength"].default_value if o["kind"] == "panel" else None})

    # World: the HDRI seen through openings (and the daylight that enters), else a dark world.
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    wn, wl = world.node_tree.nodes, world.node_tree.links
    hdri = None
    if any(o["kind"] == "opening" for o in openings.values()):
        hdri = rng.choice(hdris)
        env = wn.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(os.path.abspath(hdri["path"]))
        mapping, coords = wn.new("ShaderNodeMapping"), wn.new("ShaderNodeTexCoord")
        rotation = rng.uniform(0, 2 * math.pi)
        mapping.inputs["Rotation"].default_value = (0, 0, rotation)
        wl.new(coords.outputs["Generated"], mapping.inputs["Vector"])
        wl.new(mapping.outputs["Vector"], env.inputs["Vector"])
        wl.new(env.outputs["Color"], wn["Background"].inputs["Color"])
        strength = rng.uniform(1.0, 3.0)
        wn["Background"].inputs["Strength"].default_value = strength
        hdri = {"slug": hdri["slug"], "rotation_rad": rotation, "strength": strength}
    else:
        wn["Background"].inputs["Color"].default_value = (0.02, 0.02, 0.02, 1)
    scene.world = world
    lights = []
    if rng.random() < 0.5:
        bpy.ops.object.light_add(type="AREA", location=(rng.uniform(-W / 4, W / 4), rng.uniform(-D / 4, D / 4), H - 0.05))
        fill = bpy.context.active_object
        fill.data.size = 1.0
        fill.data.energy = rng.uniform(80, 200)
        fill.visible_camera = False
        lights.append({"id": "ceiling-fill", "type": "area", "center": list(fill.location), "energy": fill.data.energy})

    # Furniture on the floor, props on table tops.
    models = [m for m in manifest["models"] if not SKIP.search(m["slug"])]
    rng.shuffle(models)
    placed, furniture = [], []
    pass_index = 0
    for model in models[: rng.randint(*args.objects)]:
        root, meshes, size = import_model(model["path"])
        if max(size) > 2.6 or max(size) < 0.08:
            for obj in [root] + list(root.children_recursive):
                bpy.data.objects.remove(obj, do_unlink=True)
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
            same_level = [b for b in placed if b["support"] == support]
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
        record = {"id": f"{model['slug']}-{pass_index}", "slug": model["slug"], "name": clean_name(model["slug"]), "class": "furniture",
                  "pass_index": pass_index, "support": support, "center": [x, y, base_z + size[2] / 2],
                  "size": [float(size[0]), float(size[1]), float(size[2])], "yaw_deg": math.degrees(yaw),
                  "bottom_z": base_z, "top_z": float(corners[:, 2].max()), "footprint": fp.tolist(),
                  "model_path": model["path"], "license": "CC0", "page": model["page"]}
        placed.append(record)
        furniture += [root] + list(root.children_recursive)

    # Door leaf with casing and handle; a door is an object (box truth, instance mask) and a scale anchor.
    door_mat = textured_material("door", None, 1, rng.choice([(0.9, 0.9, 0.88), (0.45, 0.3, 0.18), (0.3, 0.35, 0.4)]), roughness=0.45)
    for door in doors:
        wall = by_id[door["wall"]]
        to_world, t, n = wall_frame(wall, H)
        axes = Matrix((t, Vector((0, 0, 1)), n)).transposed()
        pass_index += 1
        u, dw, dh = door["u"], door["width"], door["height"]
        add_box(f"door-{pass_index}", to_world(u, dh / 2, 0.02), (dw, dh, 0.04), axes, door_mat, pass_index)
        for k, (center, size) in enumerate([(to_world(u - dw / 2 - 0.035, (dh + 0.07) / 2, 0.01), (0.07, dh + 0.07, 0.02)),
                                            (to_world(u + dw / 2 + 0.035, (dh + 0.07) / 2, 0.01), (0.07, dh + 0.07, 0.02)),
                                            (to_world(u, dh + 0.035, 0.01), (dw + 0.14, 0.07, 0.02))]):
            add_box(f"door-{pass_index}-casing-{k}", center, size, axes, trim_mat, pass_index)
        side = rng.choice([-1, 1])
        add_box(f"door-{pass_index}-handle", to_world(u + side * (dw / 2 - 0.08), 1.0, 0.065), (0.12, 0.025, 0.05), axes, trim_mat, pass_index)
        yaw = math.atan2(t.y, t.x)
        placed.append({"id": f"door-{pass_index}", "slug": "door", "name": "door", "class": "door", "pass_index": pass_index,
                       "support": "floor", "center": list(to_world(u, dh / 2, 0.02)), "size": [dw + 0.14, 0.04, dh + 0.07],
                       "yaw_deg": math.degrees(yaw), "bottom_z": 0.0, "top_z": dh + 0.07, "wall": door["wall"],
                       "leaf_height_m": dh, "leaf_width_m": dw})

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
                 "use_pass_transmission_direct", "use_pass_transmission_indirect", "use_pass_emit", "use_pass_environment",
                 "use_pass_object_index"):
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
              "TransCol", "TransDir", "TransInd", "Emit", "Env", "IndexOB"]
    for name in passes:
        output.file_slots.new(f"{name}_")
        tree.links.new(render_layers.outputs[name], output.inputs[f"{name}_"])
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(render_layers.outputs["Image"], composite.inputs["Image"])

    camera_data = bpy.data.cameras.new("camera")
    camera = bpy.data.objects.new("camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    targets = np.array([p["center"] for p in placed if p["class"] == "furniture"]) if any(p["class"] == "furniture" for p in placed) else np.array([[0, 0, 0.5]])

    def render(view_dir, names):
        output.base_path = view_dir
        scene.frame_set(1)
        bpy.ops.render.render(write_still=False)
        frame = lambda name: os.path.join(view_dir, f"{name}_0001.exr")
        exr = {name: read_exr(frame(name), 1 if name in ("Depth", "IndexOB") else 3) for name in names}
        for name in passes:
            if os.path.exists(frame(name)):
                os.remove(frame(name))
        return exr

    for view in range(args.views):
        # Three views in four frame a window (its center inside the image): the camera aims between the
        # furniture and a window. The rest aim at the furniture alone.
        want_window = bool(windows) and rng.random() < 0.75
        for attempt in range(30):
            fov = math.radians(rng.uniform(50, 75))
            camera_data.sensor_fit = "HORIZONTAL"
            camera_data.lens = camera_data.sensor_width / (2 * math.tan(fov / 2))
            eye = Vector((rng.uniform(-W / 2 + 0.6, W / 2 - 0.6), -D / 2 + rng.uniform(0.3, 0.9), rng.uniform(1.15, 1.7)))
            target = Vector(targets.mean(0)) + Vector((rng.uniform(-0.4, 0.4), rng.uniform(-0.2, 0.4), rng.uniform(-0.2, 0.2)))
            if want_window:
                target = target.lerp(Vector(rng.choice(windows)["center"]), rng.uniform(0.3, 0.8))
            forward = (target - eye).normalized()
            roll = math.radians(rng.uniform(-2.5, 2.5))
            camera.matrix_world = Matrix.Translation(eye) @ (forward.to_track_quat("-Z", "Y").to_matrix().to_4x4()) @ Matrix.Rotation(roll, 4, "Z")
            bpy.context.view_layer.update()
            seen = [w for w in windows if (lambda p: p.z > 0 and 0.05 < p.x < 0.95 and 0.05 < p.y < 0.95)(world_to_camera_view(scene, camera, Vector(w["center"])))]
            if bool(seen) == want_window:
                break
        view_dir = os.path.join(args.out, f"view-{view}")
        os.makedirs(view_dir, exist_ok=True)

        for obj in furniture:
            obj.hide_render = False
        exr = render(view_dir, passes)
        depth = exr["Depth"][..., 0]
        sky = depth > FAR
        window = ((exr["Emit"] @ LUMA) > 1e-4) | sky
        # Exposure in post: median luminance of non-window pixels at a photographic target near middle gray.
        target_gray = rng.uniform(0.11, 0.22)
        exposure = float(target_gray / max(np.median((exr["Image"] @ LUMA)[~window]), 1e-6))
        if rng.random() < 0.5:
            tone = {"curve": "clip"}
        else:
            lum = (exr["Image"] * exposure) @ LUMA
            tone = {"curve": "rolloff", "white": float(max(np.percentile(lum, 99.9), 1.0) * rng.uniform(1.0, 1.5))}
        noise = {"shot": rng.uniform(1e-4, 1e-3), "read": rng.uniform(0.001, 0.004)}
        np_rng = np.random.default_rng(args.seed * 100 + view)
        write_png(os.path.join(view_dir, "rgb.png"), develop(exr["Image"], exposure, tone, noise, np_rng))
        diffuse = (exr["DiffDir"] + exr["DiffInd"]) * exposure
        residual = (exr["GlossCol"] * (exr["GlossDir"] + exr["GlossInd"]) + exr["TransCol"] * (exr["TransDir"] + exr["TransInd"])
                    + exr["Emit"] + exr["Env"]) * exposure
        arrays = {"depth": np.where(sky, np.inf, depth).astype(np.float32), "normal_world": exr["Normal"].astype(np.float16),
                  "albedo": exr["DiffCol"].astype(np.float16), "shading": diffuse.astype(np.float16), "residual": residual.astype(np.float16),
                  "emission": ((exr["Emit"] + exr["Env"]) * exposure).astype(np.float16), "window": window,
                  "instance": np.rint(exr["IndexOB"][..., 0]).astype(np.int16)}
        clipped = float((exr["Image"] * exposure >= 0.98).all(-1).mean()) if tone["curve"] == "clip" else 0.0

        if not args.no_empty:
            for obj in furniture:
                obj.hide_render = True
            empty = render(view_dir, ["Image", "Depth"])
            empty_depth = empty["Depth"][..., 0]
            write_png(os.path.join(view_dir, "empty.png"), develop(empty["Image"], exposure, tone, noise, np_rng))
            arrays["depth_empty"] = np.where(empty_depth > FAR, np.inf, empty_depth).astype(np.float32)
        np.savez_compressed(os.path.join(view_dir, "gt.npz"), **arrays)

        matrix = camera.matrix_world
        fx = WIDTH / (2 * math.tan(fov / 2))
        cam_forward = -(matrix.to_3x3() @ Vector((0, 0, 1)))
        cam_right = matrix.to_3x3() @ Vector((1, 0, 0))
        cam_up = matrix.to_3x3() @ Vector((0, 1, 0))
        gt = {
            "schema_version": 2,
            "frame": "blender world: meters, +Z up",
            "image": "rgb.png",
            "empty_image": None if args.no_empty else "empty.png",
            "image_size": [WIDTH, HEIGHT],
            "depth": "planar depth along the camera axis (meters), Cycles Z pass; inf where the sky shows through an opening",
            "decomposition": "exposed linear rgb = albedo * shading + residual (residual = glossy + transmission + emission + sky)",
            "window_mask": "emissive window panels and sky seen through openings",
            "exposure": exposure,
            "tone": tone,
            "noise": {**noise, "model": "linear x + N(0, shot * x + read^2), before the tone curve"},
            "clipped_fraction": clipped,
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
            "structure": structure,
            "objects": [{k: v for k, v in p.items() if k != "footprint"} for p in placed],
            "doors": [p["id"] for p in placed if p["class"] == "door"],
            "windows": windows,
            "hdri": hdri,
            "lights": lights,
            "seed": args.seed,
            "view": view,
            "framing": "window" if want_window else "furniture",
        }
        with open(os.path.join(view_dir, "gt.json"), "w") as f:
            json.dump(gt, f, indent=2)
        print(json.dumps({"view": view_dir, "objects": len(placed), "windows": [w["kind"] for w in windows], "doors": len(doors),
                          "tone": tone["curve"], "clipped": round(clipped, 4)}), flush=True)


if __name__ == "__main__":
    main()
