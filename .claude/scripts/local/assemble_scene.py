#!/usr/bin/env python3
"""Assembles the full scene in the layout frame: environment mesh, placed object meshes, source camera, lights.

Objects use the placement recorded in each object's latest model request. Light sources from the layout's
lighting become KHR_lights_punctual spot lights shining into the room (intensities are relative), plus a
directional light when the dominant direction has more than low confidence.
Writes worlds/<world>/output/scene/N-scene.glb, N-scene.json, and .N-scene-request.json.
"""
import argparse
import json
import math
import os
import struct

import numpy as np
import trimesh

from common import latest, load_layout, next_index, world_path, write_request


def quaternion_from_basis(x, y, z):
    m = np.array([x, y, z], dtype=np.float64).T
    w = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    qx = math.copysign(math.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    qy = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    qz = math.copysign(math.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    q = np.array([qx, qy, qz, w])
    return (q / np.linalg.norm(q)).round(6).tolist()


def facing(direction):
    """Node rotation whose local -Z points along `direction` (glTF lights and cameras look down -Z)."""
    z = -np.asarray(direction, dtype=np.float64)
    z /= np.linalg.norm(z)
    x = np.cross([0, 1, 0], z)
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0, 0])
    return quaternion_from_basis(x, np.cross(z, x), z)


def add_extensions(glb, camera, lights):
    """Adds a camera node and KHR_lights_punctual light nodes to a GLB produced by trimesh."""
    json_length = struct.unpack_from("<I", glb, 12)[0]
    gltf = json.loads(glb[20:20 + json_length])
    rest = glb[20 + json_length:]
    nodes = gltf.setdefault("nodes", [])
    root = gltf["scenes"][gltf.get("scene", 0)]["nodes"]
    gltf.setdefault("cameras", []).append({"name": camera["name"], "type": "perspective", "perspective": camera["perspective"]})
    nodes.append({"name": camera["name"], "camera": len(gltf["cameras"]) - 1, "translation": camera["translation"], "rotation": camera["rotation"]})
    root.append(len(nodes) - 1)
    if lights:
        gltf.setdefault("extensionsUsed", []).append("KHR_lights_punctual")
        gltf.setdefault("extensions", {})["KHR_lights_punctual"] = {"lights": [light["light"] for light in lights]}
        for k, light in enumerate(lights):
            nodes.append({"name": light["light"]["name"], "translation": light["translation"], "rotation": light["rotation"],
                          "extensions": {"KHR_lights_punctual": {"light": k}}, "extras": light.get("extras", {})})
            root.append(len(nodes) - 1)
    text = json.dumps(gltf, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    body = struct.pack("<I", len(text)) + b"JSON" + text + rest
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    args = parser.parse_args()

    _, layout_path, layout = load_layout(args.world)
    scene_dir = world_path(args.world, "output", "scene")
    environment = latest(scene_dir, "scene-environment", ".glb")
    if not environment:
        raise SystemExit("No environment mesh. Run environment_mesh.py first.")

    scene = trimesh.Scene()
    for name, geometry in trimesh.load(environment[1], force="scene").geometry.items():
        scene.add_geometry(geometry, node_name="environment", geom_name=f"environment-{name}")

    placed = []
    for obj in layout["objects"]:
        object_dir = world_path(args.world, "output", obj["object_id"])
        model = latest(object_dir, obj["object_id"], ".glb")
        if not model:
            continue
        request = os.path.join(object_dir, f".{model[0]}-{obj['object_id']}__model-request.json")
        meta = json.load(open(request)) if os.path.exists(request) else {}
        placements = meta.get("placements") or ([meta["placement"]] if meta.get("placement") else [])
        if not placements:
            continue
        mesh = trimesh.load(model[1], force="mesh")
        for k, placement in enumerate(placements):
            transform = trimesh.transformations.rotation_matrix(math.radians(placement["yaw_deg"]), [0, 1, 0])
            transform[:3, :3] *= placement.get("scale", 1)
            transform[:3, 3] = placement["translation"]
            node = placement.get("instance_id", f"{obj['object_id']}-{k + 1}")
            scene.add_geometry(mesh, node_name=node, geom_name=obj["object_id"], transform=transform)
            placed.append({"object_id": obj["object_id"], "instance_id": node, "model": model[1], "placement": placement})

    cam = layout["camera"]
    camera = {
        "name": "source-camera",
        "perspective": {"yfov": math.radians(cam["fov_y_deg"]), "aspectRatio": cam["image_size"][0] / cam["image_size"][1], "znear": 0.01},
        "translation": cam["position"],
        "rotation": cam["rotation_quaternion"],
    }
    lights = []
    lighting = layout.get("lighting", {})
    if lighting.get("status") == "fitted":
        color = [min(1.0, c) for c in lighting["light_color"]["linear_rgb"]]
        for emitter in lighting["emitters"]:
            normal = np.array(emitter["normal"], dtype=np.float64)
            area = emitter["size"][0] * emitter["size"][1]
            lights.append({
                "light": {"name": emitter["id"], "type": "spot", "color": color, "intensity": round(10 * area, 3),
                          "spot": {"innerConeAngle": 0.6, "outerConeAngle": 1.4}},
                "translation": (np.array(emitter["center"]) + 0.05 * normal).round(4).tolist(),
                "rotation": facing(normal),
                "extras": {"class": "light_source", "size_m": emitter["size"], "intensity_units": "relative"},
            })
        dominant = lighting["dominant_light"]
        if dominant.get("confidence") != "low":
            lights.append({
                "light": {"name": "dominant-light", "type": "directional", "color": color, "intensity": 1},
                "translation": [0, 3, 0],
                "rotation": facing(-np.array(dominant["direction"])),
                "extras": {"class": "dominant_light", "confidence": dominant["confidence"]},
            })

    n = next_index(scene_dir)
    out = os.path.join(scene_dir, f"{n}-scene.glb")
    with open(out, "wb") as f:
        f.write(add_extensions(scene.export(file_type="glb"), camera, lights))
    manifest = {
        "schema_version": 1,
        "world": args.world,
        "frame": "layout (meters, right-handed, +Y up, source camera looks down -Z, origin on the floor below it)",
        "layout": layout_path,
        "environment": environment[1],
        "objects": placed,
        "camera": camera,
        "lights": [light["light"]["name"] for light in lights],
        "file": out,
    }
    with open(os.path.join(scene_dir, f"{n}-scene.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    write_request(os.path.join(scene_dir, f".{n}-scene-request.json"), {
        "kind": "scene", "provider": "local/assemble", "endpoint": "local/assemble", "index": n, "status": "completed",
        "input_files": [environment[1], layout_path] + [p["model"] for p in placed], "output_files": [out],
    })
    print(json.dumps({"scene": out, "instances": len(placed), "lights": manifest["lights"], "bytes": os.path.getsize(out)}))


if __name__ == "__main__":
    main()
