#!/usr/bin/env python3
"""Object meshes with TripoSR from the layout's instance masks.

For each object instance: the photo crop gives the geometry; when a light estimate exists, vertex colors are
queried from the albedo crop's triplane at the same vertices, so the mesh carries surface color without baked
lighting. The asset is written in a canonical frame (meters, +Y up, front +Z, base at y=0, centered on x/z), and
the request metadata records its placement in the layout frame (box height, front face on the box's front face,
front turned toward the source camera).

Writes worlds/<world>/output/<object>/N-<object>.glb, N-<object>.png (input crop), .N-<object>__model-request.json.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import trimesh
from PIL import Image

from common import REPO, latest, load_layout, load_mask, next_index, object_instances, world_path, write_request

sys.path.insert(0, os.path.join(REPO, "third_party", "TripoSR"))
from tsr.system import TSR  # noqa: E402
from tsr.utils import resize_foreground  # noqa: E402

# TripoSR frame: +Z up, input camera on +X. Canonical asset: +Y up, front +Z.
TSR_TO_CANONICAL = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)


def crop(image, mask, ratio):
    rgba = np.dstack([np.asarray(image.convert("RGB")), (mask * 255).astype(np.uint8)])
    framed = np.asarray(resize_foreground(Image.fromarray(rgba, "RGBA"), ratio)).astype(np.float32) / 255.0
    rgb = framed[..., :3] * framed[..., 3:4] + (1 - framed[..., 3:4]) * 0.5
    return Image.fromarray((rgb * 255).astype(np.uint8)).resize((512, 512), Image.BICUBIC)


def largest_parts(mesh):
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    biggest = max(len(p.faces) for p in parts)
    return trimesh.util.concatenate([p for p in parts if len(p.faces) >= 0.1 * biggest])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--object", action="append", help="object id (repeatable); default: every found object")
    parser.add_argument("--resolution", type=int, default=256, help="marching cubes resolution")
    parser.add_argument("--foreground-ratio", type=float, default=0.85)
    parser.add_argument("--faces", type=int, default=30000, help="target face count after decimation")
    args = parser.parse_args()

    layout_index, layout_path, layout = load_layout(args.world)
    photo = Image.open(layout["source_image"]).convert("RGB")
    size = photo.size
    light = latest(world_path(args.world, "output", "light"), "light", ".json")
    albedo = None
    if light:
        light_json = json.load(open(light[1]))
        if os.path.normpath(light_json["source_image"]) == os.path.normpath(layout["source_image"]):
            albedo = Image.open(light_json["files"]["albedo"]).convert("RGB").resize(size, Image.BICUBIC)

    torch.set_num_threads(os.cpu_count())
    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    camera = np.array(layout["camera"]["position"], dtype=np.float64)

    results = []
    seen = set()
    for obj, instance in object_instances(layout):
        if args.object and obj["object_id"] not in args.object:
            continue
        if obj["object_id"] in seen:  # one asset per object; every instance gets its own placement
            continue
        seen.add(obj["object_id"])
        started = time.time()
        mask = load_mask(instance["mask_file"], size)
        photo_crop = crop(photo, mask, args.foreground_ratio)
        with torch.no_grad():
            codes = model([photo_crop], device="cpu")
            mesh = largest_parts(model.extract_mesh(codes, False, resolution=args.resolution)[0])
            if len(mesh.faces) > args.faces:
                mesh = mesh.simplify_quadric_decimation(face_count=args.faces)
            # Colors are queried at the final vertices: from the albedo triplane when available.
            color_codes, color_source = codes[0], "photo"
            if albedo is not None:
                color_codes, color_source = model([crop(albedo, mask, args.foreground_ratio)], device="cpu")[0], "albedo"
            colors = model.renderer.query_triplane(model.decoder, torch.tensor(mesh.vertices, dtype=torch.float32), color_codes)["color"]
            mesh.visual.vertex_colors = np.c_[(colors.clamp(0, 1).numpy() * 255).astype(np.uint8), np.full(len(colors), 255, np.uint8)]

        # Canonical asset: metric height from the layout box, base at y=0, centered on x/z.
        vertices = mesh.vertices @ TSR_TO_CANONICAL.T
        low, high = vertices.min(0), vertices.max(0)
        scale = instance["size"][1] / max(high[1] - low[1], 1e-6)
        vertices = (vertices - [(low[0] + high[0]) / 2, low[1], (low[2] + high[2]) / 2]) * scale
        mesh.vertices = vertices
        extent = vertices.max(0) - vertices.min(0)

        # One placement per instance of this object: the asset scaled to the instance's measured height,
        # front (+Z) toward the source camera, front face on the box's front face, base on its support.
        placements = []
        for other_obj, other in object_instances(layout):
            if other_obj["object_id"] != obj["object_id"]:
                continue
            center = np.array(other["center"], dtype=np.float64)
            toward = camera - center
            toward[1] = 0
            toward /= max(np.linalg.norm(toward), 1e-9)
            yaw = math.atan2(toward[0], toward[2])
            box_yaw = math.radians(other["yaw_deg"])
            local_x = np.array([math.cos(box_yaw), 0, -math.sin(box_yaw)])
            local_z = np.array([math.sin(box_yaw), 0, math.cos(box_yaw)])
            # Scale from the measured height (snapped to the support), unless the frame cuts the object top or
            # bottom: then from the width seen by the camera, unless that is cut too.
            edges = set(other.get("truncated_edges", []))
            side = np.array([toward[2], 0, -toward[0]])
            seen_width = abs(local_x @ side) * other["size"][0] + abs(local_z @ side) * other["size"][2]
            if edges & {"top", "bottom"} and not edges & {"left", "right"}:
                factor, scale_basis = seen_width / max(extent[0], 1e-9), "visible_width (top or bottom cut by the frame)"
            else:
                factor = other["size"][1] / max(extent[1], 1e-9)
                scale_basis = "height" if not edges else "height (cut by the frame on both axes: lower bound)"
            box_reach = abs(local_x @ toward) * other["size"][0] / 2 + abs(local_z @ toward) * other["size"][2] / 2
            position = center + toward * (box_reach - factor * extent[2] / 2)
            position[1] = other["bottom_y_m"]
            placements.append({
                "instance_id": other["id"],
                "frame": "layout",
                "translation": [round(float(v), 4) for v in position],
                "yaw_deg": round(math.degrees(yaw), 2),
                "rotation_quaternion": [0, round(math.sin(yaw / 2), 6), 0, round(math.cos(yaw / 2), 6)],
                "scale": round(float(factor), 5),
                "scale_basis": scale_basis,
            })

        object_dir = world_path(args.world, "output", obj["object_id"])
        n = next_index(object_dir)
        glb = os.path.join(object_dir, f"{n}-{obj['object_id']}.glb")
        reference = os.path.join(object_dir, f"{n}-{obj['object_id']}.png")
        mesh.export(glb, include_normals=True)  # without normals, viewers shade the mesh black
        photo_crop.save(reference)
        summary = {
            "object_id": obj["object_id"],
            "model": glb,
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "size_m": [round(float(v), 3) for v in extent],
            "box_size_m": instance["size"],
            "color_source": color_source,
            "seconds": round(time.time() - started, 1),
        }
        write_request(os.path.join(object_dir, f".{n}-{obj['object_id']}__model-request.json"), {
            "kind": "3d",
            "provider": "local/triposr",
            "endpoint": "local/triposr",
            "index": n,
            "status": "completed",
            "input_files": [layout["source_image"], instance["mask_file"]] + ([light_json["files"]["albedo"]] if albedo is not None else []),
            "layout": layout_path,
            "asset_frame": "meters, +Y up, front +Z, base at y=0, centered on x/z",
            "placements": placements,
            "output_files": [glb, reference],
            "result": summary,
        })
        print(json.dumps(summary), flush=True)
        results.append(summary)


if __name__ == "__main__":
    main()
