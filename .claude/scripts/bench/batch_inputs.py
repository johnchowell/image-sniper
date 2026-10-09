#!/usr/bin/env python3
"""Runs the light and layout stages on every bench world of a split with each model loaded once.

Stage by stage (one model in memory at a time): Marigold light estimate, MoGe-2 depth, Grounding DINO + SAM 2.1
masks, then build-layout.mjs. Writes the same indexed files and request metadata as generate-light.mjs and
layout_inputs.py (index 0 in a fresh world), and skips a world whose stage output exists, so a run resumes.
Noisy-name worlds (evaluate.py prepare --names noisy) copy light and depth from their oracle twin (same photo)
and differ only in masks.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, ".claude", "scripts", "local"))
sys.path.insert(0, os.path.join(REPO, ".claude", "scripts", "light"))
sys.path.insert(0, HERE)
import estimate_light  # noqa: E402
import evaluate  # noqa: E402
import layout_inputs  # noqa: E402
from common import world_lock, write_request  # noqa: E402


def log(message):
    print(message, flush=True)


def worlds_of(split, names):
    out = []
    for view_dir in evaluate.views(split):
        name = evaluate.world_name(view_dir, names)
        if os.path.isdir(os.path.join(REPO, "worlds", name)):
            out.append((name, evaluate.world_name(view_dir, "oracle")))
    return out


def light_stage(worlds, timings):
    todo = [(w, twin) for w, twin in worlds if not os.path.exists(os.path.join(REPO, "worlds", w, "output", "light", "0-light.json"))]
    pipe = None
    for world, twin in todo:
        out_dir = f"worlds/{world}/output/light"
        if world != twin and os.path.exists(os.path.join(REPO, "worlds", twin, "output", "light", "0-light.json")):
            shutil.copytree(os.path.join(REPO, "worlds", twin, "output", "light"), os.path.join(REPO, out_dir), dirs_exist_ok=True)
            continue  # same photo: the twin's estimate is this world's estimate
        if pipe is None:
            started = time.time()
            pipe = estimate_light.load_model()
            timings["light_model_load"] = round(time.time() - started, 1)
        os.makedirs(os.path.join(REPO, out_dir), exist_ok=True)
        source = f"worlds/{world}/source/0-render.png"
        image = Image.open(os.path.join(REPO, source)).convert("RGB")
        started = time.time()
        prediction = estimate_light.predict(pipe, image)
        seconds = round(time.time() - started, 1)
        np.savez_compressed(os.path.join(REPO, out_dir, ".0-light-prediction.npz"), **prediction)
        settings = {"steps": 4, "ensemble": 1, "processing_resolution": 768, "seed": 0, **estimate_light.DEFAULT_RULE}
        summary = estimate_light.write_estimate(source, image, prediction, out_dir, 0, settings, {"inference": seconds})
        write_request(os.path.join(REPO, out_dir, ".0-light-request.json"), {
            "schema_version": 1, "kind": "light", "provider": f"local/{estimate_light.MODEL_ID}", "endpoint": f"local/{estimate_light.MODEL_ID}",
            "index": 0, "input_files": [source], "input": {k: settings[k] for k in ("steps", "ensemble", "processing_resolution")},
            "status": "completed", "output_files": [f"{out_dir}/0-light.json", *summary["files"].values()],
            "result": {"reconstruction_r2": summary["reconstruction_r2"], "scales": summary["scales"], "emitters": len(summary["emitters"])}})
        timings.setdefault("light_inference", []).append(seconds)
        log(f"light {world} {seconds}s")


def depth_stage(worlds, timings):
    todo = [(w, twin) for w, twin in worlds if not os.path.exists(os.path.join(REPO, "worlds", w, "output", "layout", "0-layout-points.ply"))]
    moge = None
    for world, twin in todo:
        out_rel = f"worlds/{world}/output/layout"
        os.makedirs(os.path.join(REPO, out_rel), exist_ok=True)
        twin_dir = os.path.join(REPO, "worlds", twin, "output", "layout")
        if world != twin and os.path.exists(os.path.join(twin_dir, "0-layout-points.ply")):
            for name in ("0-layout-points.ply", "0-layout-valid-mask.png", "0-layout-depth-preview.png", ".0-layout__depth-request.json"):
                text = open(os.path.join(twin_dir, name), "rb").read()
                if name.endswith(".json"):
                    text = text.replace(twin.encode(), world.encode())
                open(os.path.join(REPO, out_rel, name), "wb").write(text)
            continue
        if moge is None:
            started = time.time()
            moge = layout_inputs.load_depth_model()
            timings["depth_model_load"] = round(time.time() - started, 1)
        source = f"worlds/{world}/source/0-render.png"
        rgb = np.asarray(Image.open(os.path.join(REPO, source)).convert("RGB"))
        started = time.time()
        layout_inputs.depth_step(moge, rgb, out_rel, 0, source)
        timings.setdefault("depth_inference", []).append(round(time.time() - started, 1))
        log(f"depth {world} {timings['depth_inference'][-1]}s")


def mask_stage(worlds, timings):
    todo = [w for w, _ in worlds if not glob.glob(os.path.join(REPO, "worlds", w, "output", "layout", ".0-layout__mask-*-request.json"))]
    models = None
    for world in todo:
        objects = layout_inputs.confirmed_objects(world)
        if models is None:
            started = time.time()
            models = layout_inputs.load_mask_models()
            timings["mask_model_load"] = round(time.time() - started, 1)
        source = f"worlds/{world}/source/0-render.png"
        image = Image.open(os.path.join(REPO, source)).convert("RGB")
        started = time.time()
        layout_inputs.mask_step(models, image, np.asarray(image), objects, f"worlds/{world}/output/layout", 0, source, 0.25, 1)
        timings.setdefault("mask_inference", []).append(round(time.time() - started, 1))
        log(f"masks {world} {timings['mask_inference'][-1]}s ({len(objects)} names)")


def build_stage(worlds, rebuild):
    for world, _ in worlds:
        if not rebuild and os.path.exists(os.path.join(REPO, "worlds", world, "output", "layout", "0-layout.json")):
            continue
        process = subprocess.run(["node", ".claude/scripts/layout/build-layout.mjs", "--world", world, "--index", "0"], cwd=REPO, capture_output=True, text=True)
        log(f"build {world} " + ("ok" if process.returncode == 0 else "failed: " + (process.stderr or process.stdout).strip().splitlines()[-1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--names", choices=["oracle", "noisy"], default="oracle")
    parser.add_argument("--stages", nargs="+", default=["light", "depth", "masks", "build"])
    parser.add_argument("--rebuild", action="store_true", help="rebuild layouts that exist (after a build-layout change)")
    parser.add_argument("--limit", type=int, help="only the first N worlds (smoke test)")
    parser.add_argument("--threads", type=int, default=os.cpu_count(), help="torch threads (leave a core when another job runs)")
    args = parser.parse_args()
    os.chdir(REPO)  # the stage helpers write repo-relative paths (worlds/...)
    torch.set_num_threads(args.threads)
    worlds = worlds_of(args.split, args.names)[: args.limit]
    timings = {}
    locks = []
    try:
        for world, _ in worlds:  # hold every world of the batch: no second pass can write into them meanwhile
            lock = world_lock(world)
            lock.__enter__()
            locks.append(lock)
        if "light" in args.stages:
            light_stage(worlds, timings)
        if "depth" in args.stages:
            depth_stage(worlds, timings)
        if "masks" in args.stages:
            mask_stage(worlds, timings)
        if "build" in args.stages:
            build_stage(worlds, args.rebuild)
    finally:
        for lock in reversed(locks):
            lock.__exit__(None, None, None)
    summary = {k: (round(float(np.median(v)), 1) if isinstance(v, list) else v) for k, v in timings.items()}
    log(json.dumps({"split": args.split, "names": args.names, "worlds": len(worlds), "median_seconds": summary}))


if __name__ == "__main__":
    main()
