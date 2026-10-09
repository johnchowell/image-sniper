#!/usr/bin/env python3
"""Full local IMAGE-BLAST pass with open-weight models, in pipeline order:

  light       Marigold-IID-Lighting   albedo, shading, light sources
  layout      MoGe-2 + Grounding DINO + SAM 2.1, then build-layout.mjs (structure, boxes, lighting)
  plate       big-lama                 removes every found object instance
  environment MoGe-2 on the plate      textured metric mesh of the empty room
  objects     TripoSR                  one mesh per object, albedo vertex colors, placement from the layout
  scene       assembly                 one GLB in the layout frame with camera and lights

Each step writes its own indexed files and request metadata. --skip reuses the latest output of a step.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from common import world_lock  # noqa: E402

STEPS = ["light", "layout", "plate", "environment", "objects", "scene"]


def run(command, label):
    started = time.time()
    env = {**os.environ, "PYTHONPATH": HERE + os.pathsep + os.environ.get("PYTHONPATH", "")}
    process = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True)
    if process.returncode != 0:
        tail = "\n".join((process.stderr or process.stdout).strip().splitlines()[-15:])
        raise SystemExit(f"{label} failed:\n{tail}")
    lines = [line for line in process.stdout.strip().splitlines() if line.strip()]
    print(f"[{label}] {time.time() - started:.0f}s", flush=True)
    return lines, round(time.time() - started, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--skip", action="append", default=[], choices=STEPS)
    args = parser.parse_args()
    with world_lock(args.world):
        run_pass(args)


def run_pass(args):
    py = sys.executable
    summary = {"world": args.world, "seconds": {}}

    if "light" not in args.skip:
        lines, summary["seconds"]["light"] = run(
            ["node", ".claude/scripts/light/generate-light.mjs", "--world", args.world, "--python", py, "--regenerate"], "light")
        light = json.loads("\n".join(lines))
        summary["light"] = {k: light.get(k) for k in ("index", "reconstruction_r2", "light_color", "nondiffuse_fraction")}

    if "layout" not in args.skip:
        lines, inputs_seconds = run([py, os.path.join(HERE, "layout_inputs.py"), "--world", args.world], "layout inputs")
        inputs = json.loads(lines[-1])
        lines, build_seconds = run(["node", ".claude/scripts/layout/build-layout.mjs", "--world", args.world, "--index", str(inputs["index"])], "layout build")
        layout = json.loads("\n".join(lines))
        summary["seconds"]["layout"] = round(inputs_seconds + build_seconds, 1)
        summary["layout"] = {"index": inputs["index"], "instances": inputs["instances"], "camera": layout["camera"],
                             "structure": layout["structure"], "warnings": layout["warnings"]}

    for step, script in [("plate", "make_plate.py"), ("environment", "environment_mesh.py"), ("objects", "object_mesh.py"), ("scene", "assemble_scene.py")]:
        if step in args.skip:
            continue
        lines, summary["seconds"][step] = run([py, os.path.join(HERE, script), "--world", args.world], step)
        summary[step] = [line for line in lines if line.startswith("{")]

    summary["seconds"]["total"] = round(sum(summary["seconds"].values()), 1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
