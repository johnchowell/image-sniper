#!/usr/bin/env python3
"""Chooses the light-source rule by the task metric: true windows located in 3D after the layout (rectangle IoU
on the true window plane >= 0.3, scale-aligned), on the train split; check val, then test once.

For each candidate rule: re-derive every view's light estimate from its cached prediction (no inference), rebuild
its layout (same index), and score with evaluate.py. Candidates are compared on the split given (use train to
choose; run test once with the chosen rule).
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
import evaluate  # noqa: E402

CANDIDATES = {
    "v1-default": {"dominance": 0.4, "min_luminance": 0.3, "source_clipped": 0.7, "clipped_candidates": 1, "source_relative_luminance": 0},
    "v1-residual-only": {"dominance": 0.6, "min_luminance": 0.5, "source_clipped": 0.5, "clipped_candidates": 0, "source_relative_luminance": 0},
    "relative-8": {"dominance": 0.4, "min_luminance": 0.3, "source_clipped": 0.7, "clipped_candidates": 1, "source_relative_luminance": 8},
    "relative-5": {"dominance": 0.4, "min_luminance": 0.3, "source_clipped": 0.7, "clipped_candidates": 1, "source_relative_luminance": 5},
    "relative-3": {"dominance": 0.4, "min_luminance": 0.3, "source_clipped": 0.7, "clipped_candidates": 1, "source_relative_luminance": 3},
}


def apply(split, params, python):
    for view_dir in evaluate.views(split):
        world = evaluate.world_name(view_dir)
        light = os.path.join("worlds", world, "output", "light")
        meta = json.load(open(os.path.join(REPO, light, "0-light.json")))
        subprocess.run([python, ".claude/scripts/light/estimate_light.py", "--image", meta["source_image"], "--out-dir", light,
                        "--index", "0", "--postprocess-only", "--dominance", str(params["dominance"]),
                        "--min-luminance", str(params["min_luminance"]), "--source-clipped", str(params["source_clipped"]),
                        "--clipped-candidates", str(params["clipped_candidates"]),
                        "--source-relative-luminance", str(params["source_relative_luminance"])], cwd=REPO, check=True, capture_output=True)
        subprocess.run(["node", ".claude/scripts/layout/build-layout.mjs", "--world", world, "--index", "0"], cwd=REPO, check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--candidates", nargs="*", default=list(CANDIDATES))
    args = parser.parse_args()
    rows = {}
    for name in args.candidates:
        apply(args.split, CANDIDATES[name], sys.executable)
        summary = evaluate.summarize([evaluate.score_view(v) for v in evaluate.views(args.split)])
        rows[name] = {"params": CANDIDATES[name], **summary["light"]}
        print(name, json.dumps(rows[name]), flush=True)
    json.dump(rows, open(os.path.join(REPO, "benchmark", "results", f"light-select-{args.split}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
