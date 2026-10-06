#!/usr/bin/env python3
"""Fits the light-source rule of estimate_light.py on the training renders, then scores it on the test renders.

Uses each bench world's cached model prediction (no inference). Score: pooled pixel precision / recall / F1 of
light-source pixels against the true emissive windows over all views of a split (views without windows count
their false detections).
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, ".claude", "scripts", "light"))
from estimate_light import LUMA, find_emitters, srgb_to_linear  # noqa: E402

GRID = {"dominance": [0.4, 0.5, 0.6, 0.7], "min_luminance": [0.3, 0.5, 0.7], "source_clipped": [0.3, 0.5, 0.7], "clipped_candidates": [0, 1]}
PREVIOUS = {"dominance": 0.6, "min_luminance": 0.5, "source_clipped": 0.5, "clipped_candidates": 0}


def load(split):
    views = []
    for d in sorted(glob.glob(os.path.join(REPO, "benchmark", "renders", split, "room-*", "view-*"))):
        room, view = d.split("/")[-2:]
        light = os.path.join(REPO, "worlds", f"bench-{split}-{room}-{view}", "output", "light")
        meta = json.load(open(os.path.join(light, "0-light.json")))
        cache = np.load(os.path.join(light, ".0-light-prediction.npz"))
        srgb = np.asarray(Image.open(os.path.join(d, "rgb.png")).convert("RGB"), dtype=np.float64) / 255
        photo = srgb_to_linear(srgb)
        albedo = cache["albedo"].astype(np.float64)
        shading = cache["shading"].astype(np.float64) * meta["scales"]["shading"]
        residual = cache["residual"].astype(np.float64) * meta["scales"]["residual"]
        residual_luma = residual @ LUMA
        dominance = residual_luma / np.maximum((albedo * shading) @ LUMA + residual_luma, 1e-9)
        windows = (np.load(os.path.join(d, "gt.npz"))["emission"] @ LUMA) > 1e-4
        views.append({"photo": photo, "luma": photo @ LUMA, "unclipped": (srgb < 0.98).all(-1), "dominance": dominance, "windows": windows})
    return views


def score(views, params):
    tp = fp = fn = 0
    for v in views:
        emitters, ids = find_emitters(v["photo"], v["luma"], v["unclipped"], v["dominance"], params["dominance"],
                                      params["min_luminance"], params["source_clipped"], params["clipped_candidates"])
        pred = np.isin(ids, [e["id"] for e in emitters if e["kind"] == "light_source"])
        tp += int((pred & v["windows"]).sum())
        fp += int((pred & ~v["windows"]).sum())
        fn += int((~pred & v["windows"]).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    train, test = load("train"), load("test")
    results = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        results.append({**params, **score(train, params)})
    results.sort(key=lambda r: -r["f1"])
    best = {k: results[0][k] for k in GRID}
    report = {
        "previous": {"params": PREVIOUS, "train": score(train, PREVIOUS), "test": score(test, PREVIOUS)},
        "best_on_train": {"params": best, "train": score(train, best), "test": score(test, best)},
        "top_train": results[:10],
    }
    json.dump(report, open(os.path.join(REPO, "benchmark", "results", "light-calibration.json"), "w"), indent=2)
    print(json.dumps({k: report[k] for k in ("previous", "best_on_train")}, indent=2))
    for r in results[:6]:
        print(r)


if __name__ == "__main__":
    main()
