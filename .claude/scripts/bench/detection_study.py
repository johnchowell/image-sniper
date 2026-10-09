#!/usr/bin/env python3
"""Box-level detection study against ground truth for the names each bench world lists (oracle or noisy names):
greedy vs global (one-to-one) box assignment, over score thresholds and extra boxes per object. Grounding DINO
runs once per world; the strategies are scored offline on its outputs. A kept box is a true positive when it
overlaps its own object's true box at IoU >= 0.5. Writes benchmark/results/detection-<split>.json.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, ".claude", "scripts", "local"))
sys.path.insert(0, HERE)
import evaluate  # noqa: E402
from layout_inputs import DETECTOR, assign_boxes, box_iou, detector_outputs  # noqa: E402


def listed_objects(world, gt, instance):
    by_slug = {evaluate.slug(o["id"]): o for o in gt["objects"]}
    out = []
    for path in sorted(glob.glob(os.path.join(REPO, "worlds", world, "output", "*", "object.json"))):
        obj = json.load(open(path))["object"]
        truth = by_slug[obj["id"]]
        ys, xs = np.nonzero(instance == truth["pass_index"])
        out.append({"name": obj["name"].lower().strip().rstrip("."), "count": int(obj.get("count_estimate", 1) or 1),
                    "box": [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]})
    return out


def score(views, mode, threshold, extra):
    tp = fp = fn = 0
    for view in views:
        kept = assign_boxes(view["scores"], view["boxes"], [o["count"] for o in view["objects"]], threshold, extra, mode)
        for k, obj in enumerate(view["objects"]):
            hits = [box_iou(view["boxes"][i], obj["box"]) >= 0.5 for i in kept[k]]
            tp += any(hits)
            fn += not any(hits)
            fp += len(hits) - (1 if any(hits) else 0)
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--names", nargs="+", default=["oracle", "noisy"])
    args = parser.parse_args()
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    torch.set_num_threads(os.cpu_count())
    processor = AutoProcessor.from_pretrained(DETECTOR)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR).eval()
    report = {"split": args.split, "results": {}}
    for names in args.names:
        views = []
        for view_dir in evaluate.views(args.split):
            world = evaluate.world_name(view_dir, names)
            if not os.path.isdir(os.path.join(REPO, "worlds", world)):
                continue
            gt = json.load(open(os.path.join(view_dir, "gt.json")))
            objects = listed_objects(world, gt, np.load(os.path.join(view_dir, "gt.npz"))["instance"])
            if not objects:
                continue
            image = Image.open(os.path.join(view_dir, "rgb.png")).convert("RGB")
            _, scores, boxes = detector_outputs(processor, detector, image, [o["name"] for o in objects])
            views.append({"objects": objects, "scores": scores, "boxes": boxes})
        rows = []
        for mode in ("greedy", "global"):
            for threshold in (0.2, 0.25, 0.3, 0.35):
                for extra in (0, 1):
                    rows.append({"mode": mode, "threshold": threshold, "extra": extra, **score(views, mode, threshold, extra)})
        rows.sort(key=lambda r: -r["f1"])
        report["results"][names] = {"views": len(views), "objects": sum(len(v["objects"]) for v in views), "rows": rows}
        for r in rows[:4]:
            print(names, r, flush=True)
        print(names, "current", next(r for r in rows if r["mode"] == "greedy" and r["threshold"] == 0.25 and r["extra"] == 1), flush=True)
    json.dump(report, open(os.path.join(REPO, "benchmark", "results", f"detection-{args.split}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
