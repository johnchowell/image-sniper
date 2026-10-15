#!/usr/bin/env python3
"""Box-level detection study against ground truth for the names each bench world lists (oracle or noisy names):
greedy vs global (one-to-one) box assignment, merged clusters, crop-classifier weight, score thresholds and extra
boxes per object. Grounding DINO and the crop classifier run once per world; the strategies are scored offline. A kept box is a true positive when it
overlaps its own object's true box at IoU >= 0.5. Writes benchmark/results/detection-<split>.json.
"""
import argparse
import glob
import hashlib
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
from layout_inputs import DETECTOR, VERIFIER, assign_boxes, box_clusters, box_iou, detector_outputs, name_logprobs  # noqa: E402

LOWEST = 0.08  # crops are classified for every box kept at this threshold; kept boxes at a higher one are a subset


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


def score(views, mode, threshold, extra, merge, weight):
    tp = fp = fn = 0
    per_view = []
    for view in views:
        keep, scores = box_clusters(view["scores"], view["boxes"], threshold, merge)
        logprob = view["logprob"][[view["row"][i] for i in keep]] if weight else None
        kept = assign_boxes(scores, [o["count"] for o in view["objects"]], threshold, extra, mode, logprob, weight)
        v = [0, 0, 0]
        for k, obj in enumerate(view["objects"]):
            hits = [box_iou(view["boxes"][keep[c]], obj["box"]) >= 0.5 for c in kept[k]]
            v[0] += any(hits)
            v[2] += not any(hits)
            v[1] += len(hits) - (1 if any(hits) else 0)
        per_view.append(v)
    return np.array(per_view)


def f1(counts):
    tp, fp, fn = counts.sum(0)
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return precision, recall, 2 * precision * recall / max(precision + recall, 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--names", nargs="+", default=["oracle", "noisy"])
    args = parser.parse_args()
    from transformers import AutoModel, AutoModelForZeroShotObjectDetection, AutoProcessor

    torch.set_num_threads(os.cpu_count())
    processor = detector = verifier = None
    report = {"split": args.split, "verifier": VERIFIER, "results": {}}
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
            phrases = [o["name"] for o in objects]
            # model outputs are cached by image and phrases (benchmark/detections/, untracked) so strategies rerun offline
            key = f"{world}-{hashlib.sha1(json.dumps(phrases).encode()).hexdigest()[:10]}"
            cache = os.path.join(REPO, "benchmark", "detections", f"{key}.npz")
            image = Image.open(os.path.join(view_dir, "rgb.png")).convert("RGB")
            if os.path.exists(cache) and json.loads(str(np.load(cache)["phrases"])) == phrases:
                scores, boxes = np.load(cache)["scores"], np.load(cache)["boxes"]
            else:
                if detector is None:
                    processor = AutoProcessor.from_pretrained(DETECTOR)
                    detector = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR).eval()
                _, scores, boxes = detector_outputs(processor, detector, image, phrases)
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                np.savez(cache, scores=scores, boxes=boxes, phrases=json.dumps(phrases))
            keep, _ = box_clusters(scores, boxes, LOWEST)
            crops = cache.replace(".npz", "-siglip.npz")
            if os.path.exists(crops) and np.load(crops)["keep"].tolist() == [int(i) for i in keep]:
                logprob = np.load(crops)["logprob"]
            else:
                if verifier is None:
                    verifier = AutoProcessor.from_pretrained(VERIFIER), AutoModel.from_pretrained(VERIFIER).eval()
                logprob = name_logprobs(verifier, image, boxes[keep], phrases) if keep else np.zeros((0, len(phrases)))
                np.savez(crops, keep=np.array(keep, dtype=int), logprob=logprob)
            views.append({"objects": objects, "scores": scores, "boxes": boxes, "logprob": logprob,
                          "row": {int(i): r for r, i in enumerate(keep)}})
        before = score(views, "greedy", 0.25, 0, False, 0)
        rng = np.random.default_rng(0)
        resample = rng.integers(0, len(views), (2000, len(views)))
        rows = []
        for merge in (False, True):
            for mode in ("greedy", "global"):
                for threshold in (0.1, 0.125, 0.15, 0.2, 0.25, 0.3):
                    for extra in (0, 1):
                        for weight in (0, 0.5, 1, 2):
                            counts = score(views, mode, threshold, extra, merge, weight)
                            precision, recall, f = f1(counts)
                            delta = np.array([f1(counts[i])[2] - f1(before[i])[2] for i in resample])
                            rows.append({"merge": merge, "mode": mode, "threshold": threshold, "extra": extra, "weight": weight,
                                         "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f, 4),
                                         "df1_vs_before": round(f - f1(before)[2], 4),
                                         "df1_ci95": [round(np.percentile(delta, 2.5), 4), round(np.percentile(delta, 97.5), 4)]})
        rows.sort(key=lambda r: -r["f1"])
        report["results"][names] = {"views": len(views), "objects": sum(len(v["objects"]) for v in views),
                                    "before": {"merge": False, "mode": "greedy", "threshold": 0.25, "extra": 0, "weight": 0}, "rows": rows}
        for r in rows[:6]:
            print(names, r, flush=True)
    json.dump(report, open(os.path.join(REPO, "benchmark", "results", f"detection-{args.split}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
