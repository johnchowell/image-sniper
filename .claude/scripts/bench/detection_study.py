#!/usr/bin/env python3
"""Box-level detection study against ground truth: independent per-name queries vs joint (competing names),
over score thresholds and boxes kept per object. Scores precision, recall, F1 (a kept box is a true positive
when it overlaps its own object's true box at IoU >= 0.5). Grounding DINO runs once per view per strategy.
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
from layout_inputs import DETECTOR, box_iou, nms  # noqa: E402


def gt_objects(view_dir):
    gt = json.load(open(os.path.join(view_dir, "gt.json")))
    instance = np.load(os.path.join(view_dir, "gt.npz"))["instance"]
    objects = []
    for obj in gt["objects"]:
        mask = instance == obj["pass_index"]
        if mask.mean() < 0.002:
            continue
        ys, xs = np.nonzero(mask)
        objects.append({"name": obj["name"], "box": [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]})
    return objects


def raw_outputs(processor, detector, image, prompt):
    with torch.no_grad():
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        outputs = detector(**inputs)
    offsets = processor.tokenizer(prompt, return_offsets_mapping=True)["offset_mapping"]
    W, H = image.size
    cx, cy, w, h = outputs.pred_boxes[0].numpy().T
    boxes = np.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], 1)
    return outputs.logits[0].sigmoid().numpy(), offsets, boxes


def phrase_scores(probs, offsets, prompt, phrases):
    cursor, columns = 0, []
    for phrase in phrases:
        start = prompt.index(phrase, cursor)
        cursor = start + len(phrase)
        span = [i for i, (a, b) in enumerate(offsets) if b > a and a >= start and b <= cursor and i < probs.shape[1]]
        columns.append(probs[:, span].max(1) if span else np.zeros(len(probs)))
    return np.stack(columns, 1)


def evaluate(views, mode, threshold, keep_per_object):
    tp = fp = fn = 0
    for view in views:
        objects = view["objects"]
        kept = {k: [] for k in range(len(objects))}
        if mode == "joint":
            scores, boxes = view["joint"]
            best, score = scores.argmax(1), scores.max(1)
            for i in nms(boxes, score, iou=0.6):
                if score[i] >= threshold and len(kept[best[i]]) < keep_per_object:
                    kept[best[i]].append(boxes[i])
        else:
            for k, (score, boxes) in enumerate(view["independent"]):
                for i in nms(boxes, score, iou=0.5):
                    if score[i] >= threshold and len(kept[k]) < keep_per_object:
                        kept[k].append(boxes[i])
        for k, obj in enumerate(objects):
            hits = [box_iou(box, obj["box"]) >= 0.5 for box in kept[k]]
            tp += any(hits)
            fn += not any(hits)
            fp += len(hits) - (1 if any(hits) else 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    torch.set_num_threads(os.cpu_count())
    processor = AutoProcessor.from_pretrained(DETECTOR)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR).eval()
    views = []
    for view_dir in sorted(glob.glob(os.path.join(REPO, "benchmark", "renders", args.split, "room-*", "view-*"))):
        image = Image.open(os.path.join(view_dir, "rgb.png")).convert("RGB")
        objects = gt_objects(view_dir)
        phrases = [o["name"] for o in objects]
        prompt = " ".join(f"{p}." for p in phrases)
        probs, offsets, boxes = raw_outputs(processor, detector, image, prompt)
        joint = (phrase_scores(probs, offsets, prompt, phrases), boxes)
        independent = []
        for phrase in phrases:
            p, o, b = raw_outputs(processor, detector, image, f"{phrase}.")
            independent.append((phrase_scores(p, o, f"{phrase}.", [phrase])[:, 0], b))
        views.append({"objects": objects, "joint": joint, "independent": independent})
        print(view_dir.split("renders/")[1], len(objects), "objects", flush=True)

    results = []
    for mode in ("independent", "joint"):
        for threshold in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4):
            for keep in (1, 2):
                results.append({"mode": mode, "threshold": threshold, "keep_per_object": keep, **evaluate(views, mode, threshold, keep)})
    results.sort(key=lambda r: -r["f1"])
    json.dump({"split": args.split, "results": results}, open(os.path.join(REPO, "benchmark", "results", f"detection-{args.split}.json"), "w"), indent=2)
    for r in results[:8]:
        print(r)
    for mode in ("independent", "joint"):
        print("current" if mode == "joint" else "previous", next(r for r in results if r["mode"] == mode and r["threshold"] == 0.3 and r["keep_per_object"] == (1 if mode == "joint" else 2)))


if __name__ == "__main__":
    main()
