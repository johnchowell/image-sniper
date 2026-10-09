#!/usr/bin/env python3
"""Local inputs for image-blast-layout: MoGe-2 depth plus text-prompted instance masks
(Grounding DINO finds boxes from each confirmed object's name, SAM 2.1 cuts each box into a mask).

Writes the same files and request metadata as the fal path (fal-ai/moge-2 + fal-ai/sam-3/image) into
worlds/<world>/output/layout/ at the next index, so build-layout.mjs runs unchanged. Prints the index.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from common import next_index, world_path, write_request

RESERVED = {"world", "sfx", "light", "layout", "scene"}
DETECTOR = "IDEA-Research/grounding-dino-base"
SEGMENTER = "facebook/sam2.1-hiera-large"


def confirmed_objects(world):
    out = world_path(world, "output")
    objects = []
    for name in sorted(os.listdir(out)):
        path = os.path.join(out, name, "object.json")
        if name in RESERVED or not os.path.exists(path):
            continue
        obj = json.load(open(path)).get("object", {})
        objects.append({"id": obj.get("id", name), "name": obj.get("name", name), "count": int(obj.get("count_estimate", 1) or 1)})
    return objects


def original_source(world):
    src = world_path(world, "source")
    images = [f for f in os.listdir(src) if not f.startswith(".") and f.split("-")[0].isdigit()
              and f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    return os.path.join(src, min(images, key=lambda f: (int(f.split("-")[0]), f)))


def detect_jointly(processor, detector, image, objects, threshold, extra):
    """One Grounding DINO pass with every object name as a competing phrase.

    A box's score for a phrase is the max token probability over that phrase's tokens. Each box goes to its best
    phrase; overlapping boxes keep the higher score across all phrases (one image region, one object); each object
    keeps at most count_estimate + extra boxes. Independent per-name queries let similar names claim the same region.
    Threshold and extra box count were chosen on the benchmark training renders (bench/detection_study.py,
    results in benchmark/results/detection-*.json).
    """
    phrases = [obj["name"].lower().strip().rstrip(".") for obj in objects]
    prompt = " ".join(f"{p}." for p in phrases)
    W, H = image.size
    with torch.no_grad():
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        outputs = detector(**inputs)
    offsets = processor.tokenizer(prompt, return_offsets_mapping=True)["offset_mapping"]
    probs = outputs.logits[0].sigmoid().numpy()  # queries x text positions
    spans, cursor = [], 0
    for phrase in phrases:
        start = prompt.index(phrase, cursor)
        end = start + len(phrase)
        cursor = end
        spans.append([i for i, (a, b) in enumerate(offsets) if b > a and a >= start and b <= end and i < probs.shape[1]])
    class_scores = np.stack([probs[:, span].max(1) if span else np.zeros(len(probs)) for span in spans], 1)
    best = class_scores.argmax(1)
    score = class_scores.max(1)
    cx, cy, w, h = outputs.pred_boxes[0].numpy().T
    boxes = np.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], 1)
    keep = [i for i in nms(boxes, score, iou=0.6) if score[i] >= threshold]
    detections = {obj["id"]: [] for obj in objects}
    for i in keep:
        obj = objects[best[i]]
        if len(detections[obj["id"]]) < max(1, obj["count"]) + extra:
            detections[obj["id"]].append((boxes[i].clip([0, 0, 0, 0], [W, H, W, H]), float(score[i])))
    return prompt, detections


def nms(boxes, scores, iou=0.5):
    order = np.argsort(-scores)
    keep = []
    for i in order:
        if all(box_iou(boxes[i], boxes[j]) < iou for j in keep):
            keep.append(i)
    return keep


def box_iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / max(area(a) + area(b) - inter, 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--image")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--extra-per-object", type=int, default=1, help="boxes kept beyond count_estimate")
    args = parser.parse_args()

    source = args.image or original_source(args.world)
    image = Image.open(source).convert("RGB")
    rgb = np.asarray(image)
    H, W = rgb.shape[:2]
    out_rel = world_path(args.world, "output", "layout")
    index = next_index(out_rel)
    torch.set_num_threads(os.cpu_count())
    timings = {}

    # Depth: same checkpoint and defaults as the fal request.
    from moge.model.v2 import MoGeModel

    started = time.time()
    moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").eval()
    with torch.no_grad():
        result = moge.infer(torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1), resolution_level=9, apply_mask=True, use_fp16=False)
    del moge
    points = result["points"].numpy()
    mask = result["mask"].numpy().astype(bool) & np.isfinite(points).all(-1)
    K = result["intrinsics"].numpy().tolist()
    timings["depth"] = round(time.time() - started, 1)

    pts = points[mask] * np.array([1, -1, -1], dtype=np.float32)  # MoGe export convention (OpenGL axes)
    cols = rgb[mask]
    vertex = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = pts.T
    vertex["red"], vertex["green"], vertex["blue"] = cols.T
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n") % len(pts)
    files = {k: f"{out_rel}/{index}-layout-{s}" for k, s in [("point_cloud", "points.ply"), ("mask", "valid-mask.png"), ("depth_map", "depth-preview.png")]}
    with open(files["point_cloud"], "wb") as f:
        f.write(header.encode("ascii") + vertex.tobytes())
    Image.fromarray((mask * 255).astype(np.uint8)).save(files["mask"])
    depth = result["depth"].numpy()
    inv = np.where(mask, 1 / np.maximum(depth, 1e-6), 0)
    lo, hi = np.percentile(inv[mask], [1, 99])
    Image.fromarray((np.clip((inv - lo) / (hi - lo), 0, 1) * 255 * mask).astype(np.uint8)).save(files["depth_map"])
    write_request(os.path.join(out_rel, f".{index}-layout__depth-request.json"), {
        "kind": "layout-depth", "provider": "local/moge-2-vitl-normal", "endpoint": "local/moge-2-vitl-normal", "index": index,
        "status": "completed", "input": {"model": "vitl-normal", "resolution_level": 9, "apply_mask": True},
        "input_files": [source], "output_files": list(files.values()),
        "downloaded_files": [{"label": k, "path": p} for k, p in files.items()],
        "result": {"intrinsics": K, "fov_x": float(np.degrees(2 * np.arctan(0.5 / K[0][0])))},
    })

    # Masks: text -> boxes (Grounding DINO) -> masks (SAM 2.1).
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    started = time.time()
    processor = AutoProcessor.from_pretrained(DETECTOR)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR).eval()
    predictor = SAM2ImagePredictor.from_pretrained(SEGMENTER, device="cpu")
    predictor.set_image(rgb)
    found = {}
    objects = confirmed_objects(args.world)
    prompt, detections = detect_jointly(processor, detector, image, objects, args.box_threshold, args.extra_per_object) if objects else ("", {})
    for obj in objects:
        downloaded, kept_scores, kept_boxes = [], [], []
        for k, (box, box_score) in enumerate(detections[obj["id"]], start=1):
            with torch.no_grad():
                masks, _, _ = predictor.predict(box=box, multimask_output=False)
            path = f"{out_rel}/{index}-layout-mask-{obj['id']}-{k}.png"
            Image.fromarray((masks[0] > 0).astype(np.uint8) * 255).save(path)
            downloaded.append({"label": f"mask-{k}", "path": path})
            kept_scores.append(round(box_score, 4))
            kept_boxes.append([round(float(v), 1) for v in box])
        write_request(os.path.join(out_rel, f".{index}-layout__mask-{obj['id']}-request.json"), {
            "kind": "layout-mask", "provider": f"local/{DETECTOR.split('/')[-1]}+{SEGMENTER.split('/')[-1]}",
            "endpoint": "local/grounded-sam2", "index": index, "status": "completed",
            "object_id": obj["id"], "object_name": obj["name"], "prompt": prompt, "detection": "joint (all names compete)",
            "mask_threshold": args.box_threshold,
            "input_files": [source], "output_files": [d["path"] for d in downloaded], "downloaded_files": downloaded,
            "result": {"scores": kept_scores, "boxes_px": kept_boxes},
        })
        found[obj["id"]] = len(downloaded)
    timings["masks"] = round(time.time() - started, 1)
    print(json.dumps({"index": index, "source_image": source, "instances": found, "seconds": timings}))


if __name__ == "__main__":
    main()
