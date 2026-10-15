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

from common import exif_fov_x, next_index, world_path, write_request

RESERVED = {"world", "sfx", "light", "layout", "scene"}
DETECTOR = "IDEA-Research/grounding-dino-base"
SEGMENTER = "facebook/sam2.1-hiera-large"
VERIFIER = "google/siglip2-base-patch16-224"
# Chosen on the benchmark training renders by box F1 and confirmed on val (bench/detection_study.py,
# benchmark/results/detection-{train,val}.json): F1 0.756 -> 0.839 on train, 0.786 -> 0.867 on val.
BOX_THRESHOLD = 0.15
EXTRA_PER_OBJECT = 0
ASSIGNMENT = "global"
MERGE_CLUSTERS = True
VERIFY_WEIGHT = 1.0  # weight of the crop classifier's log p(name | crop); 0 does not load it


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


def detector_outputs(processor, detector, image, phrases):
    """One Grounding DINO pass with every phrase in the prompt: per-box phrase scores (max token probability over the
    phrase's tokens) and boxes in pixels."""
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
    cx, cy, w, h = outputs.pred_boxes[0].numpy().T
    boxes = np.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], 1)
    return prompt, class_scores, boxes.clip([0, 0, 0, 0], [W, H, W, H])


def box_clusters(class_scores, boxes, threshold, merge=False):
    """NMS on each box's best score (one image region, one object): kept box indices, in descending score order, and
    their per-name scores. merge: a kept box takes, per name, the best score of the boxes it suppresses, so a duplicate
    query on the same region that scores another name keeps that evidence."""
    score = class_scores.max(1)
    # A box below the threshold is never kept, cannot raise a kept box's score to the threshold, and cannot suppress
    # a kept box (NMS visits boxes by descending score), so it leaves NMS first; the result is the same.
    live = np.flatnonzero(score >= threshold)
    clusters = {live[i]: list(live[members]) for i, members in nms_clusters(boxes[live], score[live], iou=0.6).items()}
    keep = list(clusters)
    rows = [class_scores[clusters[i]].max(0) if merge else class_scores[i] for i in keep]
    return keep, np.array(rows).reshape(len(keep), class_scores.shape[1])


def assign_boxes(scores, counts, threshold, extra, mode="global", name_logprob=None, weight=0.0):
    """Kept boxes per object: object k may keep counts[k] boxes plus `extra`. scores: kept boxes x names (detector).
    The rank of a (box, name) pair is log score + weight * name_logprob (a crop classifier's log p(name | crop));
    a pair is accepted only if its detector score is at least the threshold.
    greedy: each box goes to its best-ranked name.
    global: one-to-one assignment maximizing the summed rank (Hungarian), so with similar names ("table",
            "coffee table", "side table") every object gets the box it ranks best among the boxes the others do
            not need more; leftover boxes then go to their best-ranked name as extras.
    Returns name index -> row indices of `scores`."""
    from scipy.optimize import linear_sum_assignment

    n, K = scores.shape
    kept = {k: [] for k in range(K)}
    if not n:
        return kept
    rank = np.log(np.maximum(scores, 1e-6))
    if name_logprob is not None and weight:
        rank = rank + weight * name_logprob
    best = rank.argmax(1)
    room = lambda k: len(kept[k]) < max(1, counts[k]) + extra
    used = set()
    if mode == "global":
        rows = [k for k in range(K) for _ in range(max(1, counts[k]))]
        for r, c in zip(*linear_sum_assignment(-rank[:, rows].T)):
            if scores[c, rows[r]] >= threshold:
                kept[rows[r]].append(c)
                used.add(c)
    for c in range(n):
        if c not in used and room(best[c]) and scores[c, best[c]] >= threshold:
            kept[best[c]].append(c)
    return kept


def name_logprobs(verifier, image, boxes, names):
    """log p(name | crop) for each box (padded 10%) over the names, from a SigLIP 2 image-text model."""
    processor, model = verifier
    W, H = image.size
    crops = []
    for x0, y0, x1, y1 in boxes:
        px, py = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
        crops.append(image.crop((max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py))))
    with torch.no_grad():
        inputs = processor(text=[f"a photo of a {name}." for name in names], images=crops, return_tensors="pt",
                           padding="max_length", max_length=64)
        logits = model(**inputs).logits_per_image.double()  # crops x names
    return torch.log_softmax(logits, 1).numpy()


def detect_jointly(models, image, objects, threshold, extra, mode=ASSIGNMENT, merge=MERGE_CLUSTERS, weight=VERIFY_WEIGHT):
    """Boxes for every confirmed object from one Grounding DINO pass in which all names compete, optionally checked
    by a crop classifier (see assign_boxes). The settings were chosen on the benchmark training renders
    (bench/detection_study.py, results in benchmark/results/detection-*.json)."""
    processor, detector, verifier = models[0], models[1], models[3]
    phrases = [obj["name"].lower().strip().rstrip(".") for obj in objects]
    prompt, class_scores, boxes = detector_outputs(processor, detector, image, phrases)
    keep, scores = box_clusters(class_scores, boxes, threshold, merge)
    logprob = name_logprobs(verifier, image, boxes[keep], phrases) if weight and keep else None
    kept = assign_boxes(scores, [obj["count"] for obj in objects], threshold, extra, mode, logprob, weight)
    detections = {obj["id"]: [(boxes[keep[c]], float(scores[c, k])) for c in kept[k]] for k, obj in enumerate(objects)}
    return prompt, detections


def nms(boxes, scores, iou=0.5):
    return list(nms_clusters(boxes, scores, iou))


def nms_clusters(boxes, scores, iou=0.5):
    """Greedy NMS: kept box -> the boxes it suppresses (itself included), in descending score order."""
    clusters = {}
    for i in np.argsort(-scores):
        owner = next((j for j in clusters if box_iou(boxes[i], boxes[j]) >= iou), None)
        if owner is None:
            clusters[i] = [i]
        else:
            clusters[owner].append(i)
    return clusters


def box_iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / max(area(a) + area(b) - inter, 1e-9)


def load_depth_model():
    from moge.model.v2 import MoGeModel

    return MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").eval()


def depth_step(moge, rgb, out_rel, index, source, fov_x=None, fov_basis=None):
    """MoGe-2 depth with the same checkpoint and defaults as the fal request; writes the point cloud, mask,
    preview and request metadata of this index. A known fov_x (deg) fixes the focal, so only the depth shift is solved."""
    with torch.no_grad():
        result = moge.infer(torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1), resolution_level=9, apply_mask=True, use_fp16=False,
                            **({"fov_x": float(fov_x)} if fov_x else {}))
    points = result["points"].numpy()
    mask = result["mask"].numpy().astype(bool) & np.isfinite(points).all(-1)
    K = result["intrinsics"].numpy().tolist()

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
        "status": "completed", "input": {"model": "vitl-normal", "resolution_level": 9, "apply_mask": True,
                                          **({"fov_x": float(fov_x), "fov_x_basis": fov_basis} if fov_x else {})},
        "input_files": [source], "output_files": list(files.values()),
        "downloaded_files": [{"label": k, "path": p} for k, p in files.items()],
        "result": {"intrinsics": K, "fov_x": float(np.degrees(2 * np.arctan(0.5 / K[0][0])))},
    })


def load_mask_models():
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(DETECTOR)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR).eval()
    predictor = SAM2ImagePredictor.from_pretrained(SEGMENTER, device="cpu")
    verifier = None
    if VERIFY_WEIGHT:
        from transformers import AutoModel
        verifier = AutoProcessor.from_pretrained(VERIFIER), AutoModel.from_pretrained(VERIFIER).eval()
    return processor, detector, predictor, verifier


def assignment_settings():
    return {"mode": ASSIGNMENT, "merge_clusters": MERGE_CLUSTERS, "verifier": VERIFIER if VERIFY_WEIGHT else None,
            "verify_weight": VERIFY_WEIGHT}


def mask_step(models, image, rgb, objects, out_rel, index, source, threshold, extra):
    """Text -> boxes (Grounding DINO, all names compete) -> masks (SAM 2.1); one request file per object."""
    predictor = models[2]
    predictor.set_image(rgb)
    found = {}
    prompt, detections = detect_jointly(models, image, objects, threshold, extra) if objects else ("", {})
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
            "assignment": assignment_settings(),
            "mask_threshold": threshold,
            "input_files": [source], "output_files": [d["path"] for d in downloaded], "downloaded_files": downloaded,
            "result": {"scores": kept_scores, "boxes_px": kept_boxes},
        })
        found[obj["id"]] = len(downloaded)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--image")
    parser.add_argument("--fov-x", type=float, help="horizontal field of view (deg); default: from the photo's EXIF focal length, else estimated")
    parser.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD)
    parser.add_argument("--extra-per-object", type=int, default=EXTRA_PER_OBJECT, help="boxes kept beyond count_estimate")
    args = parser.parse_args()

    source = args.image or original_source(args.world)
    image = Image.open(source).convert("RGB")
    rgb = np.asarray(image)
    out_rel = world_path(args.world, "output", "layout")
    index = next_index(out_rel)
    torch.set_num_threads(os.cpu_count())
    timings = {}

    started = time.time()
    moge = load_depth_model()
    fov = (args.fov_x, "--fov-x") if args.fov_x else exif_fov_x(source)
    depth_step(moge, rgb, out_rel, index, source, *(fov or ()))
    del moge
    timings["depth"] = round(time.time() - started, 1)

    started = time.time()
    found = mask_step(load_mask_models(), image, rgb, confirmed_objects(args.world), out_rel, index, source,
                      args.box_threshold, args.extra_per_object)
    timings["masks"] = round(time.time() - started, 1)
    print(json.dumps({"index": index, "source_image": source, "instances": found, "seconds": timings}))


if __name__ == "__main__":
    main()
