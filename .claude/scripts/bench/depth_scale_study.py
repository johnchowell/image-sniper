#!/usr/bin/env python3
"""Metric-scale study: how far each open metric-depth model's absolute scale is from ground truth.

For each view: scale = median(true depth / predicted depth) over valid, non-emitting pixels (1.0 = perfect).
Models: the pipeline's MoGe-2 depth (from the latest layout), plus extra Hugging Face depth models.
Also scores blends of MoGe-2's scale with each extra model (geometric mean of the two scale estimates).
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LUMA = np.array([0.2126, 0.7152, 0.0722])
MODELS = {
    "depth-anything-v2-metric-indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "depth-pro": "apple/DepthPro-hf",
}


def predict(name, model_id, image, cache):
    if name not in cache:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        cache[name] = (AutoImageProcessor.from_pretrained(model_id), AutoModelForDepthEstimation.from_pretrained(model_id).eval())
    processor, model = cache[name]
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
    post = processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])[0]
    return post["predicted_depth"].numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count())
    cache, rows = {}, []
    for view_dir in sorted(glob.glob(os.path.join(REPO, "benchmark", "renders", args.split, "room-*", "view-*"))):
        room, view = view_dir.split("/")[-2:]
        gt = np.load(os.path.join(view_dir, "gt.npz"))
        image = Image.open(os.path.join(view_dir, "rgb.png")).convert("RGB")
        valid = (gt["depth"] > 0) & (gt["depth"] < 50) & ((gt["emission"] @ LUMA) < 1e-4)
        row = {"view": f"{room}/{view}"}
        world = os.path.join(REPO, "worlds", f"bench-{args.split}-{room}-{view}", "output", "layout")
        layouts = sorted(glob.glob(os.path.join(world, "[0-9]*-layout-depth.png")), key=lambda p: int(os.path.basename(p).split("-")[0]))
        if layouts:
            moge = np.asarray(Image.open(layouts[0])).astype(np.float64) / 1000
            moge = np.asarray(Image.fromarray(moge.astype(np.float32)).resize(image.size, Image.NEAREST))
            ok = valid & (moge > 0)
            row["moge-2"] = float(np.median(gt["depth"][ok] / moge[ok]))
        for name in args.models:
            depth = predict(name, MODELS[name], image, cache)
            ok = valid & (depth > 0)
            row[name] = float(np.median(gt["depth"][ok] / depth[ok]))
        for name in args.models:
            if "moge-2" in row:
                row[f"moge-2+{name}"] = float(np.sqrt(row["moge-2"] * row[name]))
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)

    keys = [k for k in rows[0] if k != "view"]
    summary = {k: {"median_abs_log_err": round(float(np.median([abs(np.log(r[k])) for r in rows if k in r])), 4),
                   "median_scale": round(float(np.median([r[k] for r in rows if k in r])), 4)} for k in keys}
    os.makedirs(os.path.join(REPO, "benchmark", "results"), exist_ok=True)
    json.dump({"split": args.split, "summary": summary, "views": rows}, open(os.path.join(REPO, "benchmark", "results", f"depth-scale-{args.split}.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
