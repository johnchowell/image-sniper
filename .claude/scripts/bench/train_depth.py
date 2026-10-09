#!/usr/bin/env python3
"""Fine-tunes an open metric depth model on the benchmark renders (CPU: the backbone stays frozen and only the
DPT neck and head train), selects the epoch by val error, and scores it against the zero-shot model.

Model: depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf. Data: benchmark/dataset (export_dataset.py).
Loss: mean |log(pred) - log(true)| on pixels with true depth (sky and windows excluded): it trains metric scale,
the largest error the benchmark measures. Weights go to benchmark/models/ (not tracked); scores to
benchmark/results/depth-finetune.json.

  train_depth.py train           fine-tune on train, pick the epoch by val, score val
  train_depth.py score --split S score the zero-shot and fine-tuned models on one split
"""
import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DATA = os.path.join(REPO, "benchmark", "dataset")
OUT = os.path.join(REPO, "benchmark", "models", "da2-metric-indoor-small-bench")
RESULTS = os.path.join(REPO, "benchmark", "results", "depth-finetune.json")


def load_split(split):
    items = []
    for path in sorted(glob.glob(os.path.join(DATA, split, "images", "*.png"))):
        room = os.path.splitext(os.path.basename(path))[0]
        depth = np.asarray(Image.open(os.path.join(DATA, split, "depth", f"{room}.png"))).astype(np.float32) / 1000
        window = np.asarray(Image.open(os.path.join(DATA, split, "windows", f"{room}.png"))) > 127
        items.append({"room": room, "image": Image.open(path).convert("RGB"), "depth": depth, "valid": (depth > 0.1) & (depth < 20) & ~window})
    return items


def predict(model, processor, image):
    inputs = processor(images=image, return_tensors="pt")
    out = model(**inputs).predicted_depth  # (1, h, w) at the processing size
    return F.interpolate(out[:, None], size=(image.height, image.width), mode="bilinear", align_corners=False)[0, 0]


def score(model, processor, items):
    model.eval()
    rows = []
    with torch.no_grad():
        for item in items:
            pred = predict(model, processor, item["image"]).numpy()
            v = item["valid"] & (pred > 0)
            g, p = item["depth"][v], pred[v]
            rows.append({"room": item["room"], "absrel": float(np.mean(np.abs(p - g) / g)), "log_scale": float(np.log(np.median(g / p)))})
    return {"absrel_mean": round(float(np.mean([r["absrel"] for r in rows])), 4),
            "abs_log_scale_median": round(float(np.median([abs(r["log_scale"]) for r in rows])), 4),
            "log_scale_bias": round(float(np.mean([r["log_scale"] for r in rows])), 4), "views": rows}


def load(path=MODEL):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    return AutoImageProcessor.from_pretrained(MODEL), AutoModelForDepthEstimation.from_pretrained(path)


def train(args):
    torch.manual_seed(0)
    random.seed(0)
    processor, model = load()
    for p in model.backbone.parameters():
        p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    train_items, val_items = load_split("train"), load_split("val")
    history = [{"epoch": 0, "val": {k: v for k, v in score(model, processor, val_items).items() if k != "views"}}]
    print(json.dumps(history[-1]), flush=True)
    best = (history[0]["val"]["absrel_mean"], 0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.backbone.eval()
        random.shuffle(train_items)
        losses, started = [], time.time()
        for item in train_items:
            image, depth, valid = item["image"], item["depth"], item["valid"]
            if random.random() < 0.5:  # horizontal flip keeps intrinsics and metric depth
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                depth, valid = depth[:, ::-1].copy(), valid[:, ::-1].copy()
            pred = predict(model, processor, image)
            v = torch.from_numpy(valid) & (pred > 1e-3)
            loss = (torch.log(pred[v]) - torch.log(torch.from_numpy(depth)[v])).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        val = {k: v for k, v in score(model, processor, val_items).items() if k != "views"}
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4), "val": val, "seconds": round(time.time() - started, 1)})
        print(json.dumps(history[-1]), flush=True)
        if val["absrel_mean"] < best[0]:
            best = (val["absrel_mean"], epoch)
            model.save_pretrained(OUT)
    report = {"model": MODEL, "trained": "DPT neck and head; backbone frozen", "loss": "mean |log pred - log true|",
              "epochs": args.epochs, "lr": args.lr, "train_views": len(train_items), "val_views": len(val_items),
              "best_epoch": best[1], "history": history, "weights": os.path.relpath(OUT, REPO) if best[1] else None}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    json.dump(report, open(RESULTS, "w"), indent=2)


def score_split(split):
    items = load_split(split)
    processor, zero_shot = load()
    out = {"split": split, "zero_shot": score(zero_shot, processor, items)}
    if os.path.isdir(OUT):
        out["fine_tuned"] = score(load(OUT)[1], processor, items)
    report = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    report[f"score_{split}"] = {k: {m: v for m, v in s.items() if m != "views"} if isinstance(s, dict) else s for k, s in out.items()}
    json.dump(report, open(RESULTS, "w"), indent=2)
    print(json.dumps(report[f"score_{split}"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "score"])
    parser.add_argument("--split", default="val")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count())
    train(args) if args.command == "train" else score_split(args.split)


if __name__ == "__main__":
    main()
