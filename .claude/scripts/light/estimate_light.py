#!/usr/bin/env python3
"""Intrinsic light decomposition with Marigold-IID-Lighting v1.1.

The model splits a photo into linear albedo A, diffuse shading S, and a non-diffuse residual R
(I = A * S + R). Shading and residual are predicted up to scale, so this script fits the two
scales against the linearized photo and stores everything in image units.

Writes into --out-dir with generation index N:
  N-light-albedo.png             8-bit sRGB albedo (lighting removed)
  N-light-shading.png            16-bit RGB linear diffuse shading (x encoding scale in JSON)
  N-light-residual.png           16-bit RGB linear non-diffuse light (x encoding scale in JSON)
  N-light-shading-preview.png    8-bit tone-mapped shading
  N-light-emitters.png           8-bit bright non-diffuse region ids (0 = none), kind in JSON
  N-light.json                   scales, reconstruction check, light color, emitter regions
"""
import argparse
import json
import struct
import time
import zlib

import numpy as np
from PIL import Image

MODEL_ID = "prs-eth/marigold-iid-lighting-v1-1"
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def write_png16_rgb(path, rgb):
    """16-bit RGB PNG (PIL cannot write 16-bit RGB)."""
    height, width, _ = rgb.shape
    rows = np.ascontiguousarray(rgb.astype(">u2")).reshape(height, width * 3)
    raw = b"".join(b"\x00" + rows[y].tobytes() for y in range(height))
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    header = struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def encode16(linear):
    peak = float(np.percentile(linear, 99.9)) or 1.0
    scaled = np.clip(linear / peak, 0, 1)
    return np.round(scaled * 65535).astype(np.uint16), peak


def kelvin_from_rgb(rgb):
    """Correlated color temperature of a linear sRGB color (McCamy approximation)."""
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    x_, y_, z_ = m @ rgb
    total = x_ + y_ + z_
    x, y = x_ / total, y_ / total
    n = (x - 0.3320) / (0.1858 - y)
    return float(449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33)


def find_emitters(photo, photo_luma, unclipped, residual_dominance, dominance, min_luminance, source_clipped, clipped_candidates):
    """Bright regions of light that is not diffusely reflected: candidates are bright pixels the model assigns to the
    residual, plus (clipped_candidates) overexposed pixels. Light seen directly (a window, a lamp) is either; the model
    is inconsistent about which. Regions mostly overexposed are light sources, the rest reflections; the layout stage
    reclassifies light sources that sit on confirmed objects as highlights."""
    import cv2

    candidates = (residual_dominance > dominance) & (photo_luma >= min_luminance)
    if clipped_candidates:
        candidates |= ~unclipped
    candidates = cv2.morphologyEx(candidates.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    min_area = 0.001 * labels.size
    emitters = []
    emitter_map = np.zeros(labels.shape, dtype=np.uint8)
    for k in sorted(range(1, count), key=lambda k: -stats[k, cv2.CC_STAT_AREA]):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < min_area or len(emitters) >= 255:
            continue
        region = labels == k
        emitter_id = len(emitters) + 1
        emitter_map[region] = emitter_id
        x, y, w, h = (int(stats[k, i]) for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        clipped = float(1 - unclipped[region].mean())
        emitters.append({
            "id": emitter_id,
            "kind": "light_source" if clipped >= source_clipped else "reflection",
            "image_bbox_px": [x, y, x + w, y + h],
            "area_fraction": round(area / labels.size, 5),
            "centroid_px": [round(float(c), 1) for c in centroids[k]],
            "mean_linear_rgb": [round(float(c), 4) for c in photo[region].mean(0)],
            "relative_luminance": round(float(photo_luma[region].mean() / np.median(photo_luma)), 3),
            "clipped_fraction": round(clipped, 3),
        })
    return emitters, emitter_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--postprocess-only", action="store_true", help="reuse the cached model prediction of this index")
    # Defaults chosen on the benchmark training renders by windows located within 1 m in 3D (light_select.py):
    # 0.17 with the previous rule (0.6, 0.5, 0.5, 0), 0.58 with this one.
    parser.add_argument("--dominance", type=float, default=0.4, help="residual share above which a bright pixel is non-diffuse")
    parser.add_argument("--min-luminance", type=float, default=0.3, help="linear photo luminance for emitter candidates")
    parser.add_argument("--source-clipped", type=float, default=0.7, help="clipped share that makes a region a light source")
    parser.add_argument("--clipped-candidates", type=int, default=1, help="1: overexposed pixels are emitter candidates too")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    photo = srgb_to_linear(np.asarray(image, dtype=np.float64) / 255.0)
    cache = f"{args.out_dir.rstrip('/')}/.{args.index}-light-prediction.npz"

    started = time.time()
    if args.postprocess_only:
        cached = np.load(cache)
        albedo, shading, residual = (cached[k].astype(np.float64) for k in ("albedo", "shading", "residual"))
    else:
        import torch
        from diffusers import MarigoldIntrinsicsPipeline

        pipe = MarigoldIntrinsicsPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        result = pipe(
            image,
            num_inference_steps=args.steps,
            ensemble_size=args.ensemble,
            processing_resolution=args.resolution,
            output_type="np",
            generator=torch.Generator().manual_seed(args.seed),
        )
        names = pipe.target_properties["target_names"]
        prediction = {name: result.prediction[k].astype(np.float32) for k, name in enumerate(names)}
        np.savez_compressed(cache, **prediction)  # raw model output, so thresholds can be refit without inference
        albedo, shading, residual = (prediction[k].astype(np.float64) for k in ("albedo", "shading", "residual"))
    seconds = time.time() - started

    # Shading and residual are up to scale: fit photo ~= a * albedo * shading + b * residual.
    # Clipped pixels are not linear measurements, so they stay out of the fit and the check.
    srgb = np.asarray(image, dtype=np.float64) / 255.0
    unclipped = (srgb < 0.98).all(-1)
    design = np.stack([(albedo * shading)[unclipped].reshape(-1), residual[unclipped].reshape(-1)], axis=1)
    target = photo[unclipped].reshape(-1)
    (a, b), *_ = np.linalg.lstsq(design, target, rcond=None)
    a, b = max(float(a), 1e-6), max(float(b), 0.0)
    reconstruction = design @ np.array([a, b])
    r2 = 1 - np.sum((target - reconstruction) ** 2) / np.sum((target - target.mean()) ** 2)
    shading_lin = a * shading
    residual_lin = b * residual

    shading_luma = shading_lin @ LUMA
    residual_luma = residual_lin @ LUMA
    photo_luma = photo @ LUMA
    weights = shading_luma.reshape(-1)
    light_rgb = (shading_lin.reshape(-1, 3) * weights[:, None]).sum(0) / max(weights.sum(), 1e-9)
    light_rgb = light_rgb / max(light_rgb @ LUMA, 1e-9)

    diffuse_luma = (albedo * shading_lin) @ LUMA
    residual_dominance = residual_luma / np.maximum(diffuse_luma + residual_luma, 1e-9)
    emitters, emitter_map = find_emitters(photo, photo_luma, unclipped, residual_dominance, args.dominance,
                                          args.min_luminance, args.source_clipped, args.clipped_candidates)

    outside = unclipped & (emitter_map == 0)
    out = args.out_dir.rstrip("/")
    n = args.index
    files = {
        "albedo": f"{out}/{n}-light-albedo.png",
        "shading": f"{out}/{n}-light-shading.png",
        "residual": f"{out}/{n}-light-residual.png",
        "shading_preview": f"{out}/{n}-light-shading-preview.png",
        "emitters": f"{out}/{n}-light-emitters.png",
    }
    Image.fromarray(np.round(linear_to_srgb(albedo) * 255).astype(np.uint8)).save(files["albedo"])
    shading16, shading_peak = encode16(shading_lin)
    residual16, residual_peak = encode16(residual_lin)
    write_png16_rgb(files["shading"], shading16)
    write_png16_rgb(files["residual"], residual16)
    preview = linear_to_srgb(shading_lin / max(np.percentile(shading_luma, 99), 1e-9))
    Image.fromarray(np.round(preview * 255).astype(np.uint8)).save(files["shading_preview"])
    Image.fromarray(emitter_map).save(files["emitters"])

    summary = {
        "schema_version": 1,
        "kind": "light",
        "model": MODEL_ID,
        "decomposition": "photo_linear = albedo * shading + residual",
        "source_image": args.image,
        "image_size": [image.width, image.height],
        "settings": {"steps": args.steps, "ensemble": args.ensemble, "processing_resolution": args.resolution, "seed": args.seed,
                     "dominance": args.dominance, "min_luminance": args.min_luminance, "source_clipped": args.source_clipped,
                     "clipped_candidates": bool(args.clipped_candidates)},
        "prediction_cache": cache,
        "inference_seconds": round(seconds, 1),
        "scales": {"shading": round(a, 6), "residual": round(b, 6)},
        "reconstruction_r2": round(float(r2), 4),
        "reconstruction_scope": "unclipped pixels",
        "encoding": {
            "albedo": "8-bit sRGB",
            "shading": {"format": "16-bit RGB linear", "value = sample / 65535 *": round(shading_peak, 6)},
            "residual": {"format": "16-bit RGB linear", "value = sample / 65535 *": round(residual_peak, 6)},
            "emitters": "8-bit region id, 0 = none",
        },
        "light_color": {
            "linear_rgb": [round(float(c), 4) for c in light_rgb],
            "cct_kelvin": round(kelvin_from_rgb(light_rgb)),
        },
        "clipped_fraction": round(float(1 - unclipped.mean()), 4),
        "nondiffuse_fraction": round(float(residual_luma[outside].sum() / max(photo_luma[outside].sum(), 1e-9)), 4),
        "nondiffuse_fraction_scope": "unclipped pixels outside emitter regions",
        "shading_dynamic_range": round(float(np.percentile(shading_luma, 99) / max(np.percentile(shading_luma, 1), 1e-6)), 2),
        "emitters": emitters,
        "files": files,
    }
    with open(f"{out}/{n}-light.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("reconstruction_r2", "scales", "light_color", "nondiffuse_fraction", "inference_seconds")} | {"emitters": len(emitters)}))


if __name__ == "__main__":
    main()
