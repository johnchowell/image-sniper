#!/usr/bin/env python3
"""Clean plate with LaMa (big-lama): removes every confirmed object instance found by the layout.

The mask is the union of the layout's instance masks, dilated so contact shadows and mask edges go too.
Writes the next source index: worlds/<world>/source/N-<slug>-plate.png and .N-<slug>-plate-request.json.
"""
import argparse
import os
import time

import cv2
import numpy as np
from PIL import Image

from common import load_layout, load_mask, next_index, object_instances, world_path, write_request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--dilate", type=float, default=0.012, help="mask dilation as a fraction of the image diagonal")
    args = parser.parse_args()

    _, layout_path, layout = load_layout(args.world)
    source = layout["source_image"]
    image = Image.open(source).convert("RGB")
    size = image.size

    mask = np.zeros((size[1], size[0]), dtype=bool)
    removed = []
    for obj, instance in object_instances(layout):
        mask |= load_mask(instance["mask_file"], size)
        removed.append(instance["id"])
    if not removed:
        raise SystemExit("The layout has no object instances to remove.")
    radius = max(1, int(args.dilate * np.hypot(*size)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    mask = cv2.dilate(mask.astype(np.uint8) * 255, kernel) > 0

    from simple_lama_inpainting import SimpleLama

    started = time.time()
    plate = SimpleLama()(image, Image.fromarray(mask.astype(np.uint8) * 255))
    plate = plate.crop((0, 0, size[0], size[1]))  # LaMa pads to a multiple of 8

    source_dir = world_path(args.world, "source")
    index = next_index(source_dir)
    slug = os.path.splitext(os.path.basename(source))[0].split("-", 1)[1] + "-plate"
    out = os.path.join(source_dir, f"{index}-{slug}.png")
    mask_out = os.path.join(source_dir, f".{index}-{slug}-mask.png")
    plate.save(out)
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_out)
    write_request(os.path.join(source_dir, f".{index}-{slug}-request.json"), {
        "kind": "2d",
        "role": "plate",
        "provider": "local/big-lama",
        "endpoint": "local/big-lama",
        "index": index,
        "status": "completed",
        "input_files": [source],
        "layout": layout_path,
        "removed_instances": removed,
        "mask_file": mask_out,
        "mask_dilation_px": radius,
        "mask_fraction": round(float(mask.mean()), 4),
        "inference_seconds": round(time.time() - started, 1),
        "output_files": [out],
    })
    print(f'{{"plate": "{out}", "removed": {len(removed)}, "mask_fraction": {mask.mean():.4f}, "seconds": {time.time() - started:.1f}}}')


if __name__ == "__main__":
    main()
