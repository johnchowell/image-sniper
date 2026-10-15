"""Shared helpers for the local (open-weight) generation pass: project paths, indexed files, layout data."""
import contextlib
import fcntl
import json
import math
import os
import re
import time

import numpy as np
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
INDEXED = re.compile(r"^(\.)?(\d+)-(.+?)(?:__([a-z0-9._-]+))?(-request\.json|\.[^.]+)$", re.I)


def world_path(world, *parts):
    return os.path.join("worlds", world, *parts)


@contextlib.contextmanager
def world_lock(world):
    """Holds worlds/<world>/.lock for a whole pass: a second pass on the same world fails at once instead of
    writing into the same generation index (indexes are chosen by scanning the directory)."""
    path = world_path(world, ".lock")
    with open(path, "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another pass is running on {world} ({path} is locked).")
        handle.write(str(os.getpid()))
        handle.flush()
        yield


def next_index(directory):
    os.makedirs(directory, exist_ok=True)
    indexes = [int(m.group(2)) for name in os.listdir(directory) if (m := INDEXED.match(name))]
    return max(indexes, default=-1) + 1


def latest(directory, slug, extension):
    """Highest-index visible file N-<slug><extension> in a directory."""
    best = None
    for name in os.listdir(directory) if os.path.isdir(directory) else []:
        m = INDEXED.match(name)
        if m and not m.group(1) and m.group(3) == slug and m.group(5) == extension:
            if best is None or int(m.group(2)) > best[0]:
                best = (int(m.group(2)), os.path.join(directory, name))
    return best


def exif_fov_x(path):
    """Horizontal field of view (deg) of the stored pixel grid from the photo's EXIF focal length, with its basis;
    None when the file has no usable focal length. The depth model then solves only depth, not depth and focal."""
    try:
        image = Image.open(path)
        sub = image.getexif().get_ifd(0x8769)
    except Exception:
        return None
    W, H = image.size
    f35 = float(sub.get(0xA405) or 0)
    if f35 > 0:  # 35 mm equivalent: matched on the diagonal of a 36 x 24 mm frame
        half = 43.2666 / (2 * f35) * W / math.hypot(W, H)
        basis = f"FocalLengthIn35mmFilm {f35:g} mm"
    else:  # true focal length and the sensor width from the focal-plane resolution
        f = float(sub.get(0x920A) or 0)
        resolution = float(sub.get(0xA20E) or 0)
        unit_mm = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}.get(int(sub.get(0xA210) or 2))
        if not (f > 0 and resolution > 0 and unit_mm):
            return None
        # focal length in stored pixels; the long-side ratio corrects a resize and is the same for a rotated grid
        sampled = max(float(sub.get(0xA002) or 0), float(sub.get(0xA003) or 0)) or max(W, H)
        f_px = f * resolution / unit_mm * max(W, H) / sampled
        half = W / (2 * f_px)
        basis = f"FocalLength {f:g} mm, {f_px:.0f} px"
    fov = math.degrees(2 * math.atan(half))
    return (round(fov, 2), basis) if 15 <= fov <= 130 else None


def write_request(path, payload):
    payload = {"schema_version": 1, **payload, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_layout(world):
    found = latest(world_path(world, "output", "layout"), "layout", ".json")
    if not found:
        raise SystemExit(f"No layout for {world}. Run image-blast-layout first.")
    with open(found[1]) as f:
        return found[0], found[1], json.load(f)


def load_mask(path, size):
    """Binary mask from a PNG, resized (nearest) to size=(width, height)."""
    image = Image.open(path)
    if image.mode in ("RGBA", "LA"):
        alpha = np.asarray(image.getchannel("A"))
        values = alpha if alpha.min() < 255 else np.asarray(image.convert("L"))
    else:
        values = np.asarray(image.convert("L"))
    mask = Image.fromarray(((values > 127) * 255).astype(np.uint8))
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    return np.asarray(mask) > 0


def object_instances(layout):
    for obj in layout["objects"]:
        for instance in obj["instances"]:
            yield obj, instance


def camera_matrix(layout):
    """4x4 layout-from-OpenCV-camera matrix and pixel intrinsics."""
    cam = layout["camera"]
    return np.array(cam["matrix_world_from_camera_opencv"], dtype=np.float64), cam["intrinsics_px"], cam["image_size"]
