"""Shared helpers for the local (open-weight) generation pass: project paths, indexed files, layout data."""
import contextlib
import fcntl
import json
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
