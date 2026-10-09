#!/usr/bin/env python3
"""Downloads CC0 assets from Poly Haven for the ground-truth benchmark: indoor models (glTF, 1k textures),
floor/wall textures, and outdoor HDRIs (seen through window openings, and the daylight that enters). Writes benchmark/assets/manifest.json with license and source for every asset.
"""
import argparse
import hashlib
import json
import os
import random
import urllib.request

API = "https://api.polyhaven.com"
HEADERS = {"User-Agent": "image-blaster-benchmark/1.0 (CC0 asset fetch for rendering ground truth)"}
INDOOR = {"seating", "table", "shelves", "furniture", "lighting", "decorative", "plants", "electronics", "containers", "appliances"}
OUTDOOR = {"nature", "rocks", "trees", "ground cover", "structures", "industrial", "collection: project_lighthouse",
           "collection: namaqualand", "collection: verdant_trail", "collection: smugglers_cove", "collection: hidden_alley"}


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
        return json.load(response)


def download(url, path, md5=None):
    if os.path.exists(path) and (md5 is None or hashlib.md5(open(path, "rb").read()).hexdigest() == md5):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as response:
        data = response.read()
    if md5 and hashlib.md5(data).hexdigest() != md5:
        raise RuntimeError(f"Checksum mismatch for {url}")
    with open(path, "wb") as f:
        f.write(data)


def fetch_model(slug, out_dir):
    files = get_json(f"{API}/files/{slug}")["gltf"]["1k"]["gltf"]
    root = os.path.join(out_dir, "models", slug)
    download(files["url"], os.path.join(root, f"{slug}.gltf"), files.get("md5"))
    for rel, entry in files["include"].items():
        download(entry["url"], os.path.join(root, rel), entry.get("md5"))
    return os.path.join(root, f"{slug}.gltf")


def fetch_texture(slug, out_dir):
    files = get_json(f"{API}/files/{slug}")
    maps = {}
    for key in ("Diffuse", "Rough", "nor_gl"):
        entry = files.get(key, {}).get("1k", {}).get("jpg")
        if entry:
            path = os.path.join(out_dir, "textures", slug, f"{slug}_{key}.jpg")
            download(entry["url"], path, entry.get("md5"))
            maps[key] = path
    return maps


def fetch_hdri(slug, out_dir):
    entry = get_json(f"{API}/files/{slug}")["hdri"]["1k"]["hdr"]
    path = os.path.join(out_dir, "hdris", f"{slug}_1k.hdr")
    download(entry["url"], path, entry.get("md5"))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmark/assets")
    parser.add_argument("--models", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--textures-per-role", type=int, default=5)
    parser.add_argument("--hdris", type=int, default=10)
    args = parser.parse_args()

    catalog = get_json(f"{API}/assets?type=models")
    candidates = sorted(
        slug for slug, info in catalog.items()
        if INDOOR & set(info.get("categories", [])) and not OUTDOOR & set(info.get("categories", []))
    )
    random.Random(args.seed).shuffle(candidates)
    by_category = {}
    for slug in candidates:  # round-robin over primary categories for variety
        primary = next((c for c in catalog[slug]["categories"] if c in INDOOR - {"furniture"}), "furniture")
        by_category.setdefault(primary, []).append(slug)
    chosen = []
    while len(chosen) < args.models and any(by_category.values()):
        for slugs in by_category.values():
            if slugs and len(chosen) < args.models:
                chosen.append(slugs.pop(0))

    manifest = {"source": "https://polyhaven.com", "license": "CC0 1.0", "models": [], "textures": [], "hdris": []}
    for slug in chosen:
        try:
            path = fetch_model(slug, args.out)
        except Exception as error:  # some assets lack a 1k glTF; skip them
            print(f"skip {slug}: {error}")
            continue
        info = catalog[slug]
        manifest["models"].append({"slug": slug, "name": info.get("name", slug), "categories": info.get("categories", []),
                                   "authors": list(info.get("authors", {}).keys()), "path": path,
                                   "page": f"https://polyhaven.com/a/{slug}"})
        print(f"model {slug}")

    textures = get_json(f"{API}/assets?type=textures")
    wanted = {"floor": ("floor", "wood"), "wall": ("wall", "plaster")}
    for role, words in wanted.items():
        slugs = sorted(s for s, info in textures.items() if all(any(w in c for c in info.get("categories", [])) for w in words[:1])
                       and any(words[1] in t for t in info.get("tags", []) + info.get("categories", [])))[:args.textures_per_role]
        for slug in slugs:
            maps = fetch_texture(slug, args.out)
            dimensions = get_json(f"{API}/info/{slug}").get("dimensions")  # real-world size of one tile, mm
            manifest["textures"].append({"slug": slug, "role": role, "maps": maps, "tile_m": dimensions[0] / 1000 if dimensions else None,
                                         "page": f"https://polyhaven.com/a/{slug}"})
            print(f"texture {role} {slug}")

    hdris = get_json(f"{API}/assets?type=hdris")
    outdoor = sorted(s for s, info in hdris.items() if "outdoor" in info.get("categories", []) and "night" not in info.get("categories", []))
    random.Random(args.seed).shuffle(outdoor)
    for slug in outdoor[:args.hdris]:
        path = fetch_hdri(slug, args.out)
        manifest["hdris"].append({"slug": slug, "categories": hdris[slug].get("categories", []), "path": path,
                                  "page": f"https://polyhaven.com/a/{slug}"})
        print(f"hdri {slug}")

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"{len(manifest['models'])} models, {len(manifest['textures'])} textures, {len(manifest['hdris'])} hdris")


if __name__ == "__main__":
    main()
