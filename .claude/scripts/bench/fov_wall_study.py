import json, sys, glob, numpy as np
from PIL import Image
R="/home/user/image-sniper"
def walls_raw(L, wdir):
    lab = np.asarray(Image.open(f"{wdir}/0-layout-labels.png").convert("RGB")).astype(int)
    dep = np.asarray(Image.open(f"{wdir}/0-layout-depth.png")).astype(float) / 1000
    h, w = dep.shape
    W0, H0 = L["camera"]["image_size"]; K = L["camera"]["intrinsics_px"]
    fx, fy, cx, cy = K["fx"]*w/W0, K["fy"]*h/H0, K["cx"]*w/W0, K["cy"]*h/H0
    v, u = np.mgrid[0:h, 0:w]
    P = np.stack([(u+0.5-cx)/fx*dep, (v+0.5-cy)/fy*dep, dep], -1)
    M = np.array(L["camera"]["matrix_world_from_camera_opencv"])
    Rw, tw = M[:3, :3], M[:3, 3]
    wallpx = (lab == [120, 120, 120]).all(-1) & (dep > 0)
    pts = P[wallpx]
    planes = [p for p in L["structure"] if p["class"] == "wall"]
    if len(planes) < 2: return []
    # plane in camera frame: n_c = Rw^T n, point c_c = Rw^T (c - tw)
    ncs = [Rw.T @ np.array(p["normal"]) for p in planes]; ccs = [Rw.T @ (np.array(p["center"]) - tw) for p in planes]
    d = np.stack([np.abs((pts - c) @ n) for n, c in zip(ncs, ccs)], 1)
    a = d.argmin(1); ok = d.min(1) < 0.05
    out = []
    for k in range(len(planes)):
        q = pts[(a == k) & ok]
        if len(q) < 200: out.append(None); continue
        q = q - q.mean(0); n = np.linalg.svd(q, full_matrices=False)[2][-1]
        out.append(n)
    return [n for n in out if n is not None]
for split in ("train", "val"):
    rows = []
    for gtp in sorted(glob.glob(f"{R}/benchmark/renders/{split}/room-*/view-0/gt.json")):
        room = gtp.split("room-")[1].split("/")[0]
        gt = json.load(open(gtp)); wdir = f"{R}/worlds/bench-{split}-room-{room}-view-0/output/layout"
        L = json.load(open(f"{wdir}/0-layout.json"))
        k = np.tan(np.radians(L["camera"]["fov_x_deg"]/2)) / np.tan(np.radians(gt["camera"]["fov_x_deg"]/2))
        ns = walls_raw(L, wdir); best = None
        for i in range(len(ns)):
            for j in range(i+1, len(ns)):
                a, b = ns[i], ns[j]
                if abs(a @ b) > 0.5: continue
                num = -(a[0]*b[0] + a[1]*b[1]); den = a[2]*b[2]
                if abs(den) < 0.05 or num / den <= 0: continue
                if best is None or abs(den) > best[1]: best = (np.sqrt(num/den), abs(den), np.degrees(np.arccos(abs(a @ b))))
        if best: rows.append((k, 1/best[0], best[1], best[2]))
    a = np.array(rows)
    print(split, "pairs", len(a), "raw pair angle median %.2f (90 - x: %.2f)" % (np.median(a[:, 3]), np.median(90 - a[:, 3])))
    print("  log k truth median %.4f, est median %.4f, corr %.3f" % (np.median(np.log(a[:, 0])), np.median(np.log(a[:, 1])), np.corrcoef(np.log(a[:, 0]), np.log(a[:, 1]))[0, 1]))
    print("  abs log focal err: before %.4f after %.4f" % (np.median(np.abs(np.log(a[:, 0]))), np.median(np.abs(np.log(a[:, 0]/a[:, 1])))))
    for thr in (0.1, 0.2, 0.3):
        m = a[:, 2] > thr
        if m.sum(): print(f"  den>{thr}: n {m.sum()} before %.4f after %.4f" % (np.median(np.abs(np.log(a[m, 0]))), np.median(np.abs(np.log(a[m, 0]/a[m, 1])))))
