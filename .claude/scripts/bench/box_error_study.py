import sys, os, json, numpy as np
sys.path.insert(0, "/home/user/image-sniper/.claude/scripts/bench")
import evaluate as E
from PIL import Image
from collections import defaultdict
split = sys.argv[1]
def corners(center, size, yaw):  # gt convention: z up, yaw about z
    c, s = np.cos(yaw), np.sin(yaw); sx, sy, sz = size
    loc = np.array([[x, y, z] for x in (-sx/2, sx/2) for y in (-sy/2, sy/2) for z in (-sz/2, sz/2)])
    return loc @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + center
def fit(pc):  # pred corners (gt world frame) -> center, size (x, y, z-up), yaw
    c = pc.mean(0); bot = pc[pc[:, 2] < c[2]]
    e1 = bot[1] - bot[0]; e2 = bot[3] - bot[0] if np.linalg.norm(bot[3]-bot[0]) > 1e-6 else bot[2]-bot[0]
    # bottom face corners: find two orthogonal edges
    d = [bot[k] - bot[0] for k in range(1, 4)]; d.sort(key=lambda v: np.linalg.norm(v))
    a, b = d[0], d[1]
    return c, np.array([np.linalg.norm(a), np.linalg.norm(b), np.ptp(pc[:, 2])]), np.arctan2(a[1], a[0])
rows = []
for vd in E.views(split):
    gt = json.load(open(os.path.join(vd, "gt.json"))); arr = np.load(os.path.join(vd, "gt.npz"))
    world = os.path.join(E.REPO, "worlds", E.world_name(vd))
    layout = json.load(open(os.path.join(world, "output", "layout", "0-layout.json")))
    R_gt, t_gt = E.gt_camera(gt); R_pc, t_pc = E.pred_cam(layout)
    inst = arr["instance"]; H, W = inst.shape; dt = arr["depth"].astype(float)
    pdp = np.asarray(Image.open(os.path.join(world, "output", "layout", "0-layout-depth.png"))).astype(float) / 1000
    st = layout["depth"]["grid_step_px"]; gh, gw = pdp.shape
    ys = np.minimum(H-1, np.arange(gh)*st+st//2); xs = np.minimum(W-1, np.arange(gw)*st+st//2)
    g = dt[ys][:, xs]; ok = (pdp > 0) & np.isfinite(g) & (g > 0) & (g < 50)
    scale = float(np.median(g[ok] / pdp[ok]))
    cands = defaultdict(list)
    for o in layout["objects"]:
        for i in o["instances"]: cands[o["object_id"]].append(i)
    for obj in gt["objects"]:
        m = inst == obj["pass_index"]
        if m.sum() < E.TRUTH_PX: continue
        best, bi = None, 0
        for i in cands.get(E.slug(obj["id"]), []):
            pm = np.asarray(Image.open(os.path.join(E.REPO, i["mask_file"])).convert("L").resize((W, H), Image.NEAREST)) > 127
            iou = (pm & m).sum() / max((pm | m).sum(), 1)
            if iou > bi: best, bi = i, iou
        if best is None or bi < 0.25 or not best.get("corners") or best.get("truncated_edges"): continue
        if obj["size"][1] < 0.08 or obj["size"][0] < 0.08: continue  # thin objects
        gc_w = corners(np.array(obj["center"]), obj["size"], np.radians(obj["yaw_deg"]))
        # pred corners into gt world frame via camera frames
        pc_cam = scale * E.to_cam(np.array(best["corners"]), R_pc, t_pc)
        pc_w = (pc_cam - 0) @ np.eye(3)
        Rg, tg = R_gt, t_gt
        # camera -> gt world: invert to_cam (p_cam = R (p - t) or similar); solve generically
        A = E.to_cam(np.eye(3), Rg, tg) - E.to_cam(np.zeros((1, 3)), Rg, tg)  # rows: images of unit vectors
        o0 = E.to_cam(np.zeros((1, 3)), Rg, tg)[0]
        pw = np.linalg.solve(A.T, (pc_cam - o0).T).T
        pcen, psize, pyaw = fit(pw)
        gcen, gsize, gyaw = np.array(obj["center"]), np.array(obj["size"]), np.radians(obj["yaw_deg"])
        def iou(c, s, y): return E.box_iou(E.to_cam(corners(c, s, y), Rg, tg), E.to_cam(gc_w, Rg, tg))
        # yaw ambiguity: pred size axes may be swapped; pick assignment best matching gt
        best_y = min([(pyaw + k*np.pi/2, psize if k % 2 == 0 else psize[[1, 0, 2]]) for k in range(4)], key=lambda t: abs(np.angle(np.exp(1j*(t[0]-gyaw)))))
        pyaw2, psize2 = best_y
        gcam = E.to_cam(gcen[None], Rg, tg)[0]; pcam = E.to_cam(pcen[None], Rg, tg)[0]
        rows.append({"dx": pcam[0]-gcam[0], "dy": pcam[1]-gcam[1], "dz": pcam[2]-gcam[2], "gz": gcam[2], "gdepth": float(np.ptp(E.to_cam(gc_w, Rg, tg)[:, 2])), "name": obj["name"], "pred": iou(pcen, psize2, pyaw2), "gt_center": iou(gcen, psize2, pyaw2),
                     "gt_size": iou(pcen, gsize, pyaw2), "gt_yaw": iou(pcen, psize2, gyaw), "gt_center_size": iou(gcen, gsize, pyaw2),
                     "yaw_err": float(np.degrees(abs(np.angle(np.exp(1j*(pyaw2-gyaw))))) )})
for k in ("pred", "gt_center", "gt_size", "gt_yaw", "gt_center_size"):
    print(f"{k:16s}{np.mean([r[k] for r in rows]):.3f}")
print("n", len(rows), "yaw err median %.1f p75 %.1f" % tuple(np.percentile([r["yaw_err"] for r in rows], [50, 75])))
for k in ("dx", "dy", "dz"):
    v = np.array([r[k] for r in rows]); print(k, "signed median %+.3f  mean %+.3f  median abs %.3f" % (np.median(v), v.mean(), np.median(np.abs(v))))
dz = np.array([r["dz"] for r in rows]); gd = np.array([r["gdepth"] for r in rows])
print("dz / true depth extent: median %+.3f" % np.median(dz / gd))
