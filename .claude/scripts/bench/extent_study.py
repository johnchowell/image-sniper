import sys, os, json, numpy as np
sys.path.insert(0, "/home/user/image-sniper/.claude/scripts/bench")
import evaluate as E
from PIL import Image
from collections import defaultdict
split = sys.argv[1]
rows = []
for vd in E.views(split):
    gt = json.load(open(os.path.join(vd, "gt.json"))); arr = np.load(os.path.join(vd, "gt.npz"))
    world = os.path.join(E.REPO, "worlds", E.world_name(vd))
    layout = json.load(open(os.path.join(world, "output", "layout", "0-layout.json")))
    R_gt, t_gt = E.gt_camera(gt); R_pc, t_pc = E.pred_cam(layout)
    inst = arr["instance"]; H, W = inst.shape
    dt = arr["depth"].astype(float)
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
        if best is None or bi < 0.25 or not best.get("corners"): continue
        c, s = np.cos(np.radians(obj["yaw_deg"])), np.sin(np.radians(obj["yaw_deg"]))
        sx, sy, sz = obj["size"]
        loc = np.array([[x, y, z] for x in (-sx/2, sx/2) for y in (-sy/2, sy/2) for z in (-sz/2, sz/2)])
        gc = E.to_cam(loc @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + np.array(obj["center"]), R_gt, t_gt)
        pc = scale * E.to_cam(np.array(best["corners"]), R_pc, t_pc)
        rows.append({"view": os.path.basename(world), "name": obj["name"], "edges": "+".join(best.get("truncated_edges") or []) or "none",
                     "basis": best.get("depth_basis", "-"),
                     "lat": float(np.ptp(pc[:, 0]) / np.ptp(gc[:, 0]) - 1), "dep": float(np.ptp(pc[:, 2]) / np.ptp(gc[:, 2]) - 1),
                     "near": float(pc[:, 2].min() - gc[:, 2].min()), "far": float(pc[:, 2].max() - gc[:, 2].max()),
                     "iou": E.box_iou(pc, gc)})

groups = defaultdict(list)
for r in rows: groups[r["edges"]].append(r)
print(f"{'edges':22s}{'n':>4s}{'lat':>8s}{'dep':>8s}{'near':>8s}{'far':>8s}{'iou':>7s}")
for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    med = lambda f: np.median([r[f] for r in v])
    print(f"{k:22s}{len(v):4d}{med('lat'):8.3f}{med('dep'):8.3f}{med('near'):8.3f}{med('far'):8.3f}{np.mean([r['iou'] for r in v]):7.3f}")
