#!/usr/bin/env python3
"""Metric-scale correction study: the depth model as predicted, real-world anchors (build-layout scale_anchors),
and a ridge regressor fitted on train from the same layout quantities. Fit on train, compare on val.

Needs layouts built with scale anchors on (their JSON records the anchor evidence and the applied scale) and
scored as <name>-<split>.json. For each view: raw error = log(true / depth model) = log(true / final) + log(applied).
Features: log camera height, ceiling present and its log height, object anchors present and their mean log
evidence, fov. Writes benchmark/results/scale-study.json.
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RESULTS = os.path.join(REPO, "benchmark", "results")


def rows(name, split):
    report = json.load(open(os.path.join(RESULTS, f"{name}-{split}.json")))
    out = []
    for view in report["views"]:
        if "depth" not in view:
            continue
        layout = json.load(open(os.path.join(REPO, "worlds", view["view"], "output", "layout", f"{view['layout_index']}-layout.json")))
        scale = layout.get("scale", {})
        applied = scale.get("applied", 1.0)
        evidence = scale.get("evidence", [])
        ceiling = [e for e in evidence if e["source"].startswith("ceiling")]
        objects = [e for e in evidence if "height prior (" in e["source"]]
        cam = [e for e in evidence if e["source"].startswith("camera")]
        features = [cam[0]["log_scale"] if cam else 0.0, float(bool(ceiling)), ceiling[0]["log_scale"] if ceiling else 0.0,
                    float(bool(objects)), float(np.mean([e["log_scale"] for e in objects])) if objects else 0.0,
                    layout["camera"]["fov_x_deg"] / 60 - 1]
        out.append({"view": view["view"], "raw": view["depth"]["log_scale_err"] + float(np.log(applied)),
                    "anchors": view["depth"]["log_scale_err"], "features": features})
    return out


def ridge(X, y, lam):
    X1 = np.c_[X, np.ones(len(X))]
    A = X1.T @ X1 + lam * np.diag([1] * X.shape[1] + [0])
    return np.linalg.solve(A, X1.T @ y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="anchors")
    args = parser.parse_args()
    train, val = rows(args.name, "train"), rows(args.name, "val")
    X, y = np.array([r["features"] for r in train]), np.array([r["raw"] for r in train])
    # Ridge strength by leave-one-out on train.
    best = None
    for lam in (0.01, 0.1, 1, 10, 100):
        errs = [abs(y[i] - np.r_[X[i], 1] @ ridge(np.delete(X, i, 0), np.delete(y, i), lam)) for i in range(len(y))]
        if best is None or np.median(errs) < best[1]:
            best = (lam, float(np.median(errs)))
    w = ridge(X, y, best[0])
    Xv = np.array([r["features"] for r in val])
    predicted = np.c_[Xv, np.ones(len(Xv))] @ w
    errors = {"depth model": np.array([abs(r["raw"]) for r in val]), "anchors": np.array([abs(r["anchors"]) for r in val]),
              "regressor": np.abs(np.array([r["raw"] for r in val]) - predicted)}
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(val), len(val)) for _ in range(2000)]
    ci = lambda e: [round(float(np.percentile([np.median(e[i]) for i in idx], q)), 4) for q in (2.5, 97.5)]
    diff = lambda a, b: [round(float(np.percentile([np.median(errors[b][i]) - np.median(errors[a][i]) for i in idx], q)), 4) for q in (2.5, 97.5)]
    report = {
        "train_views": len(train), "val_views": len(val), "ridge_lambda": best[0],
        "coefficients": dict(zip(["log_cam_evidence", "ceiling_present", "ceiling_log_evidence", "objects_present", "objects_log_evidence", "fov_rel", "bias"], np.round(w, 4).tolist())),
        "val_median_abs_log_error": {k: round(float(np.median(v)), 4) for k, v in errors.items()},
        "val_ci95": {k: ci(v) for k, v in errors.items()},
        "val_difference_ci95": {"anchors - depth model": diff("depth model", "anchors"), "regressor - depth model": diff("depth model", "regressor"),
                                "regressor - anchors": diff("anchors", "regressor")},
        "train_raw_log_bias": round(float(np.mean(y)), 4),
    }
    json.dump(report, open(os.path.join(RESULTS, "scale-study.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
