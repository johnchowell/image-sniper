#!/usr/bin/env python3
"""Compares build-layout parameter sets on a split: rebuilds every bench world's layout (index 0, no model
inference) with each set, scores it, and accumulates summaries in benchmark/results/params-<study>-<split>.json.
The worlds are left built with the last set given.

  param_select.py --split train --study light-lift --set default '{}' --set support-0.1 '{"light_min_support": 0.1}'
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
import evaluate  # noqa: E402


def rebuild(split, params, names="oracle"):
    for view_dir in evaluate.views(split):
        world = evaluate.world_name(view_dir, names)
        subprocess.run(["node", ".claude/scripts/layout/build-layout.mjs", "--world", world, "--index", "0", "--params", json.dumps(params)],
                       cwd=REPO, check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--study", required=True)
    parser.add_argument("--set", nargs=2, action="append", metavar=("NAME", "JSON"), required=True)
    parser.add_argument("--save-scores", action="store_true", help="also write <study>-<name>-<split>.json per set (for evaluate.py compare)")
    args = parser.parse_args()
    path = os.path.join(REPO, "benchmark", "results", f"params-{args.study}-{args.split}.json")
    rows = json.load(open(path)) if os.path.exists(path) else {}
    for name, text in args.set:
        params = json.loads(text)
        rebuild(args.split, params)
        results = [evaluate.score_view(v) for v in evaluate.views(args.split)]
        summary = evaluate.summarize(results)
        rows[name] = {"params": params, **summary}
        if args.save_scores:
            json.dump({"name": f"{args.study}-{name}", "split": args.split, "names": "oracle", "summary": summary, "views": results},
                      open(os.path.join(REPO, "benchmark", "results", f"{args.study}-{name}-{args.split}.json"), "w"), indent=2, default=float)
        print(name, json.dumps({k: summary[k] for k in ("depth", "structure", "light")}), flush=True)
        json.dump(rows, open(path, "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
