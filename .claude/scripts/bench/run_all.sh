#!/usr/bin/env bash
# Rebuilds the benchmark from nothing: assets, renders, worlds, pipeline outputs, scores.
#   bash .claude/scripts/bench/run_all.sh [stage ...]   stages: setup assets render prepare run score (default: all)
# One room per seed, one view per room (views of one room are not independent samples).
# Splits by seed: train 1001-1060, val 1061-1080, test 1081-1100. Test is scored once per work package
# (evaluate.py logs every test read in benchmark/results/test-reads.log).
set -euo pipefail
cd "$(dirname "$0")/../../.."

declare -A FIRST=([train]=1001 [val]=1061 [test]=1081)
declare -A LAST=([train]=1060 [val]=1080 [test]=1100)
stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(setup assets render prepare run score)
splits=${SPLITS:-"train val test"}

for stage in "${stages[@]}"; do
  case "$stage" in
    setup)
      bash .claude/scripts/local/setup.sh
      bash .claude/scripts/bench/setup.sh ;;
    assets)
      .venv/bin/python .claude/scripts/bench/fetch_assets.py ;;
    render)
      for split in $splits; do
        for seed in $(seq "${FIRST[$split]}" "${LAST[$split]}"); do
          out="benchmark/renders/$split/room-$seed"
          [ -f "$out/view-0/gt.json" ] && continue
          .venv-render/bin/python .claude/scripts/bench/render_scene.py --out "$out" --seed "$seed" 2>&1 | grep '^{"view"'
        done
      done ;;
    prepare)
      for split in $splits; do .venv/bin/python .claude/scripts/bench/evaluate.py prepare --split "$split"; done ;;
    run)
      for split in $splits; do .venv/bin/python .claude/scripts/bench/batch_inputs.py --split "$split"; done ;;
    score)
      for split in train val; do .venv/bin/python .claude/scripts/bench/evaluate.py score --split "$split" --name current; done ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
  esac
done
