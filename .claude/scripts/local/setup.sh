#!/usr/bin/env bash
# One-time setup of the local model runtime: .venv, pinned packages, and pinned GitHub repos in third_party/.
set -euo pipefail
cd "$(dirname "$0")/../../.."

python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.1" "torchvision==0.29.1"
.venv/bin/pip install -r .claude/scripts/local/requirements.txt
# simple-lama-inpainting pins pillow<10; its code runs on the pinned pillow, so its metadata pins are skipped.
.venv/bin/pip install --no-deps simple-lama-inpainting==0.1.2

clone() {  # repo commit directory
  if [ ! -d "third_party/$3" ]; then git clone --quiet "https://github.com/$1" "third_party/$3"; fi
  git -C "third_party/$3" fetch --quiet --depth 1 origin "$2" && git -C "third_party/$3" checkout --quiet "$2"
}
mkdir -p third_party
clone VAST-AI-Research/TripoSR 107cefdc244c39106fa830359024f6a2f1c78871 TripoSR
clone tatsy/torchmcubes 879926d0ef58e6ce0ac2630fdecb5e53af7ed3ff torchmcubes
.venv/bin/pip install --no-build-isolation ./third_party/torchmcubes

echo "Local runtime ready: .venv/bin/python (set IMAGE_BLAST_PYTHON to use another interpreter)."
