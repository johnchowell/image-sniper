#!/usr/bin/env bash
# Setup of the benchmark renderer runtime: .venv-render with Blender as a Python module (bpy needs Python 3.11).
set -euo pipefail
cd "$(dirname "$0")/../../.."

stamp=$(sha256sum .claude/scripts/bench/setup.sh | cut -c1-16)
if [ -x .venv-render/bin/python ] && [ "$(cat .venv-render/.setup-stamp 2>/dev/null)" = "$stamp" ]; then
  echo "Render runtime ready: .venv-render/bin/python"
  exit 0
fi
[ -x .venv-render/bin/python ] || python3.11 -m venv .venv-render
.venv-render/bin/pip install "bpy==4.5.14" "numpy==1.26.4" "requests==2.34.2"
echo "$stamp" > .venv-render/.setup-stamp
echo "Render runtime ready: .venv-render/bin/python"
