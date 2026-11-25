#!/usr/bin/env bash
# Download the released pi05_droid_jointpos checkpoint into weights/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p weights
src=gs://openpi-assets-simeval/pi05_droid_jointpos
gsutil -m cp -r "$src/params" "$src/assets" weights
