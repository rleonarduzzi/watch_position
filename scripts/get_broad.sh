#!/usr/bin/env bash
# Fetch the BROAD benchmark recordings into data/broad.
#
# BROAD: Berlin Robust Orientation Estimation Assessment Dataset
# (Laidig, Caruso, Cereatti, Seel, 2021), CC-BY-4.0.
# A shallow clone is enough; the history is large and we only need the HDF5
# recordings, trials.json and the reference results used to validate our metrics.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="${BROAD_ROOT:-$root/data/broad}"

if [ -d "$dest/data_hdf5" ]; then
    echo "BROAD already present at $dest"
    exit 0
fi

mkdir -p "$(dirname "$dest")"
git clone --depth 1 https://github.com/dlaidig/broad.git "$dest"

n=$(find "$dest/data_hdf5" -name '*.hdf5' | wc -l)
echo "fetched $n recordings into $dest"
[ "$n" -eq 39 ] || { echo "expected 39 recordings, got $n" >&2; exit 1; }
