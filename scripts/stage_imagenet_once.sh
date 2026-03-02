#!/usr/bin/env bash
set -euo pipefail

# One-time copy of ImageNet-1k into this repository.
# Usage:
#   bash scripts/stage_imagenet_once.sh /cluster/shared/imagenet1k data/imagenet1k

SRC_DIR="${1:-/cluster/shared/imagenet1k}"
DST_DIR="${2:-data/imagenet1k}"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "Source directory does not exist: ${SRC_DIR}" >&2
  exit 1
fi

mkdir -p "${DST_DIR}"

echo "Copying ImageNet-1k from '${SRC_DIR}' to '${DST_DIR}'..."
rsync -a --info=progress2 "${SRC_DIR}/" "${DST_DIR}/"
echo "Done."

echo "Expected structure:"
echo "  ${DST_DIR}/train"
echo "  ${DST_DIR}/val"
