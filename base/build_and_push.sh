#!/usr/bin/env bash
# Build and push the template base images with drift-hashed tags.
#
# Each flavor is tagged <registry>/lbx-tasks-base<suffix>:<prefix>-<drifthash>,
# where <drifthash> is a 12-char content hash over every base build input
# (base Dockerfiles, install scripts, requirements, and the grader /
# taiga_runtime packages baked in). Editing any input changes the tag, so a
# stale base is never silently reused.
#
# Usage:
#   base/build_and_push.sh [--push] [--flavors cpu,gpu,gpu-openroad,gpu-blackwell,cuda-graphics,tpu]
#
# Env:
#   REGISTRY   container registry prefix
#              (default: us-east1-docker.pkg.dev/gcp-taiga/labelbox)
#   PLATFORM   build platform (default: linux/amd64)
#
# Without --push the images are built locally (--load) for inspection. With
# --push they are pushed to $REGISTRY. Requires `docker buildx`.
#
# Parent-chained flavors FROM a sibling base; expand_base_flavors inserts parents
# before children, so listing a child alone also builds its parent(s) first.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="${REGISTRY:-us-east1-docker.pkg.dev/gcp-taiga/labelbox}"
PLATFORM="${PLATFORM:-linux/amd64}"
PUSH=false
FLAVORS="cpu,gpu,gpu-openroad,gpu-blackwell"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --push) PUSH=true; shift ;;
    --flavors) FLAVORS="$2"; shift 2 ;;
    --flavors=*) FLAVORS="${1#*=}"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Single source of truth for the drift hash + per-flavor metadata.
DRIFT_HASH="$(
  cd "$REPO_ROOT" && python3 -c "
from pathlib import Path
import sys
sys.path.insert(0, 'alignerr_plugin/src')
from alignerr_plugin.base_image import base_drift_hash
print(base_drift_hash(Path('.')))
"
)"
echo ":: base drift hash = ${DRIFT_HASH}"

EXPANDED_FLAVORS="$(
  cd "$REPO_ROOT" && python3 -c "
import sys
sys.path.insert(0, 'alignerr_plugin/src')
from alignerr_plugin.base_image import expand_base_flavors
print(','.join(expand_base_flavors('$FLAVORS'.split(','))))
"
)"
FLAVORS="${EXPANDED_FLAVORS}"

flavor_meta() {  # $1 = flavor -> prints "<suffix> <dockerfile> <tag_prefix>"
  cd "$REPO_ROOT" && python3 -c "
import sys
sys.path.insert(0, 'alignerr_plugin/src')
from alignerr_plugin.base_image import BASE_FLAVORS
f = BASE_FLAVORS['$1']
print(f.image_suffix, f.dockerfile, f.tag_prefix)
"
}

parent_meta() {  # $1 = flavor -> prints "<parent_suffix> <parent_tag_prefix>" or "" (no parent)
  cd "$REPO_ROOT" && python3 -c "
import sys
sys.path.insert(0, 'alignerr_plugin/src')
from alignerr_plugin.base_image import BASE_FLAVORS
f = BASE_FLAVORS['$1']
if f.parent:
    p = BASE_FLAVORS[f.parent]
    print(p.image_suffix, p.tag_prefix)
"
}

IFS=',' read -ra FLAVOR_LIST <<< "$FLAVORS"
for flavor in "${FLAVOR_LIST[@]}"; do
  read -r suffix dockerfile tag_prefix <<< "$(flavor_meta "$flavor")"
  image="${REGISTRY}/lbx-tasks-base${suffix}"
  tag="${tag_prefix}-${DRIFT_HASH}"
  ref="${image}:${tag}"
  echo ":: building ${flavor} -> ${ref} (from ${dockerfile})"
  load_or_push="--load"
  if [[ "$PUSH" == "true" ]]; then load_or_push="--push"; fi

  # Parent-chained flavors FROM a sibling base via ARG BASE_IMAGE/BASE_TAG; pass
  # the parent flavor's suffix + tag_prefix. Flavors with no parent skip these.
  parent_build_args=()
  read -r parent_suffix parent_tag_prefix <<< "$(parent_meta "$flavor")"
  if [[ -n "${parent_suffix:-}" ]]; then
    parent_build_args=(
      --build-arg "BASE_IMAGE=${REGISTRY}/lbx-tasks-base${parent_suffix}"
      --build-arg "BASE_TAG=${parent_tag_prefix}-${DRIFT_HASH}"
    )
  fi

  docker buildx build \
    --platform "$PLATFORM" \
    --file "${REPO_ROOT}/${dockerfile}" \
    --tag "$ref" \
    "${parent_build_args[@]}" \
    --label "lbx.base.flavor=${flavor}" \
    --label "lbx.base.drift_hash=${DRIFT_HASH}" \
    "$load_or_push" \
    "$REPO_ROOT"
  echo ":: done ${ref}"
done

# Point the exporter at the freshly-built cpu/gpu/gpu-openroad/cuda-graphics tag.
# Blackwell and TPU use distinct tag prefixes/env vars.
echo ":: export with: LBX_RL_TASKS_BASE_IMAGE_TAG=runtime-ml-core-py313-${DRIFT_HASH}"
if printf '%s\n' "${FLAVOR_LIST[@]}" | grep -qx gpu-blackwell; then
  echo "::            and LBX_RL_TASKS_BLACKWELL_BASE_IMAGE_TAG=runtime-ml-blackwell-py313-${DRIFT_HASH}"
fi
if printf '%s\n' "${FLAVOR_LIST[@]}" | grep -qx tpu; then
  echo "::            and LBX_RL_TASKS_TPU_BASE_IMAGE_TAG=runtime-ml-tpu-py312-${DRIFT_HASH}"
fi
