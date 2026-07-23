#!/usr/bin/env bash
# Pack a task's read-only mounts into content-addressed squashfs objects, upload
# them once to the shared cache, and stamp the deploy manifest
# (.alignerr/preloaded_files.json) so the platform mounts each object read-only
# at deploy time instead of baking datasets/weights into the per-task image.
#
# Two task shapes are handled:
#   * NATIVE (task.toml): expose bounded public data/ trees (or prompt-referenced
#     files from larger trees) for every task type; auto-mount the conventional
#     ML dataset dirs; and process declared [[preloaded_files]]. A declared
#     mount_path wins over an auto one at the same path.
#   * metadata mode (metadata.json, no task.toml):
#       - PUBLIC: expose bounded data/public trees (or prompt-referenced files)
#         for every ml_task_type so Taiga QA sees the logical payload.
#       - DATASET: additionally mount public/private data as squashfs fallbacks.
#         env / hybrid / sim_policy retain their baked data for runtime behavior.
#       - HF: hf_resources repos are fetched into an HF hub-cache layout
#         (refs/snapshots/blobs) and packed for offline from_pretrained,
#         content-addressed by commit sha under cache/huggingface/. Gated repos
#         need HF_TOKEN in the env.
#
# Usage: scripts/sync_mount.sh <problem-dir>
# Requires: mksquashfs, jq, curl, python3; TAIGA_ENV_ID + TAIGA_TOKEN for upload;
#           HF_TOKEN only for gated hf_resources.
set -euo pipefail

PROBLEM_DIR="${1:?usage: sync_mount.sh <problem-dir>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_ID="$(basename "$PROBLEM_DIR")"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PACK_HF="${REPO_ROOT}/scripts/pack_hf_resource.py"
UPLOAD="${REPO_ROOT}/scripts/upload_squashfs.sh"
TAIGA_API="${TAIGA_API:-https://taiga.ant.dev/api}"
BUCKET_PREFIX="${TAIGA_PRELOADED_BUCKET_PREFIX:-gs://anthropic-argonrl-dog-bowl-us-central1-0/biome/environment_files}"

# Deterministic bucket path for a content-addressed object. Every stamped
# remote_path derives it this ONE way (matching upload_squashfs.sh's fallback) so
# the same object always stamps the same mount source, cache-hit or -miss.
_remote_path_for() {
  printf '%s/%s/%s' "$BUCKET_PREFIX" "${TAIGA_ENV_ID:-ENV}" "$1"
}

# Is this a metadata-mode task? (metadata.json + no task.toml.)
IS_MLENVS="$(cd "$REPO_ROOT" && python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, 'alignerr_plugin/src')
from alignerr_plugin.mlenvs import is_mlenvs_task
print('1' if is_mlenvs_task(Path('$PROBLEM_DIR')) else '0')
")"

STAMP_ARGS=()
QA_PUBLIC_EXPANSION_COMPLETE=false

# --- metadata dataset data mounts -----------------------------------------
# public -> /data (agent-visible), private -> /mcp_server/data (root-only). Only
# for ml_task_type == "dataset"; env / hybrid / sim_policy bake their data.
_DATA_IDX=0
_pack_data_tree() {
  local src="$1" mount_path="$2" name address remote_name squashfs remote_path
  # Skip a missing / empty tree.
  if [[ ! -d "$src" ]] || [[ -z "$(find "$src" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo ":: skipping ${mount_path} (${src} missing or empty)"
    return 0
  fi
  _DATA_IDX=$((_DATA_IDX + 1))
  name="d${_DATA_IDX}"
  address="$(cd "$REPO_ROOT" && python3 -c "
import sys; sys.path.insert(0, 'alignerr_plugin/src')
from pathlib import Path
from alignerr_plugin.preloaded import content_address
print(content_address(Path('${src}')))
")"
  remote_name="$(cd "$REPO_ROOT" && python3 -c "
import sys; sys.path.insert(0, 'alignerr_plugin/src')
from alignerr_plugin.preloaded import remote_squashfs_name
print(remote_squashfs_name('${TASK_ID}', '${name}', '${address}'))
")"
  squashfs="${WORK}/${name}-${address}.squashfs"
  echo ":: packing ${name} (${mount_path}, ${address})"
  # zstd level 3, matching pack_hf_resource.py.
  mksquashfs "$src" "$squashfs" -noappend -quiet -comp zstd -Xcompression-level 3
  bash "$UPLOAD" "$squashfs" "$remote_name" >/dev/null
  remote_path="$(_remote_path_for "$remote_name")"
  STAMP_ARGS+=(--entry "${mount_path}::${remote_path}::true")
}

_upload_qa_visible_public_tree() {
  local src="$1" mount_path="$2" selection expand complete address encoded entry relative local_path remote_name remote_path
  QA_PUBLIC_EXPANSION_COMPLETE=false
  selection="$(cd "$REPO_ROOT" && python3 - "$src" "$mount_path" "$PROBLEM_DIR" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "alignerr_plugin/src")
from alignerr_plugin.preloaded import qa_visible_public_file_selection
from alignerr_plugin.utils import read_prompt

try:
    prompt = read_prompt(Path(sys.argv[3]))
except OSError:
    prompt = ""
files, complete = qa_visible_public_file_selection(
    Path(sys.argv[1]),
    sys.argv[2],
    prompt_text=prompt,
)
print(
    json.dumps(
        {
            "expand": files is not None,
            "complete": complete,
            "files": [
                {"relative": relative, "local_path": local_path}
                for relative, local_path in (files or [])
            ],
        },
        separators=(",", ":"),
    )
)
PY
)"
  expand="$(jq -r '.expand' <<<"$selection")"
  [[ "$expand" == "true" ]] || return 1
  complete="$(jq -r '.complete' <<<"$selection")"
  QA_PUBLIC_EXPANSION_COMPLETE="$complete"
  address="$(cd "$REPO_ROOT" && python3 -c "
import sys; sys.path.insert(0, 'alignerr_plugin/src')
from pathlib import Path
from alignerr_plugin.preloaded import content_address
print(content_address(Path('${src}')))
")"
  for encoded in $(jq -r '.files[] | @base64' <<<"$selection"); do
    entry="$(printf '%s' "$encoded" | base64 -d)"
    relative="$(jq -r '.relative' <<<"$entry")"
    local_path="$(jq -r '.local_path' <<<"$entry")"
    if [[ "$complete" != "true" ]]; then
      local_path="/lbx-public-files/${address}/${relative}"
    fi
    remote_name="${TASK_ID}/public-${address}/${relative}"
    remote_path="$(bash "$UPLOAD" "${src}/${relative}" "$remote_name")"
    STAMP_ARGS+=(--entry "${local_path}::${remote_path}::false")
  done
  echo ":: expanded $(jq '.files | length' <<<"$selection") QA-visible public files at ${mount_path} (complete=${complete})"
  return 0
}

sync_mlenvs_data_mounts() {
  local ml_task_type public_expanded=false public_complete=false
  ml_task_type="$(python3 -c "
import json
print(json.load(open('${PROBLEM_DIR}/metadata.json')).get('ml_task_type', ''))
")"
  if _upload_qa_visible_public_tree "${PROBLEM_DIR}/data/public" "/data"; then
    public_expanded=true
    public_complete="$QA_PUBLIC_EXPANSION_COMPLETE"
  fi
  if [[ "$ml_task_type" != "dataset" ]]; then
    echo ":: $TASK_ID ml_task_type=${ml_task_type:-?}: public QA expansion=${public_expanded}; data remains baked"
    return 0
  fi
  if [[ "$public_complete" != "true" ]]; then
    _pack_data_tree "${PROBLEM_DIR}/data/public" "/data"
  fi
  _pack_data_tree "${PROBLEM_DIR}/data/private" "/mcp_server/data"
}

# --- HF resources ----------------------------------------------------------
# Return 0 iff <basename> is already listed under <remote_dir>, so a cache-hit
# skips the download. Any error / missing token is treated as a miss.
hf_remote_exists() {
  local remote_dir="$1" basename="$2"
  local token="${TAIGA_TOKEN:-}"
  [[ -n "$token" && -n "${TAIGA_ENV_ID:-}" ]] || return 1
  local resp code
  resp="$(mktemp)"
  code="$(curl -sS -G "$TAIGA_API/environments/${TAIGA_ENV_ID}/files/list" \
            --data-urlencode "path=$remote_dir" \
            -H "Authorization: Bearer $token" \
            -o "$resp" -w "%{http_code}" || echo 000)"
  if [[ "$code" == 2* ]] && \
     jq -e --arg n "$basename" 'any(.[]; (.name == $n) or ((.path // "") | endswith("/" + $n)))' "$resp" >/dev/null 2>&1; then
    rm -f "$resp"; return 0
  fi
  rm -f "$resp"; return 1
}

sync_hf_resources() {
  local metadata="${PROBLEM_DIR}/metadata.json"
  [[ -f "$metadata" ]] || return 0
  local resources_raw
  resources_raw="$(python3 "$PACK_HF" --metadata "$metadata" --list)" \
    || { echo "ERROR: failed to read hf_resources from $metadata" >&2; exit 1; }
  [[ -n "$resources_raw" ]] || { echo ":: $TASK_ID declares no hf_resources"; return 0; }
  local -a resources
  mapfile -t resources <<<"$resources_raw"
  local res resolved repo_id commit_sha mount_path remote_dir remote_base remote_name remote_path packed packed_path
  for res in "${resources[@]}"; do
    resolved="$(python3 "$PACK_HF" --resource "$res" --resolve-only)"
    repo_id="$(   jq -r '.repo_id'          <<<"$resolved")"
    commit_sha="$(jq -r '.commit_sha'       <<<"$resolved")"
    mount_path="$(jq -r '.mount_local_path' <<<"$resolved")"
    remote_dir="$( jq -r '.remote_dir'      <<<"$resolved")"
    remote_base="$(jq -r '.remote_basename' <<<"$resolved")"
    remote_name="$(jq -r '.remote_name'     <<<"$resolved")"
    if hf_remote_exists "$remote_dir" "$remote_base"; then
      echo ":: cache hit: ${repo_id}@${commit_sha} already at ${remote_name}" >&2
    else
      echo ":: cache miss: ${repo_id}@${commit_sha} -- downloading + packing" >&2
      # Reuse the resolved sha so bytes, object name, and mount share ONE resolve.
      packed="$(python3 "$PACK_HF" --resource "$res" --pack --out-dir "$WORK" --commit-sha "$commit_sha")"
      packed_path="$(jq -r '.packed_path' <<<"$packed")"
      [[ -f "$packed_path" ]] || { echo "ERROR: pack produced no file for $repo_id" >&2; exit 1; }
      bash "$UPLOAD" "$packed_path" "$remote_name" >/dev/null
      rm -f "$packed_path"
    fi
    # Both branches stamp the ONE deterministic object path (see _remote_path_for).
    remote_path="$(_remote_path_for "$remote_name")"
    STAMP_ARGS+=(--entry "${mount_path}::${remote_path}::true")
  done
}

# --- Native auto-mount + declared preloaded_files -------------------------
sync_native_mounts() {
  local ENTRY_SEP=$'\x1f'
  local -a entries
  mapfile -t entries < <(cd "$REPO_ROOT" && python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, 'alignerr_plugin/src')
from alignerr_plugin.preloaded import auto_mount_entries
from alignerr_plugin.utils import load_task_toml
sep = '\x1f'
pd = Path('$PROBLEM_DIR')
task = load_task_toml(pd)
declared = task.preloaded_files
declared_paths = {e.mount_path for e in declared}
auto = auto_mount_entries(
    pd, task.difficulty.task_type, hidden_env=task.environment.hidden_env
)
auto_paths = {mount_path for _source_rel, mount_path in auto}
for source_rel, mount_path in auto:
    if mount_path in declared_paths:
        continue
    print(sep.join([source_rel, '', '', mount_path, 'true', 'auto']))
if '/data' not in declared_paths and '/data' not in auto_paths and (pd / 'data').is_dir():
    print(sep.join(['data', '', '', '/data', 'true', 'public']))
for e in declared:
    print(sep.join([
        e.source, e.hf_repo, e.hf_revision,
        e.mount_path, 'true' if e.read_only else 'false',
        'declared',
    ]))
")
  local line source hf_repo hf_revision mount_path read_only origin idx=0 name staging address remote_name squashfs remote_path
  for line in "${entries[@]}"; do
    IFS="${ENTRY_SEP}" read -r source hf_repo hf_revision mount_path read_only origin <<< "$line"
    if [[ -n "$source" && "$read_only" == "true" \
          && ( "$mount_path" == "/data" || "$mount_path" == /data/* ) ]]; then
      if _upload_qa_visible_public_tree "${PROBLEM_DIR}/${source}" "$mount_path"; then
        if [[ "$QA_PUBLIC_EXPANSION_COMPLETE" == "true" || "$origin" == "public" ]]; then
          continue
        fi
      fi
    fi
    if [[ "$origin" == "public" ]]; then
      echo ":: public data remains baked; no bounded QA-visible expansion available"
      continue
    fi
    idx=$((idx + 1))
    name="m${idx}"
    staging="${WORK}/${name}"
    mkdir -p "$staging"
    if [[ -n "$source" ]]; then
      cp -a "${PROBLEM_DIR}/${source}/." "$staging/"
    else
      echo ":: fetching HF repo ${hf_repo}@${hf_revision:-main}"
      python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='${hf_repo}', revision='${hf_revision}' or None, local_dir='${staging}')
"
    fi
    address="$(cd "$REPO_ROOT" && python3 -c "
import sys; sys.path.insert(0, 'alignerr_plugin/src')
from pathlib import Path
from alignerr_plugin.preloaded import content_address
print(content_address(Path('${staging}')))
")"
    remote_name="$(cd "$REPO_ROOT" && python3 -c "
import sys; sys.path.insert(0, 'alignerr_plugin/src')
from alignerr_plugin.preloaded import remote_squashfs_name
print(remote_squashfs_name('${TASK_ID}', '${name}', '${address}'))
")"
    squashfs="${WORK}/${name}-${address}.squashfs"
    echo ":: packing ${name} (${address}) -> ${squashfs}"
    mksquashfs "$staging" "$squashfs" -noappend -quiet
    remote_path="$(bash "$UPLOAD" "$squashfs" "$remote_name")"
    STAMP_ARGS+=(--entry "${mount_path}::${remote_path}::${read_only}")
  done
}

if [[ "$IS_MLENVS" == "1" ]]; then
  sync_mlenvs_data_mounts
  sync_hf_resources
else
  sync_native_mounts
fi

if [[ "${#STAMP_ARGS[@]}" -eq 0 ]]; then
  echo ":: $TASK_ID has no datasets to mount (no data, no preloaded_files, no hf_resources)"
  exit 0
fi

python3 "${REPO_ROOT}/scripts/stamp_preloaded_files.py" \
  --problem-dir "$PROBLEM_DIR" "${STAMP_ARGS[@]}"
echo ":: synced $(( ${#STAMP_ARGS[@]} / 2 )) mount(s) for ${TASK_ID}"
