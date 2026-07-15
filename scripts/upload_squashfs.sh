#!/usr/bin/env bash
# Upload a squashfs object to the shared environment-files cache via Taiga's
# presigned-upload-url API. Content-addressed names mean re-uploading an
# identical object is a no-op (the platform already has it -> HTTP 409, treated
# as success). Ported from ML_Envs/scripts/upload_squashfs.sh.
#
# Usage: upload_squashfs.sh <local_squashfs> <remote_name>
# Env:
#   TAIGA_API   Taiga API base (default: https://taiga.ant.dev/api)
#   TAIGA_ENV_ID   environment id the files belong to (required)
#   TAIGA_TOKEN    bearer token (required)
set -euo pipefail

LOCAL_PATH="${1:?usage: upload_squashfs.sh <local_squashfs> <remote_name>}"
REMOTE_NAME="${2:?usage: upload_squashfs.sh <local_squashfs> <remote_name>}"
TAIGA_API="${TAIGA_API:-https://taiga.ant.dev/api}"
ENV_ID="${TAIGA_ENV_ID:?TAIGA_ENV_ID is required}"
TOKEN="${TAIGA_TOKEN:?TAIGA_TOKEN is required}"
BUCKET_PREFIX="${TAIGA_PRELOADED_BUCKET_PREFIX:-gs://anthropic-argonrl-dog-bowl-us-central1-0/biome/environment_files}"

SIZE="$(stat -f%z "$LOCAL_PATH" 2>/dev/null || stat -c%s "$LOCAL_PATH")"
FULL_REMOTE_PATH="${BUCKET_PREFIX}/${ENV_ID}/${REMOTE_NAME}"

# Transient failures (network, 408, 429, any 5xx) are retried with linear
# backoff -- Taiga's presigned endpoint and GCS occasionally return a momentary
# 502/503 that a single-shot request would surface as a hard delivery failure.
MAX_ATTEMPTS="${UPLOAD_MAX_ATTEMPTS:-5}"

is_transient_code() {
  # 000 = curl transport failure (connection reset / timeout).
  case "$1" in
    000 | 408 | 425 | 429 | 5[0-9][0-9]) return 0 ;;
    *) return 1 ;;
  esac
}

# Request a presigned PUT URL for this content-addressed object, retrying
# transient failures. 409 (already present) and 2xx are terminal successes.
RESP="$(mktemp)"
HTTP_CODE=""
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  HTTP_CODE="$(curl -sS -o "$RESP" -w '%{http_code}' \
    -X POST "${TAIGA_API}/environments/${ENV_ID}/presigned-upload-url" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d "$(jq -nc --arg file_path "$REMOTE_NAME" --argjson file_size_bytes "$SIZE" \
          '{file_path: $file_path, file_size_bytes: $file_size_bytes}')" || echo 000)"
  if [[ "$HTTP_CODE" == "409" || "$HTTP_CODE" == "200" || "$HTTP_CODE" == "201" ]]; then
    break
  fi
  if is_transient_code "$HTTP_CODE" && [[ "$attempt" -lt "$MAX_ATTEMPTS" ]]; then
    sleep_s=$(( attempt * 5 ))
    echo ":: presigned-upload-url transient failure (${HTTP_CODE}); retry ${attempt}/${MAX_ATTEMPTS} in ${sleep_s}s" >&2
    sleep "$sleep_s"
    continue
  fi
  break
done

if [[ "$HTTP_CODE" == "409" ]]; then
  echo ":: ${REMOTE_NAME} already present (content-addressed); skipping upload" >&2
  echo "${FULL_REMOTE_PATH}"
  rm -f "$RESP"
  exit 0
fi
if [[ "$HTTP_CODE" != "200" && "$HTTP_CODE" != "201" ]]; then
  echo "presigned-upload-url failed (${HTTP_CODE}): $(cat "$RESP")" >&2
  rm -f "$RESP"
  exit 1
fi

UPLOAD_URL="$(jq -r '.url // .upload_url // empty' "$RESP")"
CONTENT_LENGTH_RANGE="$(jq -r '.required_headers["X-Goog-Content-Length-Range"] // .required_headers["x-goog-content-length-range"] // empty' "$RESP")"
FULL_REMOTE_PATH="$(jq -r --arg fallback "$FULL_REMOTE_PATH" '.remote_path // .gcs_path // .path // $fallback' "$RESP")"
rm -f "$RESP"
if [[ -z "$UPLOAD_URL" ]]; then
  echo "presigned-upload-url response did not include upload URL" >&2
  exit 1
fi

echo ":: uploading ${LOCAL_PATH} -> ${REMOTE_NAME}" >&2
PUT_RESP="$(mktemp)"
PUT_CODE=""
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  if [[ -n "$CONTENT_LENGTH_RANGE" ]]; then
    PUT_CODE="$(curl -sS -X PUT "$UPLOAD_URL" \
      -H "X-Goog-Content-Length-Range: ${CONTENT_LENGTH_RANGE}" \
      -H "Content-Type: application/octet-stream" \
      --data-binary "@${LOCAL_PATH}" \
      -o "$PUT_RESP" -w '%{http_code}' || echo 000)"
  else
    PUT_CODE="$(curl -sS -X PUT "$UPLOAD_URL" \
      -H "Content-Type: application/octet-stream" \
      --data-binary "@${LOCAL_PATH}" \
      -o "$PUT_RESP" -w '%{http_code}' || echo 000)"
  fi
  if [[ "$PUT_CODE" == 2* ]]; then
    break
  fi
  if is_transient_code "$PUT_CODE" && [[ "$attempt" -lt "$MAX_ATTEMPTS" ]]; then
    sleep_s=$(( attempt * 5 ))
    echo ":: PUT transient failure (${PUT_CODE}); retry ${attempt}/${MAX_ATTEMPTS} in ${sleep_s}s" >&2
    sleep "$sleep_s"
    continue
  fi
  break
done

if [[ "$PUT_CODE" != 2* ]]; then
  echo "PUT failed (${PUT_CODE}): $(cat "$PUT_RESP")" >&2
  rm -f "$PUT_RESP"
  exit 1
fi
rm -f "$PUT_RESP"

echo ":: uploaded ${REMOTE_NAME}" >&2
echo "${FULL_REMOTE_PATH}"
