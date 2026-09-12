#!/usr/bin/env bash
set -euo pipefail

source_dir="${SOURCE_DIR:-/home/yuan/projects/pi05-finetune-lab/upload/pi05_panda_multi_object_box_v2/}"
remote_host="${REMOTE_HOST:?Set REMOTE_HOST}"
remote_user="${REMOTE_USER:?Set REMOTE_USER}"
remote_port="${REMOTE_PORT:-22}"
remote_dir="${REMOTE_DIR:-/root/shared-nvme/pi05_panda_multi_object_box_v2/}"

if [[ ! -d "$source_dir" ]]; then
  echo "Local source directory not found: $source_dir" >&2
  exit 1
fi

echo "Uploading $source_dir to $remote_host:$remote_dir"
rsync -ah --info=progress2 --partial --no-owner --no-group --no-perms \
  -e "ssh -p $remote_port -l $remote_user" \
  "$source_dir" \
  "$remote_host:$remote_dir"
