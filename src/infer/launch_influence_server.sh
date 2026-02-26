#!/usr/bin/env bash
set -euo pipefail

source .env

OPEN_R1_ROOT="$(pwd)"
OLMO_ROOT="$OPEN_R1_ROOT/../OLMo"

GCS_ROOT="gs://cmu-gpucloud-zichunyu/healthcare/olmo"
# BEFORE_CKPT="${LOCAL_ROOT}/out/OLMo-300M/dclm_2.3B_all_repetition/step43680-unsharded"
BEFORE_CKPT="${LOCAL_ROOT}/out/OLMo-300M/dclm_2.3B_organic+recycled_repetition/step48048-unsharded"
AFTER_CKPT="${BEFORE_CKPT}/data_influence/step841-unsharded"

# Create local directory if it doesn't exist
mkdir -p "${LOCAL_ROOT}/out/OLMo-300M/dclm_2.3B_organic+recycled_repetition/"

# Download checkpoint from GCS if not already present
if [[ ! -d "${BEFORE_CKPT}" ]]; then
    echo "Downloading checkpoint from GCS..."
    gcloud storage cp -r "${GCS_ROOT}/out/OLMo-300M/dclm_2.3B_organic+recycled_repetition/step43680-unsharded" "${LOCAL_ROOT}/out/dclm_2.3B_organic+recycled_repetition/"
fi

export PYTHONPATH="${OPEN_R1_ROOT}/src:${OLMO_ROOT}:${PYTHONPATH:-}"
source "${OLMO_ROOT}/.venv/bin/activate"

mkdir -p "/scratch/zichunyu/logs"
python -m src.infer.influence_server \
  --before-checkpoint "${BEFORE_CKPT}" \
  --after-checkpoint "${AFTER_CKPT}" \
  --host 0.0.0.0 \
  --port 24776 \
  --batch-size 8 \
  --max-seq-len 2048 \
  > "/scratch/zichunyu/logs/influence_server.log" 2>&1 &
