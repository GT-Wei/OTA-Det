#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATA_ROOT=${DATA_ROOT:-../datasets/LLM_Caption_Parse}
API_BASE=${API_BASE:-http://localhost}
START_PORT=${START_PORT:-18000}
API_KEY=${API_KEY:-EMPTY}
MODEL=${MODEL:-openai/gpt-oss-20b}
GPUS=${GPUS:-7}
BATCH_SIZE=${BATCH_SIZE:-32}
PER_GPU_CONCURRENCY=${PER_GPU_CONCURRENCY:-4}
DELAY=${DELAY:-0.2}
MAX_RETRIES=${MAX_RETRIES:-5}

run_split() {
    local split=$1
    python "${SCRIPT_DIR}/run_vlm_parse_batch.py" \
        --input-dir "${DATA_ROOT}/annotations/${split}" \
        --out-dir "${DATA_ROOT}/annotations_parse/${split}" \
        --api-base "$API_BASE" \
        --start-port "$START_PORT" \
        --api-key "$API_KEY" \
        --model "$MODEL" \
        --gpus "$GPUS" \
        --batch-size "$BATCH_SIZE" \
        --per-gpu-concurrency "$PER_GPU_CONCURRENCY" \
        --delay "$DELAY" \
        --max-retries "$MAX_RETRIES" \
        --skip-existing-out
}

run_split OPT_RSVG_test
run_split OPT_RSVG_train
run_split OPT_RSVG_val
