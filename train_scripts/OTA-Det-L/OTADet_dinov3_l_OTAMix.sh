#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="./outputs/OTA-Det-L/OTA-Mix-FourData-LADO"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/OTADet_dinov3_l_OTAMix.log"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} torchrun \
    --nnodes=${NNODES:-1} \
    --node_rank=${NODE_RANK:-0} \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-7773} \
    --nproc_per_node=${NPROC_PER_NODE:-8} \
    train.py \
    -c configs/OTA-Det/OTA-Det-L/OTADet_dinov3_l_OTAMix.yml \
    --seed=0 \
    ${EXTRA_ARGS:-} \
    2>&1 | tee "$LOG_FILE"
