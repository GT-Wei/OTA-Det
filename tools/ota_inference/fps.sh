#!/usr/bin/env bash
set -euo pipefail

# Measure OTA-Det inference speed over a config's validation dataloader.
# Paper-style FPS uses offline text encoding: record text features once, then reuse them for timing.

CONFIG=${CONFIG:-configs/OTA-Det/OTA-Det-L/OTADet_dinov3_l_OTAMix.yml}
CHECKPOINT=${CHECKPOINT:-ckpts/OTA-Det-L.pth}
DEVICE=${DEVICE:-cuda}
CACHE_PATH=${CACHE_PATH:-outputs/text_cache_otadet_l.pt}
WARMUP_ITERS=${WARMUP_ITERS:-20}
MAX_ITERS=${MAX_ITERS:-300}

mkdir -p "$(dirname "$CACHE_PATH")"

python tools/ota_inference/torch_fps_OTA_det.py \
    -c "$CONFIG" \
    -r "$CHECKPOINT" \
    -d "$DEVICE" \
    --text-cache-mode record \
    --text-cache-path "$CACHE_PATH" \
    --text-cache-max "$MAX_ITERS" \
    --warmup-iters "$WARMUP_ITERS" \
    --max-iters "$MAX_ITERS" \
    --skip-aggregation

python tools/ota_inference/torch_fps_OTA_det.py \
    -c "$CONFIG" \
    -r "$CHECKPOINT" \
    -d "$DEVICE" \
    --text-cache-mode reuse \
    --text-cache-path "$CACHE_PATH" \
    --warmup-iters "$WARMUP_ITERS" \
    --max-iters "$MAX_ITERS" \
    --skip-aggregation
