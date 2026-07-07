#!/usr/bin/env bash
set -euo pipefail

# OTA-Det-L visualization demo. Set IMAGE to your own aerial image before running.
# Override CONFIG, CHECKPOINT, IMAGE, TEXTS, OUTPUT, SCORE, or TOPK for custom data.

CONFIG=${CONFIG:-configs/OTA-Det/OTA-Det-L/OTADet_dinov3_l_OTAMix.yml}
CHECKPOINT=${CHECKPOINT:-ckpts/OTA-Det-L.pth}
IMAGE=${IMAGE:-path/to/aerial_image.jpg}
TEXTS=${TEXTS:-tools/ota_inference/caption_example.json}
OUTPUT=${OUTPUT:-outputs/demo_result.jpg}
SCORE=${SCORE:-0.4}
TOPK=${TOPK:-1}

mkdir -p "$(dirname "$OUTPUT")"

python tools/ota_inference/torch_inf_OTA_det.py \
    -c "$CONFIG" \
    -r "$CHECKPOINT" \
    -i "$IMAGE" \
    -t "$TEXTS" \
    -o "$OUTPUT" \
    -s "$SCORE" \
    --topk "$TOPK"
