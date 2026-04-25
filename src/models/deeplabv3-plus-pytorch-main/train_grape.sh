#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p outputs/grape_seg/weights outputs/grape_seg/logs

nohup python train.py > outputs/grape_seg/train.log 2>&1 &

echo "Training started in background."
echo "Log file: outputs/grape_seg/train.log"
echo "Watch with: tail -f outputs/grape_seg/train.log"
