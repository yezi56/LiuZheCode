#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="python3"
SKIP_APT=0
SETUP_ONLY=0
TORCH_CHANNEL="cu124"

DATASET="black_rot"
DATA_ROOT="${ROOT_DIR}/datasets/data/black_rot"
IMAGE_DIR="images"
MASK_DIR=""
INPUT_PATH="${ROOT_DIR}/datasets/data/black_rot/images"

MODEL="deeplabv3plus_mobilenet"
CKPT=""
GPU_ID="0"
OUTPUT_STRIDE="16"
SEPARABLE_CONV=0
PRETRAINED_BACKBONE=0

CROP_SIZE="512"
BATCH_SIZE="4"
VAL_BATCH_SIZE="2"
NUM_WORKERS="4"
VAL_NUM_WORKERS="2"
VAL_PERCENT="0.2"
LR="0.01"
LOSS_TYPE="cross_entropy"
TOTAL_ITRS=""
EPOCHS=""

SAVE_COLOR_DIR="${ROOT_DIR}/outputs/color"
SAVE_MASK_DIR="${ROOT_DIR}/outputs/mask"

EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash server_run.sh setup [options]
  bash server_run.sh train [options] [-- extra main.py args]
  bash server_run.sh predict [options] [-- extra predict.py args]

Common options:
  --venv-dir PATH             Virtualenv path. Default: ./\.venv
  --python-bin BIN            Python binary used to create venv. Default: python3
  --skip-apt                  Skip apt install of python3/python3-venv/python3-pip
  --setup-only                Only create env and install dependencies, then exit
  --torch-channel NAME        cu118|cu121|cu124|cu126|cu128|cu130|cpu|URL
                              Default: cu124

Data options:
  --dataset NAME              Default: black_rot
  --data-root PATH            Default: ./datasets/data/black_rot
  --image-dir PATH            Default: images
  --mask-dir PATH             Optional. Defaults to latest runs/*/lesion_mask

Train options:
  --model NAME                Default: deeplabv3plus_mobilenet
  --ckpt PATH                 Optional checkpoint
  --gpu-id ID                 Default: 0
  --crop-size N               Default: 512
  --batch-size N              Default: 4
  --val-batch-size N          Default: 2
  --num-workers N             Default: 4
  --val-num-workers N         Default: 2
  --val-percent FLOAT         Default: 0.2
  --lr FLOAT                  Default: 0.01
  --loss-type NAME            cross_entropy|focal_loss
  --output-stride N           8|16, default 16
  --total-itrs N              Directly control main.py total iterations
  --epochs N                  Convenience option for black_rot, converted to total_itrs
  --separable-conv            Enable separable conv
  --pretrained-backbone       Enable ImageNet pretrained backbone when ckpt is empty

Predict options:
  --input PATH                Single image or image directory
  --save-color-dir PATH       Default: ./outputs/color
  --save-mask-dir PATH        Default: ./outputs/mask

Examples:
  bash server_run.sh setup --torch-channel cu124
  bash server_run.sh train --ckpt ./checkpoints/best.pth --epochs 20
  bash server_run.sh predict --ckpt ./checkpoints/best.pth --input ./datasets/data/black_rot/images
EOF
}

log() {
  printf '[server_run] %s\n' "$*"
}

require_mode() {
  case "${MODE}" in
    setup|train|predict) ;;
    help|-h|--help|"")
      usage
      exit 0
      ;;
    *)
      echo "Unknown mode: ${MODE}" >&2
      usage
      exit 1
      ;;
  esac
}

resolve_torch_index_url() {
  case "${TORCH_CHANNEL}" in
    cu118) echo "https://download.pytorch.org/whl/cu118" ;;
    cu121) echo "https://download.pytorch.org/whl/cu121" ;;
    cu124) echo "https://download.pytorch.org/whl/cu124" ;;
    cu126) echo "https://download.pytorch.org/whl/cu126" ;;
    cu128) echo "https://download.pytorch.org/whl/cu128" ;;
    cu130) echo "https://download.pytorch.org/whl/cu130" ;;
    cpu) echo "https://download.pytorch.org/whl/cpu" ;;
    http://*|https://*) echo "${TORCH_CHANNEL}" ;;
    *)
      echo "Unsupported --torch-channel: ${TORCH_CHANNEL}" >&2
      exit 1
      ;;
  esac
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --venv-dir) VENV_DIR="$2"; shift 2 ;;
      --python-bin) PYTHON_BIN="$2"; shift 2 ;;
      --skip-apt) SKIP_APT=1; shift ;;
      --setup-only) SETUP_ONLY=1; shift ;;
      --torch-channel) TORCH_CHANNEL="$2"; shift 2 ;;
      --dataset) DATASET="$2"; shift 2 ;;
      --data-root) DATA_ROOT="$2"; shift 2 ;;
      --image-dir) IMAGE_DIR="$2"; shift 2 ;;
      --mask-dir) MASK_DIR="$2"; shift 2 ;;
      --input) INPUT_PATH="$2"; shift 2 ;;
      --model) MODEL="$2"; shift 2 ;;
      --ckpt) CKPT="$2"; shift 2 ;;
      --gpu-id) GPU_ID="$2"; shift 2 ;;
      --crop-size) CROP_SIZE="$2"; shift 2 ;;
      --batch-size) BATCH_SIZE="$2"; shift 2 ;;
      --val-batch-size) VAL_BATCH_SIZE="$2"; shift 2 ;;
      --num-workers) NUM_WORKERS="$2"; shift 2 ;;
      --val-num-workers) VAL_NUM_WORKERS="$2"; shift 2 ;;
      --val-percent) VAL_PERCENT="$2"; shift 2 ;;
      --lr) LR="$2"; shift 2 ;;
      --loss-type) LOSS_TYPE="$2"; shift 2 ;;
      --output-stride) OUTPUT_STRIDE="$2"; shift 2 ;;
      --total-itrs) TOTAL_ITRS="$2"; shift 2 ;;
      --epochs) EPOCHS="$2"; shift 2 ;;
      --save-color-dir) SAVE_COLOR_DIR="$2"; shift 2 ;;
      --save-mask-dir) SAVE_MASK_DIR="$2"; shift 2 ;;
      --separable-conv) SEPARABLE_CONV=1; shift ;;
      --pretrained-backbone) PRETRAINED_BACKBONE=1; shift ;;
      --help|-h)
        usage
        exit 0
        ;;
      --)
        shift
        EXTRA_ARGS=("$@")
        break
        ;;
      *)
        echo "Unknown option: $1" >&2
        usage
        exit 1
        ;;
    esac
  done
}

maybe_install_system_packages() {
  if [[ "${SKIP_APT}" == "1" ]]; then
    return
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    return
  fi

  local -a prefix=()
  if [[ "${EUID}" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      prefix=(sudo)
    else
      echo "apt-get requires root or sudo. Re-run with --skip-apt if Python is already ready." >&2
      exit 1
    fi
  fi

  log "Installing Ubuntu system packages"
  "${prefix[@]}" apt-get update
  "${prefix[@]}" apt-get install -y python3 python3-venv python3-pip
}

setup_python_env() {
  maybe_install_system_packages

  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python binary not found: ${PYTHON_BIN}" >&2
    exit 1
  fi

  if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating virtualenv at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi

  VENV_PYTHON="${VENV_DIR}/bin/python"
  if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "Virtualenv python not found: ${VENV_PYTHON}" >&2
    exit 1
  fi

  log "Upgrading pip/setuptools/wheel"
  "${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

  local torch_index_url
  torch_index_url="$(resolve_torch_index_url)"
  log "Installing torch/torchvision from ${torch_index_url}"
  "${VENV_PYTHON}" -m pip install torch torchvision --index-url "${torch_index_url}"

  local tmp_requirements
  tmp_requirements="$(mktemp)"
  grep -viE '^[[:space:]]*(torch|torchvision)[[:space:]]*$' "${ROOT_DIR}/requirements.txt" > "${tmp_requirements}"
  log "Installing project requirements"
  "${VENV_PYTHON}" -m pip install -r "${tmp_requirements}"
  rm -f "${tmp_requirements}"

  log "Installed versions"
  "${VENV_PYTHON}" - <<'PY'
import sys
import torch
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("torch_cuda:", torch.version.cuda)
PY
}

resolve_mask_dir_for_black_rot() {
  if [[ -n "${MASK_DIR}" ]]; then
    echo "${MASK_DIR}"
    return
  fi

  local runs_dir="${DATA_ROOT}/runs"
  if [[ ! -d "${runs_dir}" ]]; then
    echo ""
    return
  fi

  local latest=""
  while IFS= read -r dir; do
    latest="${dir}"
    break
  done < <(find "${runs_dir}" -mindepth 2 -maxdepth 2 -type d -name lesion_mask | sort -r)

  echo "${latest}"
}

calc_total_itrs_from_epochs() {
  local resolved_mask_dir="$1"
  "${VENV_PYTHON}" - <<PY
from pathlib import Path

data_root = Path(r"${DATA_ROOT}")
image_dir = Path(r"${IMAGE_DIR}")
if not image_dir.is_absolute():
    image_dir = data_root / image_dir
mask_dir = Path(r"${resolved_mask_dir}") if r"${resolved_mask_dir}" else None
val_percent = float("${VAL_PERCENT}")
batch_size = int("${BATCH_SIZE}")
epochs = int("${EPOCHS}")

suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in suffixes])
paired = 0
for image in images:
    stem = image.stem
    candidates = []
    if mask_dir is not None:
        candidates = [
            mask_dir / f"{stem}_lesion_mask.png",
            mask_dir / f"{stem}.png",
            mask_dir / f"{stem}.jpg",
        ]
    if any(c.is_file() for c in candidates):
        paired += 1

if paired <= 0:
    raise SystemExit("No paired training samples found")

val_count = 0
if paired > 1 and val_percent > 0:
    val_count = int(round(paired * val_percent))
    val_count = max(1, min(paired - 1, val_count))

train_count = paired - val_count
steps_per_epoch = max(1, train_count // batch_size)
print(steps_per_epoch * epochs)
PY
}

build_train_command() {
  if [[ -z "${TOTAL_ITRS}" && -z "${EPOCHS}" ]]; then
    TOTAL_ITRS="5000"
  fi

  local resolved_mask_dir=""
  if [[ "${DATASET}" == "black_rot" ]]; then
    resolved_mask_dir="$(resolve_mask_dir_for_black_rot)"
    if [[ -z "${resolved_mask_dir}" ]]; then
      echo "Unable to find black_rot mask dir. Set --mask-dir explicitly." >&2
      exit 1
    fi
    if [[ -n "${EPOCHS}" && -z "${TOTAL_ITRS}" ]]; then
      TOTAL_ITRS="$(calc_total_itrs_from_epochs "${resolved_mask_dir}")"
      log "Resolved --epochs ${EPOCHS} to --total_itrs ${TOTAL_ITRS}"
    fi
  fi

  CMD=(
    "${VENV_PYTHON}" "${ROOT_DIR}/main.py"
    "--dataset" "${DATASET}"
    "--data_root" "${DATA_ROOT}"
    "--model" "${MODEL}"
    "--gpu_id" "${GPU_ID}"
    "--crop_size" "${CROP_SIZE}"
    "--batch_size" "${BATCH_SIZE}"
    "--val_batch_size" "${VAL_BATCH_SIZE}"
    "--num_workers" "${NUM_WORKERS}"
    "--val_num_workers" "${VAL_NUM_WORKERS}"
    "--output_stride" "${OUTPUT_STRIDE}"
    "--lr" "${LR}"
    "--loss_type" "${LOSS_TYPE}"
    "--total_itrs" "${TOTAL_ITRS}"
  )

  if [[ "${DATASET}" == "black_rot" ]]; then
    CMD+=(
      "--black_rot_image_dir" "${IMAGE_DIR}"
      "--black_rot_mask_dir" "${resolved_mask_dir}"
      "--val_percent" "${VAL_PERCENT}"
    )
  fi

  if [[ -n "${CKPT}" ]]; then
    CMD+=("--ckpt" "${CKPT}")
  fi
  if [[ "${SEPARABLE_CONV}" == "1" ]]; then
    CMD+=("--separable_conv")
  fi
  if [[ "${PRETRAINED_BACKBONE}" == "1" ]]; then
    CMD+=("--pretrained_backbone")
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi
}

build_predict_command() {
  if [[ -z "${CKPT}" ]]; then
    echo "predict mode requires --ckpt" >&2
    exit 1
  fi

  CMD=(
    "${VENV_PYTHON}" "${ROOT_DIR}/predict.py"
    "--dataset" "${DATASET}"
    "--input" "${INPUT_PATH}"
    "--model" "${MODEL}"
    "--gpu_id" "${GPU_ID}"
    "--crop_size" "${CROP_SIZE}"
    "--output_stride" "${OUTPUT_STRIDE}"
    "--ckpt" "${CKPT}"
    "--save_val_results_to" "${SAVE_COLOR_DIR}"
    "--save_pred_mask_to" "${SAVE_MASK_DIR}"
  )

  if [[ "${PRETRAINED_BACKBONE}" == "1" ]]; then
    CMD+=("--pretrained_backbone")
  fi
  if [[ "${SEPARABLE_CONV}" == "1" ]]; then
    CMD+=("--separable_conv")
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi
}

main() {
  require_mode
  parse_args "$@"
  setup_python_env

  if [[ "${MODE}" == "setup" || "${SETUP_ONLY}" == "1" ]]; then
    log "Setup completed"
    exit 0
  fi

  local -a CMD=()
  if [[ "${MODE}" == "train" ]]; then
    build_train_command
  else
    build_predict_command
  fi

  log "Running command:"
  printf '  %q' "${CMD[@]}"
  printf '\n'
  "${CMD[@]}"
}

main "$@"
