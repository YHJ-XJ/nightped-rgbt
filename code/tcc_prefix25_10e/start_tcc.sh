#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

BASE="$(dirname "$PWD")"
OLD="${OLD_PROJECT:-$BASE/LLVIP_YOLO26_cloud/LLVIP_YOLO26_cloud}"
DATA="${DATA_ROOT:-$OLD/data/LLVIP_prepared}"
ARCHIVE="${LLVIP_ARCHIVE:-$OLD/LLVIP_HF_McCheng.zip}"
SPLIT="${TCC_SPLIT_DIR:-$PWD/data_protocol/prefix25}"
RUNS="${TCC_RUN_ROOT:-$PWD/runs/tcc_prefix25_10e}"
DEVICE="${TCC_DEVICE:-0}"
WORKERS="${TCC_WORKERS:-8}"
ARGS=(--data-root "$DATA" --split-dir "$SPLIT" --run-root "$RUNS"
      --device "$DEVICE" --workers "$WORKERS")
SHA=9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef

verify() { printf '%s  %s\n' "$SHA" "$1" | sha256sum --check --status; }

case "${1:-prepare}" in
  prepare)
    python -m pip install -r requirements.txt
    if ! verify yolo26n.pt 2>/dev/null; then
      if verify "$OLD/yolo26n.pt" 2>/dev/null; then
        cp -- "$OLD/yolo26n.pt" yolo26n.pt.incoming
      else
        curl -fL --retry 3 -o yolo26n.pt.incoming \
          https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
      fi
      printf '%s  %s\n' "$SHA" yolo26n.pt.incoming | sha256sum --check
      mv -b -- yolo26n.pt.incoming yolo26n.pt
    fi
    python prepare_data.py setup --data-root "$DATA" --archive "$ARCHIVE" --workers "$WORKERS"
    if [[ ! -d "$SPLIT" ]]; then
      python prepare_data.py probe --data-root "$DATA" --exclude-prefix 25 --out "$SPLIT"
    fi
    python run_tcc.py quality "${ARGS[@]}"
    ;;
  train)
    shift
    mkdir -p "$RUNS"
    (
      python -c 'import sys, torch; i=int(sys.argv[1]); print(torch.cuda.get_device_name(i)); print(torch.ones(1, device=f"cuda:{i}"))' "$DEVICE"
      python -u run_tcc.py train "${ARGS[@]}" "$@"
      python -u run_tcc.py report "${ARGS[@]}" "$@"
    ) 2>&1 | tee -a "$RUNS/console.log"
    ;;
  report)
    shift
    python -u run_tcc.py report "${ARGS[@]}" "$@"
    ;;
  *)
    echo "Usage: bash start_tcc.sh {prepare|train|report}" >&2
    exit 2
    ;;
esac
