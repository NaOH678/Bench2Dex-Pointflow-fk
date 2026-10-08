#!/usr/bin/env bash
# Two local processes, one GPU: model in its own environment, Isaac in env_isaaclab.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
: "${COSMOS_PYTHON:?Set COSMOS_PYTHON to the Python executable of your Cosmos inference environment}"
: "${COSMOS_CONFIG:?Set COSMOS_CONFIG to your completed local policy YAML}"
: "${BASELINE_ANCHOR_HDF5:?Set BASELINE_ANCHOR_HDF5 to the RGB HDF5 matching the baseline training appearance}"
run_dir="${COSMOS_RUN_DIR:-outputs/cosmos_local/run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$run_dir"
export COSMOS_PREDICTION_DIR="${COSMOS_PREDICTION_DIR:-$(realpath "$run_dir")/predictions}"
# A benchmark can require complete asset pools before entering selected channels.
asset_gate="${COSMOS_ASSET_GATE:-$run_dir/../../asset_gate.json}"
if [[ -f "$asset_gate" ]]; then
  "$COSMOS_PYTHON" - "$asset_gate" "$@" <<'PY'
import json, sys, time
from pathlib import Path
gate = json.loads(Path(sys.argv[1]).read_text())
args = sys.argv[2:]
profile = 'none'
for i, arg in enumerate(args):
    if arg == '--generalization-profile':
        profile = args[i + 1]
if profile in gate['channels']:
    while not Path(gate['ready_file']).is_file():
        if Path(gate['failure_file']).is_file():
            raise RuntimeError('Benchmark asset preparation failed; inspect assets.log')
        print('Waiting for verified benchmark assets:', gate['ready_file'], flush=True)
        time.sleep(30)
PY
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Geometry consists of small matrices; avoid creating a BLAS worker pool per
# process. Explicit environment values still override these defaults.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export DEX2BENCH_RECORD_RES="${DEX2BENCH_RECORD_RES:-640}"
export DEX2BENCH_RECORD_JPEG="${DEX2BENCH_RECORD_JPEG:-90}"
export DEX2BENCH_RECORD_STRIDE="${DEX2BENCH_RECORD_STRIDE:-1}"
# Do not inherit FK injection from an earlier point-FK training shell.
unset FK_ENCODER_CHECKPOINT
"$COSMOS_PYTHON" - <<'PY'
import socket
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', 9000))  # Refuse to connect the simulator to an unrelated existing service.
PY
"$COSMOS_PYTHON" -m script.policy_model_server --config "$COSMOS_CONFIG" --host 127.0.0.1 --port 9000 \
  > "$run_dir/model.log" 2>&1 &
model_pid=$!
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,nounits -l 1 > "$run_dir/gpu.csv" &
monitor_pid=$!
trap 'kill "$model_pid" "$monitor_pid" 2>/dev/null || true; wait "$model_pid" "$monitor_pid" 2>/dev/null || true' EXIT
"$COSMOS_PYTHON" - "$model_pid" <<'PY'
import os, socket, sys, time
from pathlib import Path
pid = int(sys.argv[1])
deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    os.kill(pid, 0)
    stat = Path(f'/proc/{pid}/stat')
    if stat.exists() and stat.read_text().split(') ', 1)[1].startswith('Z'):
        raise RuntimeError('Local model process exited; inspect model.log')
    try:
        with socket.create_connection(('127.0.0.1', 9000), timeout=1):
            break
    except OSError:
        time.sleep(1)
else:
    raise TimeoutError('Local model startup timed out; inspect model.log')
PY
bash tools/run_isaaclab.sh run_policy.py \
  --policy-type REMOTE --policy-name Cosmos --remote-host 127.0.0.1 --remote-port 9000 \
  --remote-timeout-s 600 --task scenes/21_condiment_box_loading.yaml \
  --robot-key multi_ur5_wuji_with_flange --active-dof \
  --cameras cam_overhead cam_wrist_left cam_wrist_right \
  --collect-config configs/collect/default.yaml --enable-rgb --headless --enable_cameras \
  --enable-generalization --generalization-profile none --anchor-hdf5 "$BASELINE_ANCHOR_HDF5" \
  --num-episodes 1 --episode-steps 1000 --record-all \
  --record-dir "$run_dir/episodes" --output-dir "$run_dir/metrics" "$@" \
  > "$run_dir/sim.log" 2>&1
