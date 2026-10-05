#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../../../.." && pwd)
run_id=$1
envs=$2
candidate=$3
reference=${4:-}
args=()
if [[ -n "$reference" ]]; then args+=(--reference "$root/outputs/cm-interaction-oracle/$reference"); fi
mkdir -p "$root/tmp/ref13" "$root/tmp/torch_extensions"
export TMPDIR="$root/tmp" TORCH_EXTENSIONS_DIR="$root/tmp/torch_extensions"
export MAX_JOBS=2 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=6
export LD_LIBRARY_PATH="/home2/wyy/miniconda3/envs/graspenv/lib:${LD_LIBRARY_PATH:-}"
cd "$root/third_party/DExplore"
timeout 240s /home2/wyy/miniconda3/envs/graspenv/bin/python \
 "$root/src/task/cm-interaction-oracle/tools/run/collect_oracle_y_candidates.py" \
 --run-dir "$root/outputs/cm-interaction-oracle/$run_id" --candidate "$candidate" \
 --wall-seconds 180 "${args[@]}" --task Dexplore_Inspire \
 --cfg_env "$root/src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7/environment.yaml" \
 --cfg_train "$root/src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7/training.yaml" \
 --checkpoint /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-baseline/outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000260.pth \
 --motion_file /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-baseline/outputs/CmResidual/agent_contact_option_airplane_motions \
 --headless --num_envs "$envs" --seed 263 --sim_device cuda:0 --rl_device cuda:0 \
 --graphics_device_id 0 --pipeline gpu --output "$root/outputs/cm-interaction-oracle/$run_id/native_eval.json" \
 --output_path "$root/outputs/cm-interaction-oracle/$run_id/native" \
 > "$root/tmp/ref13/$run_id.log" 2>&1
