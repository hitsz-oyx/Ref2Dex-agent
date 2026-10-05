#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd "$(dirname "$0")/../../../../.." && pwd)
run_id=$1
env_count=$2
waves=$3
assignment_seed=$4
simulator_seed=$5
budget=$6
decision_region=${7:-contact}
duration_spec=${8:-4}
read -r -a duration_levels <<< "$duration_spec"
amplitude_spec=${9:-1}
read -r -a amplitude_levels <<< "$amplitude_spec"
intervention_set=${10:-synergy}
finger_range_fraction=${11:-0.05}
mkdir -p "$task_root/tmp/ref3" "$task_root/tmp/torch_extensions"
export TMPDIR="$task_root/tmp"
export TORCH_EXTENSIONS_DIR="$task_root/tmp/torch_extensions"
export MAX_JOBS=2 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=6
export LD_LIBRARY_PATH="/home2/wyy/miniconda3/envs/graspenv/lib:${LD_LIBRARY_PATH:-}"
cd "$task_root/third_party/DExplore"
timeout "$((budget+90))s" /home2/wyy/miniconda3/envs/graspenv/bin/python \
  "$task_root/src/task/cm-interaction-oracle/tools/run/collect_interventions.py" \
  --run-dir "$task_root/outputs/cm-interaction-oracle/$run_id" \
  --assignment-seed "$assignment_seed" --waves "$waves" --wall-seconds "$budget" \
  --decision-region "$decision_region" --durations "${duration_levels[@]}" --amplitudes "${amplitude_levels[@]}" \
  --intervention-set "$intervention_set" --finger-range-fraction "$finger_range_fraction" \
  --max-steps 2000 --task Dexplore_Inspire \
  --cfg_env "$task_root/src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7/environment.yaml" \
  --cfg_train "$task_root/src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7/training.yaml" \
  --checkpoint /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-baseline/outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000260.pth \
  --motion_file /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-baseline/outputs/CmResidual/agent_contact_option_airplane_motions \
  --headless --num_envs "$env_count" --seed "$simulator_seed" \
  --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 --pipeline gpu \
  --output "$task_root/outputs/cm-interaction-oracle/$run_id/native_eval.json" \
  --output_path "$task_root/outputs/cm-interaction-oracle/$run_id/native" \
  > "$task_root/tmp/ref3/$run_id.log" 2>&1
