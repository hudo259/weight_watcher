#!/usr/bin/env bash
#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#cd "$SCRIPT_DIR"
#source .venv/bin/activate

ROOT="/path/to/weight-watcher/inference_engine/data"
DATASETS_NUMCLASSES_IGNOREINDEX_TASKS=(
  "imagenet100 100 -100 classification"
  "voc2012_sbd 21 255 segmentation"
  "isic2019 8 -100 classification"
  "isic2018 2 -100 segmentation"
)
MODELS=(
  "convnextv2 base"
  "cyclemlp b2"
  "mit b2"
)
OUTPUT_ROOT="./results/method_consistency"

run_method() {
  read -a strarr <<< "$1"
  local dataset=${strarr[0]}
  local task=${strarr[3]}
  read -a strarr <<< "$2"
  local model=${strarr[0]}
  local variant=${strarr[1]}
  local folder_name="${task}_${dataset}_${model}_${variant}"
  local proj_name="${model}_${variant}_${task}"
  local proj_root="${ROOT}/projections/${folder_name}"

  echo "========================================================"
  echo "  Model   : $model $variant"
  echo "  Task    : $task"
  echo "  Dataset : ${dataset}"
  echo "  Output  : ${OUTPUT_ROOT}/${folder_name}"
  echo "========================================================"

  local run_str="${model}_${variant},${task}"
  run_str+=",${proj_root}_pca/${proj_name}_projected.pt"
  run_str+=",${proj_root}_fourier_kpca/${proj_name}_projected.pt"
  run_str+=",adaptive:${proj_root}_umap/${proj_name}_projected.pt:${proj_root}_umap/${proj_name}_decomposers.pt"

  python3 method_consistency.py \
    --runs "$run_str" \
    --out_dir "${OUTPUT_ROOT}/${folder_name}"
}

for dnit in "${DATASETS_NUMCLASSES_IGNOREINDEX_TASKS[@]}"; do
  for model in "${MODELS[@]}"; do
    run_method "$dnit" "$model"
  done
done

echo "Finished."
