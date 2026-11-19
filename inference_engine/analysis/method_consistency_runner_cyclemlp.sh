#!/usr/bin/env bash

# Activate venv
#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#cd "$SCRIPT_DIR"
#source .venv/bin/activate

# Base paths
ROOT="/path/to/weight-watcher/inference_engine/data"
DATASETS_NUMCLASSES_IGNOREINDEX_TASKS=(
  "imagenet100 100 -100 classification"
  "voc2012_sbd 21 255 segmentation"
  "isic2019 8 -100 classification"
  "isic2018 2 -100 segmentation"
)
MODELS=(
  "cyclemlp b2"
)
UMAP_RUNS=("0-1_5" "0-1_15" "0-1_25" "0-1_50")
OUTPUT_ROOT="./results/method_consistency"

run_method() {
    read -a strarr <<< "$1"
    local dataset=${strarr[0]}
    local num_classes=${strarr[1]}
    local ignore_index=${strarr[2]}
    local task=${strarr[3]}
    read -a strarr <<< "$2"
    local model=${strarr[0]}
    local variant=${strarr[1]}
    local folder_name="${task}_${dataset}_${model}_${variant}"
    local projection_folder_name="${model}_${variant}_${task}"
    local proj_root="${ROOT}/projections/${folder_name}"

    echo "========================================================"
    echo "  Model   : $model $variant"
    echo "  Task    : $task"
    echo "  Dataset : ${dataset}"
    echo "  Output  : ${OUTPUT_ROOT}/${folder_name}"
    echo "========================================================"

    # Build the run string without any embedded whitespace
    local run_str="${model}_${variant},${task}"
    run_str+=",${proj_root}_pca/${projection_folder_name}_projected.pt"
    run_str+=",${proj_root}_fourier_kpca/${projection_folder_name}_projected.pt"
    run_str+=",n5:${proj_root}_umap/0-1_5/${projection_folder_name}_projected.pt"
    run_str+=",n15:${proj_root}_umap/0-1_15/${projection_folder_name}_projected.pt"
    run_str+=",n25:${proj_root}_umap/0-1_25/${projection_folder_name}_projected.pt"
    run_str+=",n50:${proj_root}_umap/0-1_50/${projection_folder_name}_projected.pt"

    python method_consistency.py \
        --runs "$run_str" \
        --out_dir "${OUTPUT_ROOT}/${folder_name}"
}

for dnit in "${DATASETS_NUMCLASSES_IGNOREINDEX_TASKS[@]}"; do
    for model in "${MODELS[@]}"; do
        run_method "$dnit" "$model"
    done
done

echo "Finished."
