#!/usr/bin/env bash
# sensitivity_runner.sh
# Runs sensitivity.py for all (model x task) combinations.
#
# Usage:
#   ./sensitivity_runner.sh convnextv2                           # all 4 datasets
#   ./sensitivity_runner.sh convnextv2 imagenet100               # one dataset
#   ./sensitivity_runner.sh convnextv2 imagenet100 isic2019      # two datasets
#   ./sensitivity_runner.sh mit isic2018 voc2012_sbd             # two datasets
#   ./sensitivity_runner.sh all                                  # all models, all datasets

set -euo pipefail

#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#cd "$SCRIPT_DIR/.."
#source .venv/bin/activate
#cd analysis/

ROOT="/path/to/weight-watcher/inference_engine/data"
C="$ROOT/checkpoints"
P="$ROOT/projections"

# ── Argument parsing ──────────────────────────────────────────────────────────
MODEL_FILTER="${1:-all}"
shift || true
DATASET_FILTERS=("$@")   # remaining args are dataset filters (empty = all)

# ── Dataset definitions ───────────────────────────────────────────────────────
declare -A DS_CLASSES=(
    [imagenet100]=100
    [isic2019]=8
    [isic2018]=2
    [voc2012_sbd]=21
)
declare -A DS_IGNORE=(
    [imagenet100]=-100
    [isic2019]=-100
    [isic2018]=-100
    [voc2012_sbd]=255
)
declare -A DS_TASK=(
    [imagenet100]=classification
    [isic2019]=classification
    [isic2018]=segmentation
    [voc2012_sbd]=segmentation
)
declare -A DS_FOLDER=(
    [imagenet100]=imagenet100
    [isic2019]=isic2019
    [isic2018]=isic2018_task1
    [voc2012_sbd]=voc2012_sbd
)
declare -A DS_TRANSFORM=(
    [imagenet100]=imagenet100
    [isic2019]=isic2019
    [isic2018]=isic2018
    [voc2012_sbd]=voc2012
)
declare -A DS_PROJ_SUFFIX=(
    [imagenet100]=classification
    [isic2019]=classification
    [isic2018]=segmentation
    [voc2012_sbd]=segmentation
)
declare -A DS_CACHE_FP16=(
    [imagenet100]=0
    [isic2019]=0
    [isic2018]=1
    [voc2012_sbd]=1
)
declare -A DS_BATCH_SIZE=(
    [imagenet100]=100
    [isic2019]=100
    [isic2018]=10
    [voc2012_sbd]=20
)

declare -A DS_ANCHOR_SIZE=(
    [imagenet100]=100
    [isic2019]=100
    [isic2018]=10
    [voc2012_sbd]=20
)

ALL_DATASETS=(imagenet100 isic2019 isic2018 voc2012_sbd)

# ── Model definitions ─────────────────────────────────────────────────────────
declare -A MODEL_VARIANT=(
    [convnextv2]=base
    [cyclemlp]=b2
    [mit]=b2
)
ALL_MODELS=(convnextv2 cyclemlp mit)

# ── Filter helpers ────────────────────────────────────────────────────────────
should_run_dataset() {
    local ds="$1"
    if [[ ${#DATASET_FILTERS[@]} -eq 0 ]]; then
        return 0
    fi
    for f in "${DATASET_FILTERS[@]}"; do
        [[ "$ds" == *"$f"* ]] && return 0
    done
    return 1
}

should_run_model() {
    local model="$1"
    [[ "$MODEL_FILTER" == "all" ]] && return 0
    [[ "$model" == *"$MODEL_FILTER"* ]] && return 0
    return 1
}

# ── Core runner ───────────────────────────────────────────────────────────────
run_sensitivity() {
    local model="$1"
    local dataset="$2"

    local variant="${MODEL_VARIANT[$model]}"
    local num_classes="${DS_CLASSES[$dataset]}"
    local ignore_index="${DS_IGNORE[$dataset]}"
    local task="${DS_TASK[$dataset]}"
    local data_folder="${DS_FOLDER[$dataset]}"
    local transform="${DS_TRANSFORM[$dataset]}"
    local proj_suffix="${DS_PROJ_SUFFIX[$dataset]}"
    local cache_fp16="${DS_CACHE_FP16[$dataset]}"
    local batch_size="${DS_BATCH_SIZE[$dataset]}"
    local anchor_size="${DS_ANCHOR_SIZE[$dataset]}"

    local folder="${task}_${dataset}_${model}_${variant}"
    local proj_name="${model}_${variant}_${proj_suffix}"
    local proj_root="$P/${folder}_pca"
    local ckpt="$C/${folder}/best/best.pth"
    local decomp="$proj_root/${proj_name}_decomposers.pt"
    local proj="$proj_root/${proj_name}_projected.pt"
    local out_dir="./results/sensitivity/${folder}"

    for f in "$ckpt" "$decomp" "$proj"; do
        if [ ! -f "$f" ]; then
            echo "  [SKIP] Missing: $f"
            return
        fi
    done

    # Determine if we should pass the fp16 flag
    CACHE_FLAG=""
    if [ "$cache_fp16" -eq 1 ]; then
        CACHE_FLAG="--cache_fp16"
    fi

    echo "========================================================"
    echo "  Model   : $model $variant"
    echo "  Task    : $task / $dataset"
    echo "  Output  : $out_dir"
    echo "========================================================"

    mkdir -p "$out_dir"

    python -u sensitivity.py \
        --backbone          "$model" \
        --variant           "$variant" \
        --num_classes       "$num_classes" \
        --ignore_index      "$ignore_index" \
        --task              "$task" \
        --dataset           "$dataset" \
        --val_transforms    "../configs/transform/${transform}_val.yaml" \
        --data_root         "$ROOT/$data_folder" \
        --checkpoint        "$ckpt" \
        --decomposers       "$decomp" \
        --projection        "$proj" \
        --epsilon           0.25 \
        --n_directions      10 \
        --anchor_size       "$anchor_size" \
        --batch_size        "$batch_size" \
        --num_workers       4 \
        $CACHE_FLAG \
        --device            cuda:0 \
        --out_dir           "$out_dir"

    echo "  Done: $folder"
    echo ""
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for model in "${ALL_MODELS[@]}"; do
    should_run_model "$model" || continue
    for dataset in "${ALL_DATASETS[@]}"; do
        should_run_dataset "$dataset" || continue
        run_sensitivity "$model" "$dataset"
    done
done

echo "All sensitivity runs complete."
