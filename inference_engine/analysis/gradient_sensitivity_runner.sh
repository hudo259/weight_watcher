#!/usr/bin/env bash
set -euo pipefail

ROOT="/path/to/weight-watcher/inference_engine/data"
INFERENCE_ENGINE="/path/to/weight-watcher/inference_engine"

# Model definitions
declare -A MODEL_VARIANT=(
    [convnextv2]=base
    [cyclemlp]=b2
    [mit]=b2
)

# Dataset definitions
declare -A DS_TASK=(
    [imagenet100]=classification
    [isic2019]=classification
    [isic2018]=segmentation
    [voc2012_sbd]=segmentation
)
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
declare -A DS_ANCHOR_SIZE=(
    [imagenet100]=500
    [isic2019]=500
    [isic2018]=100
    [voc2012_sbd]=200
)
declare -A DS_BATCH_SIZE=(
    [imagenet100]=16
    [isic2019]=16
    [isic2018]=4
    [voc2012_sbd]=8
)
declare -A DS_LOSS=(
    [imagenet100]=cross_entropy_classification
    [isic2019]=cross_entropy_classification
    [isic2018]=dice_bce
    [voc2012_sbd]=cross_entropy_segmentation
)


ALL_MODELS=(convnextv2 cyclemlp mit)
ALL_DATASETS=(imagenet100 isic2019 isic2018 voc2012_sbd)

MODEL_FILTER="${1:-}"
DATASET_FILTER="${2:-}"

for model in "${ALL_MODELS[@]}"; do
    [[ -n "$MODEL_FILTER" && "$model" != *"$MODEL_FILTER"* ]] && continue
    variant="${MODEL_VARIANT[$model]}"

    for dataset in "${ALL_DATASETS[@]}"; do
        [[ -n "$DATASET_FILTER" && "$dataset" != *"$DATASET_FILTER"* ]] && continue

        task="${DS_TASK[$dataset]}"
        folder="${task}_${dataset}_${model}_${variant}"
        out_dir="results/sensitivity/${folder}"

        echo "========================================================"
        echo "  Model: $model $variant | Task: $task | Dataset: $dataset"
        echo "========================================================"

        IGNORE_ARG=""
        [[ "$task" == "segmentation" ]] && IGNORE_ARG="--ignore_index ${DS_IGNORE[$dataset]}"


        python gradient_sensitivity.py "$task" \
            --backbone      "$model" \
            --variant       "$variant" \
            --num_classes   "${DS_CLASSES[$dataset]}" \
            --checkpoint    "${ROOT}/checkpoints/${folder}/best/best.pth" \
            --decomposers   "${ROOT}/projections/${folder}_pca/${model}_${variant}_${task}_decomposers.pt" \
            --projection    "${ROOT}/projections/${folder}_pca/${model}_${variant}_${task}_projected.pt" \
            --dataset       "$dataset" \
            --data_root     "${ROOT}/${DS_FOLDER[$dataset]}" \
            --val_transforms "${INFERENCE_ENGINE}/configs/transform/${DS_TRANSFORM[$dataset]}_val.yaml" \
            --anchor_size   "${DS_ANCHOR_SIZE[$dataset]}" \
            --batch_size    "${DS_BATCH_SIZE[$dataset]}" \
            --num_workers   4 \
            --device        cuda:0 \
            --seed          42 \
            --loss_config   "${INFERENCE_ENGINE}/configs/loss/${DS_LOSS[$dataset]}.yaml" \
            --model_name    "${model}_${variant}" \
            --out_dir       "$out_dir" \
            $IGNORE_ARG

        echo "  Done: $folder"
        echo ""
    done
done

echo "All gradient sensitivity runs complete."
