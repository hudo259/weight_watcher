#!/usr/bin/env bash
# intervention_runner.sh
# Quantified intervention experiment for the rebuttal.
#
# Usage:
#   ./intervention_runner.sh convnextv2 imagenet100
#   ./intervention_runner.sh mit voc2012_sbd
#
# Runs the geometric / random / residual perturbation comparison and writes
# results/intervention/<folder>/intervention_results.csv
#
# Run this from inference_engine/ (same working dir as main.py).

set -euo pipefail

ROOT="/path/to/weight-watcher/inference_engine/data"
INFERENCE_ENGINE="/path/to/weight-watcher/inference_engine"

declare -A MODEL_VARIANT=( [convnextv2]=base [cyclemlp]=b2 [mit]=b2 )
declare -A DS_TASK=(
    [imagenet100]=classification [isic2019]=classification
    [isic2018]=segmentation      [voc2012_sbd]=segmentation
)
declare -A DS_CLASSES=(
    [imagenet100]=100 [isic2019]=8 [isic2018]=2 [voc2012_sbd]=21
)
declare -A DS_IGNORE=(
    [imagenet100]=-100 [isic2019]=-100 [isic2018]=-100 [voc2012_sbd]=255
)
declare -A DS_FOLDER=(
    [imagenet100]=imagenet100 [isic2019]=isic2019
    [isic2018]=isic2018_task1  [voc2012_sbd]=voc2012_sbd
)
declare -A DS_TRANSFORM=(
    [imagenet100]=imagenet100 [isic2019]=isic2019
    [isic2018]=isic2018       [voc2012_sbd]=voc2012
)
# Eval subset per dataset (segmentation is heavier per image)
declare -A DS_EVAL_SIZE=(
    [imagenet100]=500 [isic2019]=500 [isic2018]=200 [voc2012_sbd]=200
)
declare -A DS_BATCH=(
    [imagenet100]=16 [isic2019]=16 [isic2018]=4 [voc2012_sbd]=8
)

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <model> <dataset>"
    echo "  model:   convnextv2 | cyclemlp | mit"
    echo "  dataset: imagenet100 | isic2019 | isic2018 | voc2012_sbd"
    exit 1
fi

MODEL="$1"; DATASET="$2"
VARIANT="${MODEL_VARIANT[$MODEL]}"
TASK="${DS_TASK[$DATASET]}"
FOLDER="${TASK}_${DATASET}_${MODEL}_${VARIANT}"
PROJ_NAME="${MODEL}_${VARIANT}_${TASK}"
PROJ_ROOT="${ROOT}/projections/${FOLDER}_pca"

CHECKPOINT="${ROOT}/checkpoints/${FOLDER}/best/best.pth"
DECOMPOSERS="${PROJ_ROOT}/${PROJ_NAME}_decomposers.pt"
PROJECTION="${PROJ_ROOT}/${PROJ_NAME}_projected.pt"
DATA_ROOT="${ROOT}/${DS_FOLDER[$DATASET]}"
SENS_CSV="${INFERENCE_ENGINE}/analysis/results/sensitivity/${FOLDER}/sensitivity.csv"
OUT_DIR="${INFERENCE_ENGINE}/results/intervention/${FOLDER}"

IGNORE_ARG=""
[[ "$TASK" == "segmentation" ]] && IGNORE_ARG="--ignore_index ${DS_IGNORE[$DATASET]}"

SENS_ARG=""
[[ -f "$SENS_CSV" ]] && SENS_ARG="--sensitivity_csv $SENS_CSV"

echo "========================================================"
echo "  Intervention: $MODEL $VARIANT | $TASK | $DATASET"
echo "========================================================"

python analysis/intervention_experiment.py "$TASK" \
    --backbone         "$MODEL" \
    --variant          "$VARIANT" \
    --num_classes      "${DS_CLASSES[$DATASET]}" \
    --dataset          "$DATASET" \
    --data_root        "$DATA_ROOT" \
    --checkpoint_path  "$CHECKPOINT" \
    --decomposers_path "$DECOMPOSERS" \
    --projection_path  "$PROJECTION" \
    --val_transforms   "${INFERENCE_ENGINE}/configs/transform/${DS_TRANSFORM[$DATASET]}_val.yaml" \
    --eval_size        "${DS_EVAL_SIZE[$DATASET]}" \
    --batch_size       "${DS_BATCH[$DATASET]}" \
    --top_q            0.10 \
    --epsilons         1 2 4 \
    --n_seeds          3 \
    --device           cuda:0 \
    --out_dir          "$OUT_DIR" \
    $SENS_ARG \
    $IGNORE_ARG

echo "  Done: $OUT_DIR/intervention_results.csv"
