#!/usr/bin/env bash
# Tests the redundancy prediction: pruning HIGH-displacement units in projection
# layers should hurt LESS than random, while pruning ISOLATED (high-silhouette)
# units should hurt MORE.
#
# Usage (from inference_engine/):
#   ./analysis/pruning_runner.sh                      # all models, all datasets
#   ./analysis/pruning_runner.sh cyclemlp             # one model
#   ./analysis/pruning_runner.sh mit imagenet100      # one cell
#
# Output: analysis/results/pruning/pruning_<model>_<task>_<dataset>_<lt>.csv

set -euo pipefail

ROOT="/path/to/weight-watcher/inference_engine/data"
INFERENCE_ENGINE="/home/hudo/Documents/weight-watcher/inference_engine"

declare -A MODEL_VARIANT=( [convnextv2]=base [cyclemlp]=b2 [mit]=b2 )
declare -A DS_TASK=(
    [imagenet100]=classification [isic2019]=classification
    [isic2018]=segmentation      [voc2012_sbd]=segmentation
)
declare -A DS_CLASSES=( [imagenet100]=100 [isic2019]=8 [isic2018]=2 [voc2012_sbd]=21 )
declare -A DS_IGNORE=(  [imagenet100]=-100 [isic2019]=-100 [isic2018]=-100 [voc2012_sbd]=255 )
declare -A DS_FOLDER=(  [imagenet100]=imagenet100 [isic2019]=isic2019
                        [isic2018]=isic2018_task1 [voc2012_sbd]=voc2012_sbd )
declare -A DS_TRANSFORM=( [imagenet100]=imagenet100 [isic2019]=isic2019
                          [isic2018]=isic2018       [voc2012_sbd]=voc2012 )
declare -A DS_EVAL_SIZE=( [imagenet100]=500 [isic2019]=500 [isic2018]=200 [voc2012_sbd]=200 )
declare -A DS_BATCH=(     [imagenet100]=16  [isic2019]=16  [isic2018]=4   [voc2012_sbd]=8 )

TRAJ_CSV="analysis/results/trajectory_geometry/trajectory_geometry.csv"
CLUSTER_CSV="analysis/results/cluster_analysis/cluster_analysis.csv"
SENS_ROOT="analysis/results/sensitivity"
PRUNE_RATIOS="0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.10 0.15 0.20 0.50"

# Layer-type groups to test separately. The contrast is the point:
# projection layers should show the redundancy effect, conv layers should not.
LAYER_GROUPS=("linear" "conv2d" "attention_query" "attention_output" "depthwise_conv")

ALL_MODELS=(convnextv2 cyclemlp mit)
ALL_DATASETS=(imagenet100 isic2019 isic2018 voc2012_sbd)

MODEL_FILTER="${1:-}"
DATASET_FILTER="${2:-}"
OUT_DIR="analysis/results/pruning_extended"

[[ -f "$TRAJ_CSV"    ]] || { echo "ERROR: missing $TRAJ_CSV";    exit 1; }
[[ -f "$CLUSTER_CSV" ]] || { echo "ERROR: missing $CLUSTER_CSV"; exit 1; }

for model in "${ALL_MODELS[@]}"; do
    [[ -n "$MODEL_FILTER" && "$model" != *"$MODEL_FILTER"* ]] && continue
    variant="${MODEL_VARIANT[$model]}"

    for dataset in "${ALL_DATASETS[@]}"; do
        [[ -n "$DATASET_FILTER" && "$dataset" != *"$DATASET_FILTER"* ]] && continue

        task="${DS_TASK[$dataset]}"
        folder="${task}_${dataset}_${model}_${variant}"
        checkpoint="${ROOT}/checkpoints/${folder}/best/best.pth"
        proj_root="${ROOT}/projections/${folder}_pca"
        decomposers="${proj_root}/${model}_${variant}_${task}_decomposers.pt"
        projection="${proj_root}/${model}_${variant}_${task}_projected.pt"
        sens_csv="${SENS_ROOT}/${folder}/sensitivity.csv"

        [[ -f "$sens_csv" ]] || { echo "[skip] $sens_csv" - no sensitivity.csv; continue; }
        [[ -f "$checkpoint" ]] || { echo "[skip] $folder - no checkpoint";  continue; }
        [[ -f "$decomposers" ]] || { echo "[skip] $folder - no decomposers"; continue; }

        SENS_ARG=""
        [[ -f "$sens_csv" ]] && SENS_ARG="--sensitivity_csv $sens_csv"

        IGNORE_ARG=""
        [[ "$task" == "segmentation" ]] && IGNORE_ARG="--ignore_index ${DS_IGNORE[$dataset]}"

        for lt in "${LAYER_GROUPS[@]}"; do
            echo "========================================================"
            echo "  Pruning: $model $variant | $task | $dataset | layers=$lt"
            echo "========================================================"

            python analysis/pruning_experiment.py "$task" \
                --backbone         "$model" \
                --variant          "$variant" \
                --num_classes      "${DS_CLASSES[$dataset]}" \
                --dataset          "$dataset" \
                --data_root        "${ROOT}/${DS_FOLDER[$dataset]}" \
                --checkpoint_path  "$checkpoint" \
                --decomposers_path "$decomposers" \
                --projection_path  "$projection" \
                --val_transforms   "${INFERENCE_ENGINE}/configs/transform/${DS_TRANSFORM[$dataset]}_val.yaml" \
                --trajectory_csv   "$TRAJ_CSV" \
                --cluster_csv      "$CLUSTER_CSV" \
                --layer_types      "$lt" \
                --prune_ratios     $PRUNE_RATIOS \
                --n_seeds          3 \
                --eval_size        "${DS_EVAL_SIZE[$dataset]}" \
                --batch_size       "${DS_BATCH[$dataset]}" \
                --device           cuda:0 \
                --out_dir          "$OUT_DIR" \
                $SENS_ARG $IGNORE_ARG

            echo "  Done: $folder ($lt)"
            echo ""
        done
    done
done

# Concatenate everything into one table
echo "Concatenating pruning results..."
python - <<'PYEOF'
import glob, os, pandas as pd
out = "analysis/results/pruning"
files = [f for f in sorted(glob.glob(os.path.join(out, "pruning_*.csv")))
         if not f.endswith("pruning_ALL.csv")]
if files:
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    path = os.path.join(out, "pruning_ALL.csv")
    df.to_csv(path, index=False)
    print(f"  Wrote {path} ({len(df)} rows)")
    piv = df.pivot_table(index=["model","dataset","layer_types","prune_ratio"],
                         columns="condition", values="drop_from_baseline")
    print("\nMetric drop by condition (expect high_disp < random < high_sil "
          "for projection layers):")
    print(piv.to_string(float_format=lambda x: f"{x:+.4f}"))
else:
    print("  No pruning result files found.")
PYEOF

echo "All pruning runs complete."
