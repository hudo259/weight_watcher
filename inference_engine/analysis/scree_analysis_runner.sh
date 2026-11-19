#!/usr/bin/env bash
# scree_runner.sh — Part A of the k=3 rebuttal response.
#
# Computes the full per-layer explained-variance spectrum and reports
# components-to-90%/95%-EV per layer type. Pure re-analysis on existing
# training checkpoints — no GPU strictly required, no re-fitting.
#
# Usage (from inference_engine/):
#   ./analysis/scree_runner.sh                 # all models, all datasets
#   ./analysis/scree_runner.sh convnextv2      # one model, all datasets
#   ./analysis/scree_runner.sh mit imagenet100 # one model, one dataset
#
# Output: analysis/results/scree/scree[_summary]_<model>_<task_dataset>.csv

set -euo pipefail

ROOT="/path/to/weight-watcher/inference_engine/data"
INFERENCE_ENGINE="/home/hudo/Documents/weight-watcher/inference_engine"

declare -A MODEL_VARIANT=(
    [convnextv2]=base
    [cyclemlp]=b2
    [mit]=b2
)
# Config basename in configs/decomposition/ (NOTE: filenames use the variant)
declare -A MODEL_CONFIG=(
    [convnextv2]=convnextv2_base
    [cyclemlp]=cycle_mlp_b2
    [mit]=mit_b2
)

declare -A DS_TASK=(
    [imagenet100]=classification
    [isic2019]=classification
    [isic2018]=segmentation
    [voc2012_sbd]=segmentation
)

ALL_MODELS=(convnextv2 cyclemlp mit)
ALL_DATASETS=(imagenet100 isic2019 isic2018 voc2012_sbd)

MODEL_FILTER="${1:-}"
DATASET_FILTER="${2:-}"

OUT_DIR="analysis/results/scree"

for model in "${ALL_MODELS[@]}"; do
    [[ -n "$MODEL_FILTER" && "$model" != *"$MODEL_FILTER"* ]] && continue
    variant="${MODEL_VARIANT[$model]}"
    config_base="${MODEL_CONFIG[$model]}"

    for dataset in "${ALL_DATASETS[@]}"; do
        [[ -n "$DATASET_FILTER" && "$dataset" != *"$DATASET_FILTER"* ]] && continue

        task="${DS_TASK[$dataset]}"
        folder="${task}_${dataset}_${model}_${variant}"

        # Training-checkpoint directory that the decomposition was fit from.
        # Adjust the pattern below if your epoch checkpoints live elsewhere.
        input_dir="${ROOT}/checkpoints/${folder}/epochs"
        config="${INFERENCE_ENGINE}/configs/decomposition/${config_base}_${task}.yaml"

        if [[ ! -d "$input_dir" ]]; then
            echo "[skip] $folder — no checkpoint dir at $input_dir"
            continue
        fi
        if [[ ! -f "$config" ]]; then
            echo "[skip] $folder — no config at $config"
            continue
        fi

        echo "========================================================"
        echo "  Scree: $model $variant | $task | $dataset"
        echo "========================================================"

        python analysis/scree_analysis.py \
            -i           "$input_dir" \
            -c           "$config" \
            --model_name "${model}_${variant}" \
            --task       "${task}_${dataset}" \
            --thresholds 0.90 0.95 \
            --k_ref      3 \
            --device     cpu \
            -o           "$OUT_DIR"

        echo "  Done: $folder"
        echo ""
    done
done

# Concatenate all per-layer-type summaries into one global table 
echo "Concatenating summaries..."
python - <<'PYEOF'
import glob, os, pandas as pd
out = "analysis/results/scree"
summ = sorted(glob.glob(os.path.join(out, "scree_summary_*.csv")))
if summ:
    df = pd.concat([pd.read_csv(f) for f in summ], ignore_index=True)
    path = os.path.join(out, "scree_summary_ALL.csv")
    df.to_csv(path, index=False)
    print(f"  Wrote {path} ({len(df)} rows)")
else:
    print("  No summary files found.")
PYEOF

echo "All scree runs complete."
