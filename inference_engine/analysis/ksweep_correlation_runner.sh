#!/usr/bin/env bash
# Usage (from inference_engine/):
#   ./analysis/ksweep_runner.sh                  # all models, all datasets
#   ./analysis/ksweep_runner.sh cyclemlp         # one model, all datasets
#   ./analysis/ksweep_runner.sh mit imagenet100  # one model, one dataset
set -euo pipefail

ROOT="/path/to/weight-watcher/inference_engine/data"
INFERENCE_ENGINE="/path/to/weight-watcher/inference_engine"

declare -A MODEL_VARIANT=(
    [convnextv2]=base
    [cyclemlp]=b2
    [mit]=b2
)
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

# Trajectory CSV that holds the (k-independent) displacement_norm column.
TRAJ_CSV="analysis/results/trajectory_geometry/trajectory_geometry.csv"

# k values to sweep
K_VALUES="3 5 10 20 50 100"

ALL_MODELS=(convnextv2 cyclemlp mit)
ALL_DATASETS=(imagenet100 isic2019 isic2018 voc2012_sbd)

MODEL_FILTER="${1:-}"
DATASET_FILTER="${2:-}"

OUT_DIR="analysis/results/k_sweep"

if [[ ! -f "$TRAJ_CSV" ]]; then
    echo "ERROR: trajectory CSV not found at $TRAJ_CSV"
    echo "       (needed for the k-independent displacement column)"
    exit 1
fi

for model in "${ALL_MODELS[@]}"; do
    [[ -n "$MODEL_FILTER" && "$model" != *"$MODEL_FILTER"* ]] && continue
    variant="${MODEL_VARIANT[$model]}"
    config_base="${MODEL_CONFIG[$model]}"

    for dataset in "${ALL_DATASETS[@]}"; do
        [[ -n "$DATASET_FILTER" && "$dataset" != *"$DATASET_FILTER"* ]] && continue

        task="${DS_TASK[$dataset]}"
        folder="${task}_${dataset}_${model}_${variant}"

        checkpoint="${ROOT}/checkpoints/${folder}/best/best.pth"
        input_dir="${ROOT}/checkpoints/${folder}/epochs"
        config="${INFERENCE_ENGINE}/configs/decomposition/${config_base}_${task}.yaml"

        if [[ ! -f "$checkpoint" ]]; then
            echo "[skip] $folder — no checkpoint at $checkpoint"; continue
        fi
        if [[ ! -d "$input_dir" ]]; then
            echo "[skip] $folder — no epoch dir at $input_dir"; continue
        fi
        if [[ ! -f "$config" ]]; then
            echo "[skip] $folder — no config at $config"; continue
        fi

        echo "========================================================"
        echo "  k-sweep: $model $variant | $task | $dataset"
        echo "========================================================"

        IGNORE_ARG="--ignore_index ${DS_IGNORE[$dataset]}"

        python analysis/ksweep_correlation.py "$task" \
            --backbone        "$model" \
            --variant         "$variant" \
            --num_classes     "${DS_CLASSES[$dataset]}" \
            --dataset         "$dataset" \
            --data_root       "${ROOT}/${DS_FOLDER[$dataset]}" \
            --checkpoint_path "$checkpoint" \
            --input_dir       "$input_dir" \
            --config          "$config" \
            --val_transforms  "${INFERENCE_ENGINE}/configs/transform/${DS_TRANSFORM[$dataset]}_val.yaml" \
            --loss_config     "${INFERENCE_ENGINE}/configs/loss/${DS_LOSS[$dataset]}.yaml" \
            --trajectory_csv  "$TRAJ_CSV" \
            --k_values        $K_VALUES \
            --anchor_size     "${DS_ANCHOR_SIZE[$dataset]}" \
            --batch_size      "${DS_BATCH_SIZE[$dataset]}" \
            --num_workers     4 \
            --device          cuda:0 \
            --seed            42 \
            --out_dir         "$OUT_DIR" \
            $IGNORE_ARG

        echo "  Done: $folder"
        echo ""
    done
done

# Concatenate all k-sweep tables into one global file
echo "Concatenating k-sweep results..."
python - <<'PYEOF'
import glob, os, pandas as pd
out = "analysis/results/k_sweep"
files = sorted(glob.glob(os.path.join(out, "ksweep_*.csv")))
files = [f for f in files if not f.endswith("ksweep_ALL.csv")]
if files:
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    path = os.path.join(out, "ksweep_ALL.csv")
    df.to_csv(path, index=False)
    print(f"  Wrote {path} ({len(df)} rows)")
    # Quick sign-persistence view for linear/projection layers
    mask = df["layer_type"].str.contains("linear|attention|conv2d", case=False, regex=True)
    piv = df[mask].pivot_table(index=["model","task","layer_type"],
                               columns="k", values="rho_disp_sens")
    print("\nrho(displacement, sensitivity) across k "
          "(linear/projection/conv2d layers):")
    print(piv.to_string(float_format=lambda x: f"{x:+.3f}"))
else:
    print("  No k-sweep files found.")
PYEOF

echo "All k-sweep runs complete."
