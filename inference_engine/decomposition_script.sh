#!/usr/bin/env bash
# run_projections.sh
# Runs PCA, FourierKPCA, and UMAP decomposition for all models and tasks.
#
# Usage:
#   ./run_projections.sh           # all methods, all models, all tasks
#   ./run_projections.sh pca       # PCA only
#   ./run_projections.sh fourier_kpca
#   ./run_projections.sh umap
#
# Run from the inference_engine root directory.

set -euo pipefail

# Prevent BLAS thread explosion: each worker gets exactly 1 BLAS thread.
# Without this, each of N workers spawns its own thread pool, causing
# N × num_blas_threads threads competing for the same CPU cores.
#export OMP_NUM_THREADS=2
#export MKL_NUM_THREADS=2
#export OPENBLAS_NUM_THREADS=2
#export NUMEXPR_NUM_THREADS=2

# Activate venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate

# Paths
CHECKPOINT_ROOT="./data/checkpoints"
PROJECTION_ROOT="./data/projections"
CONFIG_ROOT="configs/decomposition"

# Models and tasks 
# NOTE: checkpoint dirs use "cyclemlp_b2" (no underscore between cycle and mlp)
#       config files use "cycle_mlp_b2" (with underscore)
MODELS=("convnextv2_base" "cyclemlp_b2" "mit_b2")

TASKS=(
    "classification_imagenet100"
    "classification_isic2019"
    "segmentation_isic2018"
    "segmentation_voc2012_sbd"
)

# Note: bad ISIC 2018 runs (backbone_lr_scale=0.1, mIoU 0.1-0.4) are excluded
# here. Run manually for the case study analysis.

# Method selection
# Accept optional first argument to restrict to one method.
# If no argument given, run all three methods in order.
REQUESTED="${1:-all}"

# Helpers 
get_config_model() {
    # Maps checkpoint folder model name to config file model name.
    # Checkpoint dirs: cyclemlp_b2  →  Config files: cycle_mlp_b2
    case "$1" in
        "cyclemlp_b2") echo "cycle_mlp_b2" ;;
        *) echo "$1" ;;
    esac
}

run_method() {
    local method="$1"       # pca | fourier_kpca | umap
    local model="$2"
    local task="$3"
    local params="$4"

    local folder_name="${task}_${model}"
    local input_dir="${CHECKPOINT_ROOT}/${folder_name}/epochs"
    local model_checkpoint="${CHECKPOINT_ROOT}/${folder_name}/best"

    # Safety checks
    if [ ! -d "$input_dir" ]; then
        echo "  [SKIP] Missing checkpoint dir: $input_dir"
        return
    fi

    local task_type
    if [[ "$task" == *"classification"* ]]; then
        task_type="classification"
    else
        task_type="segmentation"
    fi

    local config_model
    config_model=$(get_config_model "$model")
    local config_file="${CONFIG_ROOT}/${config_model}_${task_type}.yaml"

    if [ ! -f "$config_file" ]; then
        echo "  [SKIP] Missing config: $config_file"
        return
    fi

    local output_dir="${PROJECTION_ROOT}/${folder_name}_${method}"
    mkdir -p "$output_dir"

    echo "========================================================"
    echo "  Method : $method"
    echo "  Model  : $model"
    echo "  Task   : $task"
    echo "  Input  : $input_dir"
    echo "  Config : $config_file"
    echo "  Output : $output_dir"
    echo "========================================================"

    case "$method" in

        pca)
            # PCA is fast and exactly invertible, save trajectory for trajectory_geometry.py
            python projections/projection.py pca \
                -i "$input_dir/" \
                -m "$model_checkpoint/best.pth" \
                -c "$config_file" \
                -o "$output_dir/" \
                -k 3 \
                --save_trajectory \
                --num_workers 1 \
                --chunk_size 16 \
                --device cuda
            ;;

        fourier_kpca)
            # FourierKPCA is invertible and nonlinear, save trajectory for method_consistency
            # kpca_gamma=None: auto-estimated from data (recommended)
            # kpca_n_rff_features=2048: higher than default 1024 for better approximation
            python projections/projection.py fourier_kpca \
                -i "$input_dir/" \
                -m "$model_checkpoint/best.pth" \
                -c "$config_file" \
                -o "$output_dir/" \
                -k 3 \
                --save_trajectory \
                --kpca_n_rff_features 1024 \
                --kpca_regularization 0.001 \
                --num_workers 1 \
                --chunk_size 1 \
                --device cpu
            ;;

        umap)
            # UMAP MUST use --num_workers 1 (UMAP uses joblib internally,
            # deadlocks with Python multiprocessing).
            read -a strarr <<< "$4"
            python projections/projection.py umap \
                -i "$input_dir/" \
                -m "$model_checkpoint/best.pth" \
                -c "$config_file" \
                -o "$output_dir/" \
                -k 3 \
                --save_trajectory \
                --umap_n_neighbors 15 \
                --umap_min_dist 0.1 \
                --num_workers 1 \
                --chunk_size 4 \
                --device cpu
            ;;

        *)
            echo "  [ERROR] Unknown method: $method"
            exit 1
            ;;
    esac

    echo "  Done: $method / $folder_name"
    echo ""
}

# Main loop
mkdir -p "$PROJECTION_ROOT"

for method in pca fourier_kpca umap; do

    # Skip if a specific method was requested and this isn't it
    if [[ "$REQUESTED" != "all" && "$REQUESTED" != "$method" ]]; then
        continue
    fi

    echo ""
    echo "###############################################################"
    echo "  Starting method: $method"
    echo "###############################################################"
    echo ""

    for model in "${MODELS[@]}"; do
        for task in "${TASKS[@]}"; do
            run_method "$method" "$model" "$task"
        done
    done
    echo "Waiting for memory to be reclaimed..."
    sleep 30

    echo ""
    echo "###############################################################"
    echo "  Finished method: $method"
    echo "###############################################################"
    echo ""

done

echo "All projections complete."
