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
  "convnextv2 base"
  "cyclemlp b2"
  "mit b2"
)
UMAP_RUNS=("0-1_5" "0-1_15" "0-1_25" "0-1_50")
OUTPUT_ROOT="./results/reconstruction_fildelity"

run_method() {
    read -a strarr <<< "$1"
    local dataset=${strarr[0]}
    local num_classes=${strarr[1]}
    local ignore_index=${strarr[2]}
    local task=${strarr[3]}

    read -a strarr <<< "$2"
    local model=${strarr[0]}
    local variant=${strarr[1]}

    local umap_run="$3"

    local folder_name="${task}_${dataset}_${model}_${variant}"
    local projection_folder_name="${model}_${variant}_${task}"

    echo "========================================================"
    echo "  Model  : $model $variant"
    echo "  Task   : $task"
    echo "  Dataset : ${dataset}"
    echo "  Ignore Index : ${ignore_index}"
    echo "  UMAP Run : ${umap_run}"
    echo "  Input Projection  : ${ROOT}/projections/${folder_name}"
    echo "  Output : ${OUTPUT_ROOT}/${folder_name}/${umap_run}"
    echo "========================================================"

    python reconstruction_fidelity.py \
      --backbone ${model} --variant ${variant} --num_classes ${num_classes} --ignore_index ${ignore_index} \
      --task ${task} --dataset ${dataset} \
      --data_root ${ROOT}/${dataset} \
      --checkpoint ${ROOT}/checkpoints/${folder_name}/best/best.pth \
      --runs \
        pca,${ROOT}/projections/${folder_name}_pca/${projection_folder_name}_decomposers.pt,${ROOT}/projections/${folder_name}_pca/${projection_folder_name}_projected.pt \
        umap,${ROOT}/projections/${folder_name}_umap/${umap_run}/${projection_folder_name}_decomposers.pt,${ROOT}/projections/${folder_name}_umap/${umap_run}/${projection_folder_name}_projected.pt \
        fourier_kpca,${ROOT}/projections/${folder_name}_fourier_kpca/${projection_folder_name}_decomposers.pt,${ROOT}/projections/${folder_name}_fourier_kpca/${projection_folder_name}_projected.pt \
      --device cuda:0 \
      --out_dir ${OUTPUT_ROOT}/${folder_name}/${umap_run}
}

for dnit in "${DATASETS_NUMCLASSES_IGNOREINDEX_TASKS[@]}"; do
    for model in "${MODELS[@]}"; do
        for umap_run in "${UMAP_RUNS[@]}"; do
            run_method "$dnit" "$model" "$umap_run"
        done
    done
done

echo "Finished."


#python reconstruction_fidelity.py \
#  --backbone convnextv2 --variant base --num_classes 100 \
#  --task classification --dataset imagenet100 \
#  --data_root /home/hudo/Documents/imagenet100 \
#  --checkpoint /home/hudo/Documents/checkpoints/classification_imagenet100_convnextv2_base/best/best.pth \
#  --runs \
#    pca,/home/hudo/Documents/projections/classification_imagenet100_convnextv2_base_pca/convnextv2_base_classification_decomposers.pt,/home/hudo/Documents/projections/classification_imagenet100_convnextv2_base_pca/convnextv2_base_classification_projected.pt \
#    umap,/home/hudo/Documents/projections/classification_imagenet100_convnextv2_base_umap/0-1_5/convnextv2_base_classification_decomposers.pt,/home/hudo/Documents/projections/classification_imagenet100_convnextv2_base_umap/0-1_5/convnextv2_base_classification_projected.pt \
#    fourier_kpca,/home/hudo/Documents/projections/classification_imagenet100_convnextv2_base_fourier_kpca/convnextv2_base_classification_decomposers.pt,/home/hudo/Documents/projections/classification_imagenet100_convnextv2_base_fourier_kpca/convnextv2_base_classification_projected.pt \
#  --device cuda:0 \
#  --out_dir ./results/reconstruction_fidelity
