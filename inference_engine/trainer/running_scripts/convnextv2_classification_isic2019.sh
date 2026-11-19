#!/usr/bin/env bash
set -euo pipefail

name=classification_isic2019_convnextv2_base

#CGVR135
root_dir="<path>/<to>/<project>/<root>/inference_engine"
output_dir="${root_dir}/data/checkpoints"
data_root="${root_dir}/data/imagenet100"

python train.py classification \
    --data_root        ${data_root} \
    --dataset          isic2019 \
    --num_classes      8 \
    --model_config     ${root_dir}/configs/model/convnextv2.yaml \
    --optimizer_config ${root_dir}/configs/optimizer/adamw_${name}.yaml \
    --train_transforms ${root_dir}/configs/transform/isic2019_train.yaml \
    --val_transforms   ${root_dir}/configs/transform/isic2019_val.yaml \
    --loss_config      ${root_dir}/configs/loss/cross_entropy_classification.yaml \
    --epochs           200 \
    --batch_size       32 \
    --grad_accum       2 \
    --amp \
    --checkpoint_dir   ${output_dir}/${name} \
    --eval_every       1 \
    --save_every       5 \
    --device           "cuda" \
    --pretrained       ${output_dir}/classification_imagenet100_convnextv2_base/best.pth
