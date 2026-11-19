#!/usr/bin/env bash
set -euo pipefail

name=classification_imagenet100_mit_b2

#CGVR135
root_dir="<path>/<to>/<project>/<root>/inference_engine"
output_dir="${root_dir}/data/checkpoints"
data_root="${root_dir}/data/imagenet100"

python train.py classification \
    --data_root        ${data_root} \
    --dataset          imagenet100 \
    --num_classes      100 \
    --model_config     ${root_dir}/configs/model/mit.yaml \
    --optimizer_config ${root_dir}/configs/optimizer/adamw_${name}.yaml \
    --train_transforms ${root_dir}/configs/transform/imagenet100_train.yaml \
    --val_transforms   ${root_dir}/configs/transform/imagenet100_val.yaml \
    --loss_config      ${root_dir}/configs/loss/cross_entropy_classification.yaml \
    --epochs           300 \
    --batch_size       16 \
    --grad_accum       4 \
    --amp \
    --checkpoint_dir   ${output_dir}/${name} \
    --eval_every       1 \
    --save_every       5 \
    --mixup            0.8 \
    --cutmix           1.0 \
    --device           "cuda"
