#!/usr/bin/env bash
set -euo pipefail

name=segmentation_voc2012_sbd_mit_b2

#CGVR135
root_dir="<path>/<to>/<project>/<root>/inference_engine"
output_dir="${root_dir}/data/checkpoints"
data_root="${root_dir}/data/imagenet100"

python train.py segmentation \
    --data_root        ${data_root} \
    --dataset          voc2012_sbd \
    --num_classes      21 \
    --ignore_index     255 \
    --model_config     ${root_dir}/configs/model/mit.yaml \
    --optimizer_config ${root_dir}/configs/optimizer/adamw_${name}.yaml \
    --train_transforms ${root_dir}/configs/transform/voc2012_train.yaml \
    --val_transforms   ${root_dir}/configs/transform/voc2012_val.yaml \
    --loss_config      ${root_dir}/configs/loss/cross_entropy_segmentation.yaml \
    --epochs           200 \
    --batch_size       8 \
    --grad_accum       2 \
    --amp \
    --checkpoint_dir   ${output_dir}/${name} \
    --eval_every       1 \
    --save_every       5 \
    --device           "cuda" \
    --pretrained       ${output_dir}/classification_imagenet100_mit_b2/best.pth
