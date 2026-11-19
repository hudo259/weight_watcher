#!/usr/bin/env bash
#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#cd "$SCRIPT_DIR"
#source .venv/bin/activate

ROOT="/path/to/weight-watcher/inference_engine/data"
P="$ROOT/projections"

python variance_explained.py \
  --runs \
    convnextv2_base,classification_imagenet100,pca,$P/classification_imagenet100_convnextv2_base_pca/convnextv2_base_classification_decomposers.pt \
    convnextv2_base,classification_isic2019,pca,$P/classification_isic2019_convnextv2_base_pca/convnextv2_base_classification_decomposers.pt \
    convnextv2_base,segmentation_isic2018,pca,$P/segmentation_isic2018_convnextv2_base_pca/convnextv2_base_segmentation_decomposers.pt \
    convnextv2_base,segmentation_voc2012_sbd,pca,$P/segmentation_voc2012_sbd_convnextv2_base_pca/convnextv2_base_segmentation_decomposers.pt \
    cyclemlp_b2,classification_imagenet100,pca,$P/classification_imagenet100_cyclemlp_b2_pca/cyclemlp_b2_classification_decomposers.pt \
    cyclemlp_b2,classification_isic2019,pca,$P/classification_isic2019_cyclemlp_b2_pca/cyclemlp_b2_classification_decomposers.pt \
    cyclemlp_b2,segmentation_isic2018,pca,$P/segmentation_isic2018_cyclemlp_b2_pca/cyclemlp_b2_segmentation_decomposers.pt \
    cyclemlp_b2,segmentation_voc2012_sbd,pca,$P/segmentation_voc2012_sbd_cyclemlp_b2_pca/cyclemlp_b2_segmentation_decomposers.pt \
    mit_b2,classification_imagenet100,pca,$P/classification_imagenet100_mit_b2_pca/mit_b2_classification_decomposers.pt \
    mit_b2,classification_isic2019,pca,$P/classification_isic2019_mit_b2_pca/mit_b2_classification_decomposers.pt \
    mit_b2,segmentation_isic2018,pca,$P/segmentation_isic2018_mit_b2_pca/mit_b2_segmentation_decomposers.pt \
    mit_b2,segmentation_voc2012_sbd,pca,$P/segmentation_voc2012_sbd_mit_b2_pca/mit_b2_segmentation_decomposers.pt \
  --out_dir ./results/variance_explained
