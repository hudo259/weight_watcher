ROOT="./data"

python main.py classification \
    --backbone mit --variant b2 --num_classes 100 \
    --checkpoint_path ${ROOT}/checkpoints/classification_imagenet100_mit_b2/best/best.pth \
    --decomposers_path ${ROOT}/projections/classification_imagenet100_mit_b2_pca/mit_b2_classification_decomposers.pt \
    --projection_path ${ROOT}/projections/classification_imagenet100_mit_b2_pca/mit_b2_classification_projected.pt \
    --dataset imagenet100 \
    --data_root ${ROOT}/imagenet100 \
    --trajectory_geometry_csv analysis/results/trajectory_geometry/trajectory_geometry.csv \
    --variance_explained_csv analysis/results/variance_explained/variance_explained.csv \
    --anchor_size 100 \
    --batch_size 16 \
    --val_transforms configs/transform/imagenet100_val.yaml

#python main.py classification \
#    --backbone cyclemlp --variant b2 --num_classes 100 \
#    --checkpoint_path ${ROOT}/checkpoints/classification_imagenet100_cyclemlp_b2/best/best.pth \
#    --decomposers_path ${ROOT}/projections/classification_imagenet100_cyclemlp_b2_pca/cyclemlp_b2_classification_decomposers.pt \
#    --projection_path ${ROOT}/projections/classification_imagenet100_cyclemlp_b2_pca/cyclemlp_b2_classification_projected.pt \
#    --dataset imagenet100 \
#    --data_root ${ROOT}/imagenet100 \
#    --trajectory_geometry_csv analysis/results/trajectory_geometry/trajectory_geometry.csv \
#    --variance_explained_csv analysis/results/variance_explained/variance_explained.csv \
#    --anchor_size 100 \
#    --batch_size 16 \
#    --val_transforms configs/transform/imagenet100_val.yaml
#
#python main.py classification \
#    --backbone convnextv2 --variant base --num_classes 100 \
#    --checkpoint_path ${ROOT}/checkpoints/classification_imagenet100_convnextv2_b2/best/best.pth \
#    --decomposers_path ${ROOT}/projections/classification_imagenet100_convnextv2_b2_pca/convnextv2_b2_classification_decomposers.pt \
#    --projection_path ${ROOT}/projections/classification_imagenet100_convnextv2_b2_pca/convnextv2_b2_classification_projected.pt \
#    --dataset imagenet100 \
#    --data_root ${ROOT}/imagenet100 \
#    --trajectory_geometry_csv analysis/results/trajectory_geometry/trajectory_geometry.csv \
#    --variance_explained_csv analysis/results/variance_explained/variance_explained.csv \
#    --anchor_size 100 \
#    --batch_size 16 \
#    --val_transforms configs/transform/imagenet100_val.yaml
#
#python main.py segmentation \
#    --backbone mit --variant b2 --num_classes 21 \
#    --checkpoint_path ${ROOT}/checkpoints/segmentation_voc2012_sbd_mit_b2/best/best.pth \
#    --decomposers_path ${ROOT}/projections/segmentation_voc2012_sbd_mit_b2_pca/mit_b2_segmentation_decomposers.pt \
#    --projection_path ${ROOT}/projections/segmentation_voc2012_sbd_mit_b2_pca/mit_b2_segmentation_projected.pt \
#    --dataset voc2012_sbd \
#    --data_root ${ROOT}/voc2012_sbd \
#    --trajectory_geometry_csv analysis/results/trajectory_geometry/trajectory_geometry.csv \
#    --variance_explained_csv analysis/results/variance_explained/variance_explained.csv \
#    --anchor_size 100 \
#    --batch_size 16 \
#    --val_transforms configs/transform/voc2012_val.yaml
#
#python main.py segmentation \
#    --backbone cyclemlp --variant b2 --num_classes 21 \
#    --checkpoint_path ${ROOT}/checkpoints/segmentation_voc2012_sbd_cyclemlp_b2/best/best.pth \
#    --decomposers_path ${ROOT}/projections/segmentation_voc2012_sbd_cyclemlp_b2_pca/cyclemlp_b2_segmentation_decomposers.pt \
#    --projection_path ${ROOT}/projections/segmentation_voc2012_sbd_cyclemlp_b2_pca/cyclemlp_b2_segmentation_projected.pt \
#    --dataset voc2012_sbd \
#    --data_root ${ROOT}/voc2012_sbd \
#    --trajectory_geometry_csv analysis/results/trajectory_geometry/trajectory_geometry.csv \
#    --variance_explained_csv analysis/results/variance_explained/variance_explained.csv \
#    --anchor_size 100 \
#    --batch_size 16 \
#    --val_transforms configs/transform/voc2012_val.yaml
#
#python main.py segmentation \
#    --backbone convnextv2 --variant base --num_classes 21 \
#    --checkpoint_path ${ROOT}/checkpoints/segmentation_voc2012_sbd_convnextv2_b2/best/best.pth \
#    --decomposers_path ${ROOT}/projections/segmentation_voc2012_sbd_convnextv2_b2_pca/convnextv2_b2_segmentation_decomposers.pt \
#    --projection_path ${ROOT}/projections/segmentation_voc2012_sbd_convnextv2_b2_pca/convnextv2_b2_segmentation_projected.pt \
#    --dataset voc2012_sbd \
#    --data_root ${ROOT}/voc2012_sbd \
#    --trajectory_geometry_csv analysis/results/trajectory_geometry/trajectory_geometry.csv \
#    --variance_explained_csv analysis/results/variance_explained/variance_explained.csv \
#    --anchor_size 100 \
#    --batch_size 16 \
#    --val_transforms configs/transform/voc2012_val.yaml
#
#python main.py classification \
#    --backbone mit --variant b2 --num_classes 8 \
#    --checkpoint_path ${ROOT}/checkpoints/classification_isic2019_mit_b2/best/best.pth \
#    --decomposers_path ${ROOT}/projections/classification_isic2019_mit_b2_pca/mit_b2_classification_decomposers.pt \
#    --projection_path ${ROOT}/projections/classification_isic2019_mit_b2_pca/mit_b2_classification_projected.pt \
#    --dataset isic2019 \
#    --data_root ${ROOT}/isic2019 \
#    --trajectory_geometry_csv analysis/results/trajectory_geometry/trajectory_geometry.csv \
#    --variance_explained_csv analysis/results/variance_explained/variance_explained.csv \
#    --anchor_size 100 \
#    --batch_size 16 \
#    --val_transforms configs/transform/isic2019_val.yaml
#
#python main.py classification \
#    --backbone cyclemlp --variant b2 --num_classes 8 \
#    --checkpoint_path ${ROOT}/checkpoints/classification_isic2019_cyclemlp_b2/best/best.pth \
#    --decomposers_path ${ROOT}/projections/classification_isic2019_cyclemlp_b2_pca/cyclemlp_b2_classification_decomposers.pt \
#    --projection_path ${ROOT}/projections/classification_isic2019_cyclemlp_b2_pca/cyclemlp_b2_classification_projected.pt \
#    --dataset isic2019 \
#    --data_root ${ROOT}/isic2019 \
#    --trajectory_geometry_csv analysis/results/trajectory_geometry/trajectory_geometry.csv \
#    --variance_explained_csv analysis/results/variance_explained/variance_explained.csv \
#    --anchor_size 100 \
#    --batch_size 16 \
#    --val_transforms configs/transform/isic2019_val.yaml
#
#python main.py classification \
#    --backbone convnextv2 --variant base --num_classes 8 \
#    --checkpoint_path ${ROOT}/checkpoints/classification_isic2019_convnextv2_b2/best/best.pth \
#    --decomposers_path ${ROOT}/projections/classification_isic2019_convnextv2_b2_pca/convnextv2_b2_classification_decomposers.pt \
#    --projection_path ${ROOT}/projections/classification_isic2019_convnextv2_b2_pca/convnextv2_b2_classification_projected.pt \
#    --dataset isic2019 \
#    --data_root ${ROOT}/isic2019 \
#    --trajectory_geometry_csv analysis/results/trajectory_geometry/trajectory_geometry.csv \
#    --variance_explained_csv analysis/results/variance_explained/variance_explained.csv \
#    --anchor_size 100 \
#    --batch_size 16 \
#    --val_transforms configs/transform/isic2019_val.yaml
#
#python main.py segmentation \
#    --backbone mit --variant b2 --num_classes 2 \
#    --checkpoint_path ${ROOT}/checkpoints/segmentation_isic2018_mit_b2/best/best.pth \
#    --decomposers_path ${ROOT}/projections/segmentation_isic2018_mit_b2_pca/mit_b2_segmentation_decomposers.pt \
#    --projection_path ${ROOT}/projections/segmentation_isic2018_mit_b2_pca/mit_b2_segmentation_projected.pt \
#    --dataset isic2018 \
#    --data_root ${ROOT}/isic2018_task1 \
#    --trajectory_geometry_csv analysis/results/trajectory_geometry/trajectory_geometry.csv \
#    --variance_explained_csv analysis/results/variance_explained/variance_explained.csv \
#    --anchor_size 100 \
#    --batch_size 16 \
#    --val_transforms configs/transform/isic2018_val.yaml
#
#python main.py segmentation \
#    --backbone cyclemlp --variant b2 --num_classes 2 \
#    --checkpoint_path ${ROOT}/checkpoints/segmentation_isic2018_cyclemlp_b2/best/best.pth \
#    --decomposers_path ${ROOT}/projections/segmentation_isic2018_cyclemlp_b2_pca/cyclemlp_b2_segmentation_decomposers.pt \
#    --projection_path ${ROOT}/projections/segmentation_isic2018_cyclemlp_b2_pca/cyclemlp_b2_segmentation_projected.pt \
#    --dataset isic2018 \
#    --data_root ${ROOT}/isic2018_task1 \
#    --trajectory_geometry_csv analysis/results/trajectory_geometry/trajectory_geometry.csv \
#    --variance_explained_csv analysis/results/variance_explained/variance_explained.csv \
#    --anchor_size 100 \
#    --batch_size 16 \
#    --val_transforms configs/transform/isic2018_val.yaml
#
#python main.py segmentation \
#    --backbone convnextv2 --variant base --num_classes 2 \
#    --checkpoint_path ${ROOT}/checkpoints/segmentation_isic2018_convnextv2_b2/best/best.pth \
#    --decomposers_path ${ROOT}/projections/segmentation_isic2018_convnextv2_b2_pca/convnextv2_b2_segmentation_decomposers.pt \
#    --projection_path ${ROOT}/projections/segmentation_isic2018_convnextv2_b2_pca/convnextv2_b2_segmentation_projected.pt \
#    --dataset isic2018 \
#    --data_root ${ROOT}/isic2018_task1 \
#    --trajectory_geometry_csv analysis/results/trajectory_geometry/trajectory_geometry.csv \
#    --variance_explained_csv analysis/results/variance_explained/variance_explained.csv \
#    --anchor_size 100 \
#    --batch_size 16 \
#    --val_transforms configs/transform/isic2018_val.yaml
