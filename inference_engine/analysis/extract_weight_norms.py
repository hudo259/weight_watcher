# extract_weight_norms.py (run once per model/task)
import torch, pandas as pd, sys, os
sys.path.insert(0, '.')
from omegaconf import OmegaConf
from projections.common.templates import expand_templates
from projections.projection import _register_resolvers, get_state_dict_key
from itertools import product
_register_resolvers()

TASKS = ["classification_imagenet100", "classification_isic2019", "segmentation_voc2012_sbd", "segmentation_isic2018"]
TASK_TYPES = ["classification", "segmentation"]
MODELS = ["convnextv2_base", "cyclemlp_b2", "mit_b2"]
chkpt_path_root = "/home/hudo/Documents/checkpoints"

rows = []
for task_model in product(TASKS, MODELS):
    task_model_path = task_model[0] + "_" + task_model[1]
    ckpt_path = os.path.join(chkpt_path_root, str(task_model_path), "best/best.pth")

    model = task_model[1]
    task_type = "classification" if task_model[0][0] == 'c' else "segmentation"
    if "cyclemlp" in task_model[1]:
        model = "cycle_mlp_b2"
    config    = OmegaConf.to_container(OmegaConf.load(f"configs/decomposition/{model}_{task_type}.yaml"), resolve=True)
    mapping   = expand_templates(config)

    sdk = get_state_dict_key(ckpt_path)
    sd  = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd  = sd[sdk] if sdk else sd

    for layer_key, info in mapping.items():
        if layer_key not in sd:
            continue

        lt = info["type"] if isinstance(info, dict) else info
        W  = sd[layer_key].float()
        flat = W.reshape(W.shape[0], -1) # [C, d] unit index on dim 0
        norms = flat.norm(dim=1).numpy()

        for i, n in enumerate(norms):
            rows.append({"model": str(task_model[1]),
                        "task": str(task_model[0]),
                        "layer_key": layer_key, "layer_type": lt,
                        "unit_idx": i, "weight_norm": float(n)})

pd.DataFrame(rows).to_csv(f"analysis/results/weight_norms.csv", index=False)
