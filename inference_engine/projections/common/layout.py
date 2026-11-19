import re
import numpy as np
import pandas as pd
import torch

# Depth Layout: Basically the network topology as depicted by block diagrams in papers
_LAYER_TYPE_Z = {
    "conv2d":                       0,
    "conv1d":                       0,
    "depthwise_conv":               1,
    "linear":                       2,
    "attention_query":              3,
    "attention_key":                3,
    "attention_value":              3,
    "attention_kv":                 3,
    "attention_output":             3,
    "attention_fused_qkv":          3,
    "attention_mha_head":           3,
    "attention_mha_head_fused_qkv": 3,
    "other":                        3,
}

LAYER_TYPE_COLORS = {
    "conv2d":                       (0.122, 0.467, 0.706),
    "conv1d":                       (0.122, 0.467, 0.706),  # same as conv2d
    "depthwise_conv":               (1.000, 0.498, 0.055),
    "linear":                       (0.173, 0.627, 0.173),
    "attention_query":              (0.839, 0.153, 0.157),
    "attention_key":                (0.839, 0.153, 0.157),  # same as query
    "attention_value":              (0.839, 0.153, 0.157),  # same as query
    "attention_kv":                 (0.549, 0.337, 0.294),
    "attention_output":             (0.580, 0.404, 0.741),
    "attention_fused_qkv":          (0.839, 0.153, 0.157),  # same as query
    "attention_mha_head":           (0.839, 0.153, 0.157),
    "attention_mha_head_fused_qkv": (0.839, 0.153, 0.157),
    "other":                        (0.498, 0.498, 0.498),
}

def _secondary_index(layer_key: str) -> int:
    nums = re.findall(r'\d+', layer_key)
    return int(nums[-1]) if len(nums) > 1 else 0

def compute_depth_layout(
    layer_keys:    list[str],
    layer_mapping: dict,
    x_spacing: float = 6.0,
    y_spacing: float = 3.0,
    z_spacing: float = 4.0,
) -> np.ndarray:
    """
    Layout clouds in forward-pass order.
      X: topological depth (position in layer_keys)
      Y: within-block index (head / filter group index)
      Z: layer type category (conv / linear / attention)
    Returns float32 [L, 3], centered at origin.
    """
    positions = np.zeros((len(layer_keys), 3), dtype=np.float32)
    for depth_idx, key in enumerate(layer_keys):
        info       = layer_mapping.get(key, "linear")
        layer_type = info["type"] if isinstance(info, dict) else info
        positions[depth_idx] = [
            depth_idx                              * x_spacing,
            _secondary_index(key)                  * y_spacing,
            _LAYER_TYPE_Z.get(layer_type, 1)       * z_spacing,
        ]
    positions -= positions.mean(axis=0)
    return positions

# Centered Kernel Alignment (CKA) Layout
def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic  = (X.T @ Y).pow(2).sum()
    denom = ((X.T @ X).pow(2).sum() * (Y.T @ Y).pow(2).sum()).sqrt()
    return (hsic / denom.clamp(min=1e-10)).item()

def compute_cka_layout(
    pooled_buffers: dict[str, torch.Tensor],
    layer_keys:     list[str],
    spacing:        float = 6.0,
    random_state:   int   = 42,
) -> np.ndarray:
    """
    Pairwise linear Central Kernel Alignment (CKA) between all layers -> MDS embedding -> 3D layout.
    Returns float32 [L, 3], centered at origin.

    Taken from: 
    Kornblith, Simon, et al. "Similarity of neural network representations revisited." 
    International conference on machine learning. PMlR, 2019.
    """
    from sklearn.manifold import MDS

    L       = len(layer_keys)
    tensors = [pooled_buffers.get(k) for k in layer_keys]

    cka = np.eye(L, dtype=np.float32)
    for i in range(L):
        for j in range(i + 1, L):
            if tensors[i] is None or tensors[j] is None:
                sim = 0.0
            else:
                sim = _linear_cka(tensors[i].float(), tensors[j].float())
            cka[i, j] = cka[j, i] = sim

    dist = np.sqrt(np.clip(1.0 - cka, 0.0, 1.0))
    mds  = MDS(n_components=3, dissimilarity="precomputed",
               normalized_stress=False, random_state=random_state)
    pos  = mds.fit_transform(dist).astype(np.float32)

    mean_dist = np.linalg.norm(pos[None] - pos[:, None], axis=-1).mean()
    if mean_dist > 1e-6:
        pos *= spacing / mean_dist
    pos -= pos.mean(axis=0)
    return pos

def _norm(s: str) -> str:
    """Strip all non-alphanumeric characters and lowercase"""
    return re.sub(r'[^a-z0-9]', '', s.lower())

def _filter_experiment_data(df: pd.DataFrame, model_name: str | None = None, dataset_name:str | None = None) -> pd.DataFrame:
    """Filter to the specific model/task if those columns exist"""
    if model_name and "model" in df.columns:
        df = df[df["model"].map(_norm) == _norm(model_name)]

    if dataset_name and "task" in df.columns:
        dn = _norm(dataset_name)
        df = df[df["task"].map(lambda task: dn in _norm(task) or _norm(task) in dn)]

    return df

def compute_trajectory_geometry_layout(
    model_name: str,
    layer_keys: list[str],
    layer_types: dict[str, str],
    csv_path: str,   
    dataset_name: str | None = None,
    z_spacing: float = 6.0,
) -> np.ndarray:
    """
    Uses raw trajectory_geometry.csv (has per-unit rows with layer_key column).
    """
    df = pd.read_csv(csv_path)
    df = _filter_experiment_data(df, model_name, dataset_name)

    lookup = df.groupby("layer_key")[["ratio", "displacement_norm"]].mean()
    positions = np.zeros((len(layer_keys), 3), dtype=np.float32)

    d_min = lookup["displacement_norm"].min()
    d_max = lookup["displacement_norm"].max()
    n = len(layer_keys)
    scale = 20.0
    for depth_idx, key in enumerate(layer_keys):
        if key in lookup.index:
            ratio = float(lookup.loc[key, "ratio"])
            disp  = float(lookup.loc[key, "displacement_norm"])
        else:
            ratio, disp = 0.5, 0.0


        #lt = layer_types.get(key, "linear")

        # normalize
        depth_idx_norm = depth_idx / max(n - 1, 1)   # [0, 1]
        disp_norm = (disp - d_min) / max(d_max - d_min, 1e-8)

        positions[depth_idx] = [
            ratio * scale, # X: directionality
            disp_norm * scale, # Y: displacement
            depth_idx_norm * scale, # Z: topological depth (creates 3D timeline)
        ]
    positions -= positions.mean(axis=0)
    return positions

def compute_sensitivity_layout(
    model_name: str,
    layer_keys: list[str],
    layer_types: dict[str, str],
    csv_path: str,
    dataset_name: str | None = None,
    x_spacing: float = 6.0,
) -> np.ndarray:
    """
    X: mean sensitivity score per layer (high = functionally important)
    Y: layer type category
    Z: layer depth
    Uses raw sensitivity.csv (has per-unit rows with layer_key column).
    """
    df = pd.read_csv(csv_path)
    df = _filter_experiment_data(df, model_name, dataset_name)

    lookup = df.groupby("layer_key")["sensitivity"].mean()
    s_min  = lookup.min()
    s_max  = lookup.max()

    positions = np.zeros((len(layer_keys), 3), dtype=np.float32)

    lt_max = max(_LAYER_TYPE_Z.values())  # = 3
    n = len(layer_keys)
    scale = 20.0
    for depth_idx, key in enumerate(layer_keys):
        s_raw  = float(lookup.get(key, s_min))
        s_norm = (s_raw - s_min) / max(s_max - s_min, 1e-8)
        lt = layer_types.get(key, "linear")

        # normalize
        depth_idx_norm = depth_idx / max(n - 1, 1)   # [0, 1]

        positions[depth_idx] = [
            s_norm * scale,
            (_LAYER_TYPE_Z.get(lt, 1) / lt_max) * scale, # Z: layer type (parallel lanes, for a perfectly flat 2D graph, change to 0)
            depth_idx_norm * scale,
        ]
    positions -= positions.mean(axis=0)
    return positions

def compute_explained_variance_layout(
    model_name: str,
    layer_keys: list[str],
    layer_types: dict[str, str],
    csv_path: str,
    dataset_name: str | None = None,
    x_spacing: float = 6.0,
) -> np.ndarray:
    """
    Uses raw variance_explained.csv (has per-unit rows with layer_key column).
    """
    df = pd.read_csv(csv_path)
    df = _filter_experiment_data(df, model_name, dataset_name)

    lookup = df.groupby("layer_key")["explained_variance"].mean()
    positions = np.zeros((len(layer_keys), 3), dtype=np.float32)

    lt_max = max(_LAYER_TYPE_Z.values())  # = 3
    n = len(layer_keys)
    scale = 20.0
    for depth_idx, key in enumerate(layer_keys):
        ev = float(lookup.get(key, 0.2))
        lt = layer_types.get(key, "linear")

        # normalize
        depth_idx_norm = depth_idx / max(n - 1, 1)   # [0, 1]

        positions[depth_idx] = [
            ev * scale, # X: explained variance (Curve height)
            (_LAYER_TYPE_Z.get(lt, 1) / lt_max) * scale, # Y: layer type (parallel lanes, for a perfectly flat 2D graph, change to 0)
            depth_idx_norm * scale, # Z: Toplogical depth (creates a timeline)
        ]
    positions -= positions.mean(axis=0)
    return positions

def compute_correlation_layout(
    model_name: str,
    layer_keys: list[str],
    trajectory_csv_path: str,
    sensitivity_csv_path: str,
    dataset_name: str | None = None,
) -> np.ndarray:
    """
    X: Directionality Ratio
    Y: Normalized Displacement
    Z: Normalized Sensitivity Score
    """
    # Load Trajectory Data
    df_traj = pd.read_csv(trajectory_csv_path)
    df_traj = _filter_experiment_data(df_traj, model_name, dataset_name)

    lookup_traj = df_traj.groupby("layer_key")[["ratio", "displacement_norm"]].mean()

    # Pre-calculate displacement bounds ONCE outside the loop
    if not lookup_traj.empty:
        d_min = lookup_traj["displacement_norm"].min()
        d_max = lookup_traj["displacement_norm"].max()
    else:
        d_min, d_max = 0.0, 1.0

    # Load Sensitivity Data in a similar way
    df_sens = pd.read_csv(sensitivity_csv_path)
    df_sens = _filter_experiment_data(df_sens, model_name, dataset_name)

    lookup_sens = df_sens.groupby("layer_key")["sensitivity"].mean()

    # Purge NaNs so they don't break the math later
    lookup_sens = lookup_sens.dropna()

    # Handle completely empty/NaN sensitivity sets gracefully
    if lookup_sens.empty:
        print("[Worker] compute_correlation_layout: Sensitivity data is empty or all NaNs. Defaulting Z-axis to 0.")
        s_min, s_max = 0.0, 1.0
    else: 
        # Normalize sensitivity for better Z-axis scaling
        s_min = lookup_sens.min()
        s_max = lookup_sens.max()

    positions = np.zeros((len(layer_keys), 3), dtype=np.float32)

    scale = 20.0
    for depth_idx, key in enumerate(layer_keys):
        # Get geometry
        if key in lookup_traj.index:
            ratio = float(lookup_traj.loc[key, "ratio"])
            disp  = float(lookup_traj.loc[key, "displacement_norm"])
        else:
            ratio, disp = 0.5, 0.0

        # Get sensitivity (missing keys or purged NaNs safely fall back to s_min)
        s_raw = float(lookup_sens.get(key, s_min))
        
        # Calculate normalizations
        s_norm = (s_raw - s_min) / max(s_max - s_min, 1e-8)
        disp_norm = (disp - d_min) / max(d_max - d_min, 1e-8)

        positions[depth_idx] = [
            ratio * scale,          # X: Directionality
            disp_norm * scale,     # Y: Displacement
            s_norm * scale,         # Z: Sensitivity (Height)
        ]
        
    positions -= positions.mean(axis=0)
    return positions
