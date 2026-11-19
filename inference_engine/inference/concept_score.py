import numpy as np
import torch
from sklearn.svm import LinearSVC

from projections.common.layer_decomposition import LayerDecomposition


class ConceptScorer:
    """Scores all decomposed layer components against a concept class via CAV."""

    def score(
        self,
        pooled_buffers: dict[str, torch.Tensor],
        labels:         torch.Tensor,
        layer_keys:     list[str],
        pca_dict:       dict[str, LayerDecomposition],
        class_idx:      int,
    ) -> np.ndarray:
        """
        Return one float per point in [-1, 1] across all layers.

        For each layer a LinearSVC is fitted on the pooled activations to find
        a unit CAV vector.  The dot product of the per-filter mean activation
        with the CAV gives the score for that filter.  Layers whose module was
        deduplicated (not present in pooled_buffers) get zeros.

        Parameters
        ----------
        pooled_buffers : layer_key -> [N, C] tensor (CPU)
        labels         : [N] integer tensor, one label per sample
        layer_keys     : ordered list of all layer keys
        pca_dict       : PCA params per layer (only num_components is used here)
        class_idx      : which class to compute the CAV for

        Returns
        -------
        scores_flat : float32 array of shape [total_points_across_all_layers]
                      already normalized to [-1, 1]
        """
        if not pooled_buffers or labels is None:
            print("[ConceptScorer] No eval data cached, run eval first")
            return np.array([], dtype=np.float32)

        label_np  = labels.numpy()
        pos_mask  = (label_np == class_idx)

        all_scores: list[np.ndarray] = []

        for key in layer_keys:
            n_components = pca_dict[key].num_components

            if key not in pooled_buffers:
                # HookManager deduplicates shared modules (e.g. fused QKV): only
                # the first key for a given module path gets a buffer entry.
                # Subsequent keys that share the same module are skipped there
                # and must be zeroed out here to keep the score array aligned
                # with the point cloud layout.
                print(f"[ConceptScorer] '{key}': no activation buffer (deduplicated by HookManager). Replacing {n_components} score(s) with 0")
                all_scores.append(np.zeros(n_components, dtype=np.float32))
                continue

            X = pooled_buffers[key].numpy()   # [N, C]
            y = pos_mask.astype(int)

            if pos_mask.sum() < 2 or (~pos_mask).sum() < 2:
                # Not enough positive / negative samples
                all_scores.append(np.zeros(X.shape[1], dtype=np.float32))
                print(f"[ConceptScorer] '{key}': {pos_mask.sum()} positive, {(~pos_mask).sum()} negative samples for class {class_idx}, need >= 2 each. Scores set to 0.")
                continue

            svc = LinearSVC(max_iter=1000, class_weight="balanced")
            svc.fit(X, y)
            cav    = svc.coef_[0] / (np.linalg.norm(svc.coef_[0]) + 1e-8)
            scores = (X.mean(axis=0) * cav).astype(np.float32)

            # Normalize per layer: activation scales differ by orders of
            # magnitude across layers, so a global max would flatten most to 0
            layer_max = np.abs(scores).max()
            if layer_max > 1e-8:
                scores /= layer_max

            all_scores.append(scores)

        scores_flat = np.concatenate(all_scores)

        #q = np.percentile(np.abs(scores_flat), [50, 90, 99, 100])
        #print(f"[ConceptScorer] |score| p50={q[0]:.2e} p90={q[1]:.2e} p99={q[2]:.2e} p100={q[3]:.2e}")



        abs_max = np.abs(scores_flat).max()
        if abs_max > 1e-8:
            scores_flat /= abs_max

        return np.clip(scores_flat, -1.0, 1.0)
