"""
Owns all forward-hook registration and removal.

Three hook types:
  - Pooled hooks:   one per unique parent module, accumulate [N, C] activations
                    across the full eval pass for concept scoring.
  - Spatial hook:   on a named feature layer, captures the raw activation tensor
                    for feature vector extraction.
  - Temp hooks:     short-lived hooks for activation map requests.
                    Registered via a context manager so they are always removed
                    even if an exception is raised during the forward pass.
"""

from contextlib import contextmanager
from collections.abc import Generator

import torch
import torch.nn as nn


class HookManager:
    """Manages forward hooks on a model."""

    def __init__(self):
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        # Pooled activations: layer_key -> [N_total, C]  (accumulated across batches)
        self._pooled_buffers: dict[str, torch.Tensor] = {}
        # Raw spatial activation from the feature layer (last batch only)
        self._feature_buffer: torch.Tensor | None = None

    # Permanent hooks 
    def register_pooled_hooks(self, model: nn.Module, layer_keys: list[str]) -> None:
        """
        Register one pooled hook per unique parent module.

        Multiple layer keys may share the same parent (e.g. fused QKV).  We
        deduplicate so the module only fires once per forward pass.  The first
        key encountered is the canonical key for that module; subsequent keys
        that share the same module path are skipped and will have no entry in
        pooled_buffers. Callers must handle missing keys gracefully!
        """
        seen: set[str] = set()

        for key in layer_keys:
            module_path = self._resolve_module_path(model, key)
            if module_path in seen:
                continue
            seen.add(module_path)

            def _make_hook(k: str):
                def _hook(module, input, output):
                    with torch.no_grad():
                        act = output[0] if isinstance(output, tuple) else output
                        if act.dim() == 4:
                            pooled = act.mean(dim=(2, 3))       # [B, C]
                        elif act.dim() == 3:
                            pooled = act[:, 0, :]               # [B, D] (CLS token)
                        else:
                            pooled = act                         # [B, C]
                        pooled = pooled.detach().cpu()
                        if k in self._pooled_buffers:
                            self._pooled_buffers[k] = torch.cat(
                                [self._pooled_buffers[k], pooled], dim=0
                            )
                        else:
                            self._pooled_buffers[k] = pooled
                return _hook

            handle = model.get_submodule(module_path).register_forward_hook(
                _make_hook(key)
            )
            self._handles.append(handle)

    def register_spatial_hook(self, model: nn.Module, layer_name: str) -> None:
        """Register a hook that captures the raw spatial activation (last batch)."""
        def _hook(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            self._feature_buffer = act.detach()

        handle = model.get_submodule(layer_name).register_forward_hook(_hook)
        self._handles.append(handle)

    def clear(self) -> None:
        """Remove all permanent hooks and reset buffers."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._pooled_buffers.clear()
        self._feature_buffer = None

    def clear_pooled_buffers(self) -> None:
        """Reset accumulated pooled activations without removing hooks."""
        self._pooled_buffers.clear()

    # Temporary hooks (context manager)
    @contextmanager
    def temp_spatial_hooks(
        self,
        model: nn.Module,
        by_layer: dict[int, list[int]],
        layer_keys: list[str],
    ) -> Generator[dict[int, list[torch.Tensor]], None, None]:
        """
        Context manager that registers temporary per-cloud spatial hooks,
        yields the accumulation buffers, then removes the hooks.

        Use like this:

            with self._hooks.temp_spatial_hooks(model, by_layer, layer_keys) as bufs:
                for batch in dataloader:
                    self._forward(batch)

        Yields:
            bufs: dict[cloud_idx -> list[Tensor]]
                  Each tensor is one batch's activation [B, C, H, W] or [B, T, D].
                  Caller concatenates after the loop.
        """
        handles: list[torch.utils.hooks.RemovableHandle] = []
        buffers: dict[int, list[torch.Tensor]] = {ci: [] for ci in by_layer}

        for cloud_idx in by_layer:
            layer_key   = layer_keys[cloud_idx]
            module_name = layer_key.replace(".weight", "")
            try:
                module = model.get_submodule(module_name)
            except AttributeError:
                print(f"[HookManager] Could not find module '{module_name}'! Skipping...")
                continue

            def _make_hook(ci: int):
                def _hook(m, inp, out):
                    act = out[0] if isinstance(out, tuple) else out
                    buffers[ci].append(act.detach().cpu())
                return _hook

            handles.append(module.register_forward_hook(_make_hook(cloud_idx)))

        try:
            yield buffers
        finally:
            for h in handles:
                h.remove()

    @property
    def feature_buffer(self) -> torch.Tensor | None:
        return self._feature_buffer

    @property
    def pooled_buffers(self) -> dict[str, torch.Tensor]:
        return self._pooled_buffers

    @staticmethod
    def _resolve_module_path(model: nn.Module, layer_key: str) -> str:
        """
        Walk from the full parameter key back to the nearest existing submodule.
        E.g. 'encoder.layer1.conv.weight' -> 'encoder.layer1.conv'
        """
        parts = layer_key.split(".")
        for i in range(len(parts) - 1, 0, -1):
            candidate = ".".join(parts[:i])
            try:
                model.get_submodule(candidate)
                return candidate
            except AttributeError:
                continue
        raise AttributeError(f"Could not find parent module for key: {layer_key}")
