"""
"Orchestrator": owns the queues, the IPC socket, and the threading model.
All domain logic lives in the components it delegates to:

  WeightReconstructor: PCA state, weight reconstruction
  Evaluator: forward passes, metrics, image annotation
  HookManager: all forward hook registration / removal
  ConceptScorer: CAV scoring via LinearSVC
  ActivationMapper: spatial activation map computation

Threading model
---------------
IPC thread  (run loop): polls socket, dispatches callbacks, drains send queue.
Worker thread: dequeues jobs, calls components, pushes results.
All NNG socket calls stay on the IPC thread.
"""

import queue
import threading
from enum import IntEnum
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ipc.iipc import (
        IIPC, 
        VectorListID, 
        VectorListMsg, 
        LayoutID, 
        LayoutMsg, 
        JobStatus
)
from ipc.ipc_nng import NngIPC, PointCloudMsg, TransformedPointsMsg, TransformedPointsBatchMsg, MetricsMsg
from projections.common.layer_decomposition import LayerDecomposition
from projections.common.layout import (
    compute_depth_layout, 
    compute_cka_layout, 
    compute_sensitivity_layout, 
    compute_explained_variance_layout, 
    compute_trajectory_geometry_layout,
    compute_correlation_layout
)

from inference.weight_reconstruction import WeightReconstructor
from inference.evaluator import Evaluator
from inference.hook_manager import HookManager
from inference.concept_score import ConceptScorer
from inference.activation_map import ActivationMapper

class InferenceEngine:
    """Orchestrates IPC, weight manipulation, and inference."""

    # Values are sent to the renderer in JOB_STATUS (must match C++ ipc::JobType)
    class _JobType(IntEnum):
        POINT_CLOUD_JOB        = 0
        TRANSFORMED_POINTS_JOB = 1
        CONCEPT_SELECT_JOB     = 2
        ACTIVATION_REQUEST_JOB = 3
        RAW_IMAGES_JOB         = 4
        CKA_LAYOUT_JOB         = 5
        TRAJECTORY_GEOMETRY_JOB     = 6
        SENSITIVITY_LAYOUT_JOB      = 7
        EV_LAYOUT_JOB               = 8
        CORRELATION_LAYOUT_JOB      = 9

    def __init__(
        self,
        model:           nn.Module,
        model_name:      str,
        dataloader:      DataLoader,
        decompositions:  dict[str, LayerDecomposition],
        projections:     dict[str, dict],
        dataset_name:    str,
        num_classes:     int,
        loss_fn:         nn.Module | None = None,
        device:          str = "cuda:0",
        amp:             bool = False,
        ignore_index:    int = -100,
        feature_layer:   str | None = None,
        task:            Literal["classification", "segmentation"] = "classification",
        address:         str = "tcp://*:9999",
        trajectory_geometry_csv: str | None = None,
        sensitivity_csv: str | None = None,
        variance_explained_csv: str | None = None,
        pca_target_dims: int = -1,
    ):
        self._model   = model
        self._device  = device
        self._address = address
        self._pca_target_dims = pca_target_dims
        self.running  = False
        self._model_name = model_name
        self._dataset_name = dataset_name

        # Components
        self._reconstructor = WeightReconstructor(decompositions, projections, device)

        self._hooks = HookManager()
        if feature_layer:
            self._hooks.register_spatial_hook(model, feature_layer)
        self._hooks.register_pooled_hooks(model, self._reconstructor.layer_keys)

        self._evaluator = Evaluator(
            model=model,
            dataloader=dataloader,
            task=task,
            dataset_name=dataset_name,
            num_classes=num_classes,
            ignore_index=ignore_index,
            loss_fn=loss_fn,
            amp=amp,
            feature_layer=feature_layer,
            feature_buffer_fn=lambda: self._hooks.feature_buffer,
        )

        self._scorer = ConceptScorer()
        self._mapper = ActivationMapper(self._hooks)

        self.ipc: IIPC = NngIPC()

        # Queues
        # TODO: maybe expose maxsize as parameter 
        self._job_queue:  queue.Queue = queue.Queue(maxsize=50)
        self._send_queue: queue.Queue = queue.Queue()

        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="inference-worker"
        )

        # Cached eval labels for concept scoring 
        self._eval_labels: torch.Tensor | None = None

        # Vector list dispatch table
        self._vector_list_handlers = {
            VectorListID.CONCEPT_SELECT:  self._on_concept_select,
            VectorListID.ACTIVATION_REQ:  self._on_activation_req,
        }

        # Worker job dispatch table
        self._job_handlers = {
            self._JobType.POINT_CLOUD_JOB:        self._handle_full_cloud,
            self._JobType.TRANSFORMED_POINTS_JOB: self._handle_transform,
            self._JobType.CONCEPT_SELECT_JOB:     self._handle_concept,
            self._JobType.ACTIVATION_REQUEST_JOB: self._handle_activation,
            self._JobType.RAW_IMAGES_JOB:         self._handle_raw_images,
            self._JobType.CKA_LAYOUT_JOB:         self._handle_cka_layout,
            self._JobType.TRAJECTORY_GEOMETRY_JOB: self._handle_trajectory_geometry_layout,
            self._JobType.SENSITIVITY_LAYOUT_JOB:  self._handle_sensitivity_layout,
            self._JobType.EV_LAYOUT_JOB:           self._handle_ev_layout,
            self._JobType.CORRELATION_LAYOUT_JOB:  self._handle_correlation_layout,
        }

        # CSV file paths for the layouts that need them
        self._trajectory_geometry_csv: str | None = trajectory_geometry_csv
        self._sensitivity_csv: str | None = sensitivity_csv
        self._variance_explained_csv: str | None = variance_explained_csv

        # Instead of refitting the PCA basis from whatever the current feature distribution 
        # is after each weight manipulation, we compute it once in the beginning and use that. 
        # PCA has sign and rotation ambiguity, the axes can flip or rotate between calls even on similar data. 
        # So as points are dragged in the weight space and the model's features shift slightly, 
        # the entire feature viewer coordinate system lurches. 
        # This makes it hard to track how features move because the space itself is moving underneath them.
        # fit the PCA basis once on the initial unmodified model's features, then reuse that fixed basis for all subsequent projections.
        self._feature_pca_basis: torch.Tensor | None = None

    def start(self) -> bool:
        if not self.ipc.bind(self._address):
            print(f"[Engine] Failed to bind to {self._address}")
            return False

        self.ipc.on_ack                  = self._on_ack
        self.ipc.on_transformed_points   = self._on_transformed_points
        self.ipc.on_point_cloud          = self._on_point_cloud
        self.ipc.on_vector_list          = self._on_vector_list
        self.ipc.on_transformed_points_batch = self._on_transformed_points_batch

        self._model.eval()
        self.running = True
        self._worker_thread.start()
        print(f"[Engine] Bound to {self._address}, waiting for renderer...")
        return True

    def stop(self) -> None:
        self.running = False
        self.ipc.close()
        self._hooks.clear()
        self._worker_thread.join(timeout=5.0)
        print("[Engine] Stopped")

    def run(self) -> None:
        """IPC thread main loop."""
        while self.running:
            self._drain_send_queue()
            self.ipc.poll(timeout_ms=100)

    # IPC callbacks (IPC thread)

    def _on_ack(self) -> None:
        print("[Engine] Renderer connected. Sending initial cloud")
        self._reconstructor.reset()
        self._drain_job_queue()

        # Drained jobs never report back, so tell the renderer to forget them
        self._send_job_status(0, JobStatus.RESET)
        self._feature_pca_basis = None

        #We send the depth layout on ack
        depth_offsets = compute_depth_layout(
            self._reconstructor.layer_keys,
            self._reconstructor.layer_types,
        )
        depth_msg = LayoutMsg(LayoutID.DEPTH, 
                              "Depth / Topology", 
                              depth_offsets, 
                              axis_labels=("Layer Depth", "Within-block Index", "Layer Type")
        )

        self._send_queue.put(lambda m=depth_msg: self.ipc.send_layout(m))

        # Send layer meta info
        self.ipc.send_layer_meta(self._reconstructor.build_layer_meta(self._evaluator.num_classes))

        # Send the actual cloud
        self.ipc.send_point_cloud(self._reconstructor.build_point_cloud())

        # We send raw images and cka layout in a non-blocking way
        self._enqueue_job(self._JobType.RAW_IMAGES_JOB, None)
        self._enqueue_job(self._JobType.CKA_LAYOUT_JOB, None)

        # Send other layouts if the files are present
        self._enqueue_job(self._JobType.TRAJECTORY_GEOMETRY_JOB, None)
        self._enqueue_job(self._JobType.SENSITIVITY_LAYOUT_JOB, None)
        self._enqueue_job(self._JobType.EV_LAYOUT_JOB, None)
        self._enqueue_job(self._JobType.CORRELATION_LAYOUT_JOB, None)

        print("[Worker] ON ACK")

    def _on_transformed_points(self, msg: TransformedPointsMsg) -> None:
        print(f"[Engine] Transformed {len(msg.indices)} pts on cloud {msg.cloud_id}")
        self._enqueue_job(self._JobType.TRANSFORMED_POINTS_JOB, msg)

    def _on_transformed_points_batch(self, msg: TransformedPointsBatchMsg) -> None:
        print(f"[Engine] Transform batch: {len(msg.clouds)} cloud(s), "
            f"{sum(len(c.indices) for c in msg.clouds)} pts total")
        self._enqueue_job(self._JobType.TRANSFORMED_POINTS_JOB, msg)

    def _on_point_cloud(self, msg: PointCloudMsg) -> None:
        print(f"[Engine] Full cloud received: {len(msg.points)} pts")
        self._reconstructor.apply_full_cloud(msg)
        self._enqueue_job(self._JobType.POINT_CLOUD_JOB, None)

    def _on_vector_list(self, msg: VectorListMsg) -> None:
        handler = self._vector_list_handlers.get(msg.id)
        if handler:
            handler(msg)
        else:
            print(f"[Engine] Unhandled VectorList id={msg.id}")

    def _on_concept_select(self, msg: VectorListMsg) -> None:
        class_idx = int(msg.data[0, 0])
        self._enqueue_job(self._JobType.CONCEPT_SELECT_JOB, class_idx)

    def _on_activation_req(self, msg: VectorListMsg) -> None:
        self._enqueue_job(self._JobType.ACTIVATION_REQUEST_JOB, msg)

    # Worker thread 

    def _worker_loop(self) -> None:
        while self.running:
            try:
                job_type, payload = self._job_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                # Handlers returned SKIPPED when they had nothing to send, None otherwise
                status = self._job_handlers[job_type](payload) or JobStatus.DONE
            except Exception as e:
                import traceback
                print(f"[Worker] Error in job {job_type.name}: {e}")
                traceback.print_exc()
                status = JobStatus.FAILED

            self._send_job_status(job_type, status)

    # Job handlers (worker thread)

    def _handle_transform(self, msg: TransformedPointsMsg | TransformedPointsBatchMsg) -> None:
        if isinstance(msg, TransformedPointsMsg):
            self._reconstructor.apply_point_updates(msg)
        else:
            # Apply all clouds from this gizmo frame before reconstructing
            for cloud in msg.clouds:
                self._reconstructor.apply_point_updates(cloud)
        # One reconstruct + one eval for the whole frame
        self._reconstructor.reconstruct_and_load(self._model)
        self._run_eval_and_enqueue()

    def _handle_full_cloud(self, _) -> None:
        self._reconstructor.reconstruct_and_load(self._model)
        self._run_eval_and_enqueue()

    def _handle_concept(self, class_idx: int) -> JobStatus | None:
        scores = self._scorer.score(
            pooled_buffers=self._hooks.pooled_buffers,
            labels=self._eval_labels,
            layer_keys=self._reconstructor.layer_keys,
            pca_dict=self._reconstructor._decompositions,
            class_idx=class_idx,
        )
        if scores.size == 0:
            return JobStatus.SKIPPED

        msg = VectorListMsg(
            id=VectorListID.CONCEPT_SCORES,
            dimensions=1,
            data=scores.reshape(-1, 1),
        )
        self._send_queue.put(lambda m=msg: self.ipc.send_vector_list(m))
        print(f"[Worker] Concept scores sent for class {class_idx} "
              f"({len(scores)} points)")

    def _handle_activation(self, msg: VectorListMsg) -> JobStatus | None:
        rows = [(int(msg.data[i, 0]), int(msg.data[i, 1])) for i in range(msg.count)]
        agg  = int(msg.data[0, 2]) if msg.count > 0 else 0

        by_layer: dict[int, list[int]] = {}
        for ci, pi in rows:
            if ci < len(self._reconstructor.layer_keys):
                by_layer.setdefault(ci, []).append(pi)

        if not by_layer:
            print("[Worker] Activation request: all cloud_idx out of range")
            return JobStatus.SKIPPED

        input_h, input_w = self._evaluator.get_input_resolution()

        batch_msg = self._mapper.compute(
            model=self._model,
            dataloader=self._evaluator._dataloader,
            forward_fn=self._model, #TODO: or self._evaluator.forward
            by_layer=by_layer,
            layer_keys=self._reconstructor.layer_keys,
            pca_dict=self._reconstructor._decompositions,
            aggregation_mode=agg,
            input_h=input_h,
            input_w=input_w,
        )
        if batch_msg is not None:
            self._send_queue.put(lambda m=batch_msg: self.ipc.send_images(m))
            agg_name = "average" if agg == 0 else "maximum"
            print(f"[Worker] Activation maps sent, {len(rows)} filter(s), "
                  f"{agg_name}, {len(batch_msg.images)} samples "
                  f"at {input_w}x{input_h}")
        else:
            return JobStatus.SKIPPED

    def _handle_raw_images(self, _) -> None:
        batch = self._evaluator.build_raw_images()
        self._send_queue.put(lambda m=batch: self.ipc.send_images(m))

    def _handle_cka_layout(self, _) -> None:
        self._hooks.clear_pooled_buffers()
        result = self._evaluator.run()
        self._eval_labels = result.labels

        offsets = compute_cka_layout(
            self._hooks.pooled_buffers,
            self._reconstructor.layer_keys,
        )
        
        msg = LayoutMsg(LayoutID.CKA, 
                        "CKA Similarity", 
                        offsets,
                        axis_labels=("X", "Y", "Z"),
        )
        self._send_queue.put(lambda m=msg: self.ipc.send_layout(m))
        print("[Worker] CKA layout sent")

    def _handle_trajectory_geometry_layout(self, _) -> JobStatus:
        if self._trajectory_geometry_csv:
            offsets = compute_trajectory_geometry_layout(
                self._model_name.replace(" ", "").replace("_", "").lower(),
                self._reconstructor.layer_keys,
                self._reconstructor.layer_types,
                self._trajectory_geometry_csv,
                self._dataset_name,
            )
            msg = LayoutMsg(LayoutID.TRAJECTORY_GEOMETRY, 
                            "Trajectory Geometry", 
                            offsets,
                            axis_labels=("Directionality", "Displacement (norm.)", "Layer Depth Index"),
            )
            self._send_queue.put(lambda m=msg: self.ipc.send_layout(m))
            print("[Worker] Trajectory geometry layout sent")
        else:
            print("[Worker] Trajectory geometry has no CSV file! No OP.")
            return JobStatus.SKIPPED

    def _handle_sensitivity_layout(self, _) -> JobStatus | None:
        if self._sensitivity_csv:
            offsets = compute_sensitivity_layout(
                self._model_name.replace(" ", "").replace("_", "").lower(),
                self._reconstructor.layer_keys,
                self._reconstructor.layer_types,
                self._sensitivity_csv,
                self._dataset_name,
            )
            msg = LayoutMsg(LayoutID.SENSITIVITY, 
                            "Sensitivity", 
                            offsets,
                            axis_labels=("Mean Sensitivity", "Layer Type", "Layer Depth Index"),
            )
            self._send_queue.put(lambda m=msg: self.ipc.send_layout(m))
            print("[Worker] Sensitivity layout sent")
        else:
            print("[Worker] Sensitivity has no CSV file! No OP.")
            return JobStatus.SKIPPED

    def _handle_ev_layout(self, _) -> JobStatus | None:
        if self._variance_explained_csv:
            offsets = compute_explained_variance_layout(
                self._model_name.replace(" ", "").replace("_", "").lower(),
                self._reconstructor.layer_keys,
                self._reconstructor.layer_types,
                self._variance_explained_csv,
                self._dataset_name,
            )
            msg = LayoutMsg(LayoutID.EXPLAINED_VARIANCE, 
                            "Explained Variance", 
                            offsets,
                            axis_labels=("Explained Variance", "Layer Type", "Layer Depth Index"),
            )
            self._send_queue.put(lambda m=msg: self.ipc.send_layout(m))
            print("[Worker] Explained variance layout sent")
        else:
            print("[Worker] Variance explained has no CSV file! No OP.")
            return JobStatus.SKIPPED

    def _handle_correlation_layout(self, _) -> JobStatus | None:
        if self._trajectory_geometry_csv and self._sensitivity_csv:
            offsets = compute_correlation_layout(
                self._model_name.replace(" ", "").replace("_", "").lower(),
                self._reconstructor.layer_keys,
                self._trajectory_geometry_csv,
                self._sensitivity_csv,
                self._dataset_name,
            )
            msg = LayoutMsg(LayoutID.CORRELATION, 
                            "Correlation", 
                            offsets,
                            axis_labels=("Directionality", "Displacement", "Sensitivity"),
            )
            self._send_queue.put(lambda m=msg: self.ipc.send_layout(m))
            print("[Worker] Correlation layout sent")
        else:
            print("[Worker] Trajectory geometry or sensitivity has no CSV file! No OP.")
            return JobStatus.SKIPPED

    # Eval helper

    def _run_eval_and_enqueue(self) -> None:
        self._hooks.clear_pooled_buffers()
        result = self._evaluator.run()
        self._eval_labels = result.labels

        # Project features if requested
        features = result.features
        if self._pca_target_dims > 0:
            if self._feature_pca_basis is None:
                # Fit once on the original model's features
                _, _, V = torch.pca_lowrank(features, q=self._pca_target_dims)
                self._feature_pca_basis = V   # [D, pca_target_dims]
                print(f"[Engine] Feature PCA basis fitted: "
                    f"{features.shape[1]}D -> {self._pca_target_dims}D")
            features = features @ self._feature_pca_basis

        feat_msg = VectorListMsg(
            id=VectorListID.FEATURES,
            dimensions=features.shape[1],
            data=features.numpy(),
        )
        metrics_msg = MetricsMsg.from_dict({**result.metrics, "loss": result.loss})

        self._send_queue.put(lambda m=metrics_msg: self.ipc.send_metrics(m))
        self._send_queue.put(lambda l=result.loss: self.ipc.send_loss(l))
        self._send_queue.put(lambda m=feat_msg: self.ipc.send_vector_list(m))
        self._send_queue.put(lambda m=result.images: self.ipc.send_images(m))

    # Queue helpers

    def _enqueue_job(self, job_type, payload) -> None:
        # QUEUED goes out before the job is visible to the worker, so it always reaches the renderer before the job's final status
        self._send_job_status(job_type, JobStatus.QUEUED)

        try:
            self._job_queue.put_nowait((job_type, payload))
        except queue.Full:
            # Evict te oldest job to make room and report it as DROPPED
            try:
                dropped_type, _ = self._job_queue.get_nowait()
                self._send_job_status(dropped_type, JobStatus.DROPPED)
            except queue.Empty: pass
            try: self._job_queue.put_nowait((job_type, payload))
            except queue.Full:
                self._send_job_status(job_type, JobStatus.DROPPED)

    def _send_job_status(self, job_type: int, status: JobStatus) -> None:
        msg = VectorListMsg(
            id = VectorListID.JOB_STATUS,
            dimensions = 2,
            data=[[int(job_type), int(status)]],
        )
        self._send_queue.put(lambda m = msg: self.ipc.send_vector_list(m))

    def _drain_job_queue(self) -> None:
        while not self._job_queue.empty():
            try:    self._job_queue.get_nowait()
            except queue.Empty: break

    def _drain_send_queue(self) -> None:
        while not self._send_queue.empty():
            try:
                fn = self._send_queue.get_nowait()
                fn()
            except queue.Empty:
                break
