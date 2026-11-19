"""
IPC interface and message types for communication with C++ renderer.

This module provides:
- Message type definitions
- Message dataclasses for structured data
- Abstract IPC interface for transport implementations
- Serialization/deserialization utilities
"""

import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable

import numpy as np


# Message Types

class MsgType(IntEnum):
    """Message type identifiers (must match C++ enum)."""
    POINT_CLOUD = 1
    TRANSFORMED_POINTS = 2
    TRANSFORMED_POINTS_BATCH = 3
    VECTOR_LIST = 4
    METRICS = 5
    IMAGE = 6
    IMAGE_BATCH = 7
    LAYER_META = 8
    LAYOUT = 9
    ACK = 255


class VectorListID:
    """Well-known vector list IDs (must match C++ constants)."""
    FEATURES = 1
    LOSS = 2
    GRADIENTS = 3
    CONCEPT_SELECT = 4
    CONCEPT_SCORES = 5
    ACTIVATION_REQ = 6
    JOB_STATUS = 255

class JobStatus(IntEnum):
    """
    Lifecycle of a backend job, sent as JOB_STATUS (must match C++ ipc::JobStatus).

    Every QUEUED is followed by exactly one of DONE / FAILED / SKIPPED / DROPPED,
    so the renderer can count pending jobs per type without request IDs.
    """
    QUEUED = 1
    DONE = 2
    FAILED = 3
    SKIPPED = 4
    DROPPED = 5
    RESET = 6

class LayoutID:
    """Stables IDs for layout modes (must match C++ LayoutID namespace)"""
    DEPTH = 0
    CKA = 1
    TRAJECTORY_GEOMETRY = 2
    SENSITIVITY = 3
    EXPLAINED_VARIANCE = 4
    CORRELATION = 5

class ImageFormat(IntEnum):
    """Image format identifiers (must match C++ identifiers)"""
    GRAYSCALE = 1
    RGB = 2
    BGR = 3
    RGBA = 4
    BGRA = 5

class ImageBatchType(IntEnum):
    PREDICTIONS    = 0   # blended prediction overlays, sent after each eval
    ACTIVATION_MAP = 1   # per-filter spatial heatmaps, sent on activation request
    RAW_IMAGES     = 2   # raw input images, sent once on connect

# Message Dataclasses

@dataclass
class PointCloudMsg:
    """A point cloud with optional colors and normals."""
    cloud_id: int
    points: np.ndarray  # Shape: (N, 3), dtype: float32
    colors: np.ndarray | None = None  # Shape: (N, 3), dtype: float32
    normals: np.ndarray | None = None  # Shape: (N, 3), dtype: float32
    
    def __post_init__(self):
        # Ensure proper dtype and contiguity
        self.points = np.ascontiguousarray(self.points, dtype=np.float32)
        if self.colors is not None:
            self.colors = np.ascontiguousarray(self.colors, dtype=np.float32)
        if self.normals is not None:
            self.normals = np.ascontiguousarray(self.normals, dtype=np.float32)


@dataclass
class TransformedPointsMsg:
    """A subset of points with updated positions."""
    cloud_id: int
    indices: np.ndarray  # Shape: (M,), dtype: uint32
    positions: np.ndarray  # Shape: (M, 3), dtype: float32
    
    def __post_init__(self):
        self.indices = np.ascontiguousarray(self.indices, dtype=np.uint32)
        self.positions = np.ascontiguousarray(self.positions, dtype=np.float32)

@dataclass
class TransformedPointsBatchMsg:
    """One gizmo frame: transformed points across all affected clouds.
    Single-cloud transforms are a batch of size 1."""
    clouds: list[TransformedPointsMsg]

@dataclass
class VectorListMsg:
    """N-dimensional vectors (features, loss, gradients, etc.)."""
    id: int  # Identifies what this represents
    dimensions: int
    data: np.ndarray  # Shape: (count, dimensions), dtype: float32
    
    def __post_init__(self):
        self.data = np.ascontiguousarray(self.data, dtype=np.float32)
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, self.dimensions)
    
    @property
    def count(self) -> int:
        return len(self.data)
    
    @classmethod
    def from_loss(cls, loss: float) -> 'VectorListMsg':
        """Create a VectorListMsg containing a single loss value."""
        return cls(
            id=VectorListID.LOSS,
            dimensions=1,
            data=np.array([[loss]], dtype=np.float32)
        )
    
    @classmethod
    def from_features(cls, features: np.ndarray) -> 'VectorListMsg':
        """Create a VectorListMsg containing feature vectors."""
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return cls(
            id=VectorListID.FEATURES,
            dimensions=features.shape[1],
            data=features
        )
    
    @classmethod
    def from_concept_select(cls, concept: np.uint32)-> 'VectorListMsg':
        """Create a VectorListMsg containing a single score value."""
        return cls(
            id=VectorListID.CONCEPT_SELECT,
            dimensions=1,
            data=np.array([[concept]], dtype=np.uint32)
        )
    
    @property
    def value(self) -> float:
        """Convenience property for single-value messages (e.g., loss)."""
        return float(self.data[0, 0]) if self.data.size > 0 else 0.0


@dataclass
class MetricsMsg:
    """Named float metrics (e.g., accuracy, loss, f1_score)."""
    names: list[str]
    values: np.ndarray  # Shape: (count,), dtype: float32
    
    def __post_init__(self):
        self.values = np.ascontiguousarray(self.values, dtype=np.float32)
    
    @property
    def count(self) -> int:
        return len(self.names)
    
    def get(self, name: str, default: float = 0.0) -> float:
        """Get metric value by name."""
        try:
            idx = self.names.index(name)
            return float(self.values[idx])
        except ValueError:
            return default
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {name: float(val) for name, val in zip(self.names, self.values)}
    
    @classmethod
    def from_dict(cls, metrics: dict) -> 'MetricsMsg':
        """Create from dictionary."""
        return cls(names=list(metrics.keys()), values=np.array(list(metrics.values()), dtype=np.float32))
    
    @classmethod
    def from_lists(cls, keys: list[str], values: list[float] | np.ndarray) -> 'MetricsMsg':
        """Create from sequential containers (e.g. lists or arrays)"""
        return cls(names=keys, values=np.array(values, dtype=np.float32))

@dataclass
class ImageMsg:
    image_id: int
    image: np.ndarray
    format: ImageFormat = ImageFormat.BGR
    name: str = ""

    def __post_init__(self):
        self.image = np.ascontiguousarray(self.image, dtype=np.uint8)

    # Ensure contiguous array if image is mutated
    def __setattr__(self, name: str, value):
        if name == "image":
            value = np.ascontiguousarray(value, dtype=np.uint8)
        super().__setattr__(name, value)


@dataclass
class ImageBatchMsg:
    # Struct of array approach not necessary here because I am accessing all the fields of ImageMsg anyway, so there are no real cache misses
    batch_type: ImageBatchType
    images: list[ImageMsg] = field(default_factory=list)

@dataclass
class LayerMetaMsg:
    cloud_id: int
    layer_keys: list[str]
    point_counts: np.ndarray
    colors: np.ndarray | None
    layer_types: list[str]
    num_classes: int = 0
    
    def __post_init__(self):
        if self.colors is not None:
            self.colors = np.ascontiguousarray(self.colors, dtype=np.float32)
        self.point_counts = np.ascontiguousarray(self.point_counts, dtype=np.uint32)
    
    @property
    def count(self) -> int:
        return len(self.layer_keys)
    
    @classmethod
    def from_lists(cls, cloud_id: int, layer_keys: list[str], 
                   point_counts: list[float] | np.ndarray, 
                   colors: list[float] | np.ndarray | None,
                   layer_types: list[str], 
                   num_classes: int = 0) -> 'LayerMetaMsg':
        """Create from sequential containers (e.g. lists or arrays)"""

        # a color is rbg -> 3 floats. So |colors| = |layer_keys| * 3
        if colors is not None:
            if len(colors) == 3 * len(layer_keys):
                return cls(cloud_id=cloud_id, layer_keys=layer_keys, 
                           point_counts=np.array(point_counts, dtype=np.uint32), 
                           colors=np.array(colors, dtype=np.float32), 
                           num_classes=num_classes, 
                           layer_types=layer_types
                )
            else:
                raise ValueError("Colors is a flattened array with rgb values/vectors: len(colors) == 3 * len(layer_keys)")

        #if no colors are given, color is just black
        colors = np.array([0.0] * len(layer_keys) * 3, dtype=np.float32)
        return cls(cloud_id=cloud_id, 
                   layer_keys=layer_keys, 
                   point_counts=np.array(point_counts, dtype=np.uint32), 
                   colors=colors, 
                   num_classes=num_classes, 
                   layer_types=layer_types
        )

@dataclass
class LayoutMsg:
    layout_id: int # LayoutID constant
    name: str # Label shown in the renderer UI
    offsets: np.ndarray # float32 [L, 3]
    axis_labels: tuple[str, str, str] = ("X", "Y", "Z")  # default

    def __post_init__(self):
        self.offsets = np.ascontiguousarray(self.offsets, dtype=np.float32)
        assert self.offsets.ndim == 2 and self.offsets.shape[1] == 3


# Serialization

def serialize_point_cloud(msg: PointCloudMsg) -> bytes:
    """
    Serialize a point cloud message.
    
    Wire format:
    [type:u8][cloud_id:u32][flags:u8][num_points:u32][points...][colors?...][normals?...]
    """
    flags = 0
    if msg.colors is not None and len(msg.colors) > 0:
        flags |= 0x01
    if msg.normals is not None and len(msg.normals) > 0:
        flags |= 0x02
    
    header = struct.pack('<BIBI',
        MsgType.POINT_CLOUD,
        msg.cloud_id,
        flags,
        len(msg.points)
    )
    
    data = header + msg.points.tobytes()
    if flags & 0x01:
        data += msg.colors.tobytes()
    if flags & 0x02:
        data += msg.normals.tobytes()
    
    return data

def serialize_layer_meta(msg: LayerMetaMsg) -> bytes:
    """
    Serialize a layer meta message.

    Wire format:
    [type:u8][cloud_id:u32][num_layers:u32][point_counts: u32 x num_layers]
    [colors: f32 x num_layers*3]          -- interleaved RGB, matches glm::vec3 layout
    [str_total_len:u32][names: null-terminated, concatenated]
    [str_total_len:u32][layer_types: null-terminated, concatenated]
    """
    header = struct.pack('<BIII',
        MsgType.LAYER_META,
        msg.cloud_id,
        msg.count,
        msg.num_classes,
    )

    counts_bytes = msg.point_counts.tobytes()
    colors_bytes = msg.colors.tobytes()

    # Null-terminated names concatenated 
    names_bytes = b''.join(k.encode('utf-8') + b'\x00' for k in msg.layer_keys)
    names_len = struct.pack('<I', len(names_bytes))

    types_bytes = b''.join(t.encode('utf-8') + b'\x00' for t in msg.layer_types)
    types_len   = struct.pack('<I', len(types_bytes))

    return (header 
            + counts_bytes 
            + colors_bytes 
            + names_len + names_bytes 
            + types_len + types_bytes)

def deserialize_point_cloud(payload: bytes) -> PointCloudMsg:
    """Deserialize a point cloud message (payload without type byte)."""
    cloud_id, flags, num_points = struct.unpack('<IBI', payload[:9])
    offset = 9
    
    points_size = num_points * 12  # 3 floats * 4 bytes
    points = np.frombuffer(payload[offset:offset + points_size], dtype=np.float32).reshape(-1, 3).copy()
    offset += points_size
    
    colors = None
    if flags & 0x01:
        colors = np.frombuffer(payload[offset:offset + points_size], dtype=np.float32).reshape(-1, 3).copy()
        offset += points_size
    
    normals = None
    if flags & 0x02:
        normals = np.frombuffer(payload[offset:offset + points_size], dtype=np.float32).reshape(-1, 3).copy()
    
    return PointCloudMsg(cloud_id, points, colors, normals)


def serialize_transformed_points(msg: TransformedPointsMsg) -> bytes:
    """
    Serialize a transformed points message.
    
    Wire format:
    [type:u8][cloud_id:u32][count:u32][indices...][positions...]
    """
    header = struct.pack('<BII',
        MsgType.TRANSFORMED_POINTS,
        msg.cloud_id,
        len(msg.indices)
    )

    return header + msg.indices.tobytes() + msg.positions.tobytes()


def serialize_transformed_points_batch(msg: TransformedPointsBatchMsg) -> bytes:
    """
    Serialize a batch of transformed points message.

    Wire format:
    [type:u8][num_clouds:u32]
        [cloud_id:u32][count:u32][indices: u32*count][positions: f32x3*count]
        [cloud_id:u32][count:u32][indices: u32*count][positions: f32x3*count]
        ...
    """
    header = struct.pack('<BI', MsgType.TRANSFORMED_POINTS, len(msg.clouds))
    parts  = [header]

    for cloud in msg.clouds:
        parts.append(struct.pack('<II', cloud.cloud_id, len(cloud.indices)))
        parts.append(cloud.indices.tobytes())
        parts.append(cloud.positions.tobytes())

    return b''.join(parts)


def deserialize_transformed_points(payload: bytes) -> TransformedPointsMsg:
    """Deserialize a transformed points message (payload without type byte)."""
    cloud_id, count = struct.unpack('<II', payload[:8])
    offset = 8
    
    indices = np.frombuffer(payload[offset:offset + count * 4], dtype=np.uint32).copy()
    offset += count * 4
    
    positions = np.frombuffer(payload[offset:offset + count * 12], dtype=np.float32).reshape(-1, 3).copy()
    
    return TransformedPointsMsg(cloud_id, indices, positions)

def deserialize_transformed_points_batch(payload: bytes) -> TransformedPointsBatchMsg:
    num_clouds, = struct.unpack('<I', payload[:4])
    offset = 4
    clouds = []

    for _ in range(num_clouds):
        cloud_id, count = struct.unpack('<II', payload[offset:offset + 8])
        offset += 8
        indices   = np.frombuffer(payload[offset:offset + count * 4],
                                  dtype=np.uint32).copy()
        offset += count * 4
        positions = np.frombuffer(payload[offset:offset + count * 12],
                                  dtype=np.float32).reshape(-1, 3).copy()
        offset += count * 12
        clouds.append(TransformedPointsMsg(cloud_id, indices, positions))
    return TransformedPointsBatchMsg(clouds)

def serialize_vector_list(msg: VectorListMsg) -> bytes:
    """
    Serialize a vector list message.
    
    Wire format:
    [type:u8][id:u32][dimensions:u32][count:u32][data...]
    """
    flat_data = msg.data.flatten()
    header = struct.pack('<BIII',
        MsgType.VECTOR_LIST,
        msg.id,
        msg.dimensions,
        msg.count
    )
    return header + flat_data.tobytes()


def serialize_loss(loss: float) -> bytes:
    """Convenience function to serialize a single loss value."""
    return serialize_vector_list(VectorListMsg.from_loss(loss))


def deserialize_vector_list(payload: bytes) -> VectorListMsg:
    """Deserialize a vector list message (payload without type byte)."""
    id_, dimensions, count = struct.unpack('<III', payload[:12])
    data = np.frombuffer(payload[12:], dtype=np.float32).reshape(count, dimensions).copy()
    return VectorListMsg(id_, dimensions, data)


def serialize_metrics(msg: MetricsMsg) -> bytes:
    """
    Serialize a metrics message.
    
    Wire format:
    [type:u8][count:u32][str_total_len:u32][null_terminated_names...][values...]
    """
    # Build null-terminated string blob
    names_blob = b''.join(name.encode('utf-8') + b'\x00' for name in msg.names)
    
    header = struct.pack('<BII',
        MsgType.METRICS,
        msg.count,
        len(names_blob)
    )
    return header + names_blob + msg.values.tobytes()


def deserialize_metrics(payload: bytes) -> MetricsMsg:
    """Deserialize a metrics message (payload without type byte)."""
    count, str_total_len = struct.unpack('<II', payload[:8])
    offset = 8
    
    # Parse null-terminated strings
    names_blob = payload[offset:offset + str_total_len]
    names = []
    offset_in_blob = 0
    for _ in range(count):
        end = names_blob.index(b'\x00', offset_in_blob)
        names.append(names_blob[offset_in_blob:end].decode('utf-8'))
        offset_in_blob = end + 1
    
    values_start = 8 + str_total_len
    values = np.frombuffer(payload[values_start:offset + count * 4], dtype=np.float32).copy()
    
    return MetricsMsg(names, values)

def _serialize_image_data(msg: ImageMsg) -> bytes:
    """
    Serialize image data (without MsgType byte).
    Wire format:
    [id:u32][width:u32][height:u32][channels:u8][format:u8][name: null-terminated][data]
    """
    img = np.ascontiguousarray(msg.image, dtype=np.uint8)
    if img.ndim == 2:
        height, width = img.shape
        channels = 1
    else:
        height, width, channels = img.shape

    header = struct.pack('<IIIB B',
        msg.image_id,
        width,
        height,
        channels,
        msg.format
    )

    # Null-terminated name of arbitrary length.
    # No truncation: layer keys can be long PyTorch module paths like
    # "encoder.layer3.blocks.2.attn.query.weight".
    name_bytes = msg.name.encode('utf-8') + b'\x00'

    return header + name_bytes + img.tobytes()


def serialize_image(msg: ImageMsg) -> bytes:
    """
    Serialize one image message.
    Wire format:
    [type:u8][id:u32][width:u32][height:u32][channels:u8][format:u8][name: null-terminated][data]
    """
    return struct.pack('<B', MsgType.IMAGE) + _serialize_image_data(msg)


def serialize_image_batch(msg: ImageBatchMsg) -> bytes:
    """
    Serialize multiple images in one message.
    Wire format:
    [type:u8][batch_id:u32][count:u32]
        [id:u32][width:u32][height:u32][channels:u8][format:u8][name: null-terminated][data]
        [id:u32][width:u32][height:u32][channels:u8][format:u8][name: null-terminated][data]
        ...
    """
    header = struct.pack('<BII', MsgType.IMAGE_BATCH, msg.batch_type, len(msg.images))
    parts = [header]
    
    for img_msg in msg.images:
        parts.append(_serialize_image_data(img_msg))
    
    return b''.join(parts)

def serialize_layout(msg: LayoutMsg) -> bytes:
    """
    Wire format:
    [type:u8][layout_id:u32][num_layers:u32][name: null-terminated][offsets: f32x3*L]
    [axis_x: null_terminated][axis_y: null_terminated][axis_z: null_terminated]
    """
    name_bytes = msg.name.encode('utf-8') + b'\x00'
    axis_x_bytes = msg.axis_labels[0].encode('utf-8') + b'\x00'
    axis_y_bytes = msg.axis_labels[1].encode('utf-8') + b'\x00'
    axis_z_bytes = msg.axis_labels[2].encode('utf-8') + b'\x00'

    header = struct.pack('<BII',
                         MsgType.LAYOUT,
                         msg.layout_id,
                         len(msg.offsets),
    )

    return (header
            + name_bytes
            + msg.offsets.tobytes()
            + axis_x_bytes
            + axis_y_bytes
            + axis_z_bytes)

def serialize_ack() -> bytes:
    """Serialize an acknowledgment message."""
    return struct.pack('<B', MsgType.ACK)


# Abstract IPC Interface

class IIPC(ABC):
    """
    Abstract IPC interface for communication with C++ renderer.
    
    Implementations:
    - NngIPC: Uses NNG over TCP
    - Future: ShmIPC for shared memory
    
    Usage (Backend as Server):
    
        ipc = NngIPC()
        ipc.bind("tcp://*:9999")
        
        # Set up callbacks
        ipc.on_transformed_points = lambda msg: process_points(msg)
        
        # Send initial point cloud
        ipc.send_point_cloud(PointCloudMsg(...))
        
        # Main loop
        while running:
            ipc.poll()
            # ... do work ...
    """
    
    # Callbacks (set these to handle incoming messages)
    on_point_cloud: Callable[[PointCloudMsg], None] | None = None
    on_transformed_points: Callable[[TransformedPointsMsg], None] | None = None
    on_transformed_points_batch: Callable[[TransformedPointsBatchMsg], None] | None = None
    on_vector_list: Callable[[VectorListMsg], None] | None = None
    on_metrics: Callable[[MetricsMsg], None] | None = None
    on_ack: Callable[[], None] | None = None
    
    # Connection
    
    @abstractmethod
    def bind(self, address: str) -> bool:
        """Server mode: listen for incoming connections."""
        pass
    
    @abstractmethod
    def connect(self, address: str) -> bool:
        """Client mode: connect to a server."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the connection."""
        pass
    
    @abstractmethod
    def is_open(self) -> bool:
        """Check if connection is open."""
        pass
    
    # Send methods
    
    @abstractmethod
    def send_point_cloud(self, msg: PointCloudMsg) -> bool:
        """Send a point cloud message."""
        pass
    
    @abstractmethod
    def send_layer_meta(self, msg: LayerMetaMsg) -> bool:
        """Send a layer meta message."""
        pass

    @abstractmethod
    def send_transformed_points(self, msg: TransformedPointsMsg) -> bool:
        """Send transformed points message."""
        pass

    @abstractmethod
    def send_transformed_points_batch(self, msg: TransformedPointsBatchMsg) -> bool:
        """Send a batch of transformed points message."""
        pass

    @abstractmethod
    def send_vector_list(self, msg: VectorListMsg) -> bool:
        """Send a vector list message."""
        pass
    
    def send_loss(self, loss: float) -> bool:
        """Send a single loss value."""
        return self.send_vector_list(VectorListMsg.from_loss(loss))
    
    def send_features(self, features: np.ndarray) -> bool:
        """Send feature vectors."""
        return self.send_vector_list(VectorListMsg.from_features(features))
    
    @abstractmethod
    def send_metrics(self, msg: MetricsMsg) -> bool:
        """Send a metrics message."""
        pass
    
    def send_metrics_dict(self, metrics: dict) -> bool:
        """Send metrics from a dictionary."""
        return self.send_metrics(MetricsMsg.from_dict(metrics))

    @abstractmethod
    def send_image(self, msg:ImageMsg) -> bool:
        """Send an image"""
        pass

    @abstractmethod 
    def send_images(self, msg:ImageBatchMsg) -> bool:
        """Send multiple images in one batch"""
        pass

    @abstractmethod
    def send_layout(self, msg:LayoutMsg) -> bool:
        """Send layout offsets for one layout mode"""
        pass

    @abstractmethod
    def send_ack(self) -> bool:
        """Send an acknowledgment."""
        pass

    
    # Receive methods
    
    @abstractmethod
    def poll(self, timeout_ms: int = 0) -> int:
        """
        Process incoming messages (non-blocking by default).
        
        Args:
            timeout_ms: Maximum time to wait for a message (0 = non-blocking)
        
        Returns:
            Number of messages processed.
        """
        pass
    
    @abstractmethod
    def receive_blocking(self, timeout_ms: int = 0) -> bool:
        """
        Block until a message is received.
        
        Args:
            timeout_ms: Maximum time to wait (0 = infinite)
        
        Returns:
            True if a message was received, False on timeout.
        """
        pass
