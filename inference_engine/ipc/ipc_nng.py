"""
NNG-based IPC implementation using PAIR protocol over TCP.
"""

from typing import Callable

import pynng

from .iipc import (
    IIPC,
    MsgType,
    PointCloudMsg,
    TransformedPointsMsg,
    TransformedPointsBatchMsg,
    VectorListMsg,
    MetricsMsg,
    LayerMetaMsg,
    ImageMsg,
    LayoutMsg,
    ImageBatchMsg,
    serialize_point_cloud,
    serialize_layer_meta,
    serialize_transformed_points,
    serialize_transformed_points_batch,
    serialize_vector_list,
    serialize_metrics,
    serialize_image,
    serialize_image_batch,
    serialize_layout,
    serialize_ack,
    deserialize_point_cloud,
    deserialize_transformed_points,
    deserialize_transformed_points_batch,
    deserialize_vector_list,
    deserialize_metrics,
)


class NngIPC(IIPC):
    """
    NNG-based IPC implementation using PAIR protocol.
    
    The PAIR protocol provides bidirectional communication between
    exactly two endpoints.
    
    Example (Server/Backend):
    
        ipc = NngIPC()
        ipc.bind("tcp://*:9999")
        
        # Set callbacks
        ipc.on_transformed_points = lambda msg: handle_transform(msg)
        
        # Send initial data
        ipc.send_point_cloud(PointCloudMsg(cloud_id=0, points=my_points))
        
        # Poll for messages
        while running:
            ipc.poll()
    
    Example (Client/Renderer):
    
        ipc = NngIPC()
        ipc.connect("tcp://localhost:9999")
        
        ipc.on_point_cloud = lambda msg: display_cloud(msg)
        ipc.on_vector_list = lambda msg: handle_vectors(msg)
        
        while running:
            ipc.poll()
            # Send transformed points when user edits
            ipc.send_transformed_points(...)
    """
    
    def __init__(self):
        self._socket: pynng.Pair0 | None = None
        
        # Callbacks
        self.on_point_cloud: Callable[[PointCloudMsg], None] | None = None
        self.on_transformed_points: Callable[[TransformedPointsMsg], None] | None = None
        self.on_transformed_points_batch: Callable[[TransformedPointsBatchMsg], None] | None = None
        self.on_vector_list: Callable[[VectorListMsg], None] | None = None
        self.on_metrics: Callable[[MetricsMsg], None] | None = None
        self.on_ack: Callable[[], None] | None = None
    
    def bind(self, address: str = "tcp://*:9999") -> bool:
        """
        Server mode: listen for incoming connections.
        
        Args:
            address: Address to bind to (e.g., "tcp://*:9999")
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            self.close()
            self._socket = pynng.Pair0()
            self._socket.listen(address)
            print(f"[IPC] Listening on {address}")
            return True
        except Exception as e:
            print(f"[IPC] bind failed: {e}")
            self._socket = None
            return False
    
    def connect(self, address: str = "tcp://localhost:9999") -> bool:
        """
        Client mode: connect to a server.
        
        Args:
            address: Address to connect to (e.g., "tcp://localhost:9999")
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            self.close()
            self._socket = pynng.Pair0()
            self._socket.dial(address)
            print(f"[IPC] Connected to {address}")
            return True
        except Exception as e:
            print(f"[IPC] connect failed: {e}")
            self._socket = None
            return False
    
    def close(self) -> None:
        """Close the connection."""
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
            print("[IPC] Connection closed")
    
    def is_open(self) -> bool:
        """Check if connection is open."""
        return self._socket is not None
    
    # Send Methods
    
    def _send_raw(self, data: bytes) -> bool:
        """Send raw bytes."""
        if self._socket is None:
            return False
        try:
            self._socket.send(data)
            return True
        except Exception as e:
            print(f"[IPC] send failed: {e}")
            return False
    
    def send_point_cloud(self, msg: PointCloudMsg) -> bool:
        """Send a point cloud message."""
        return self._send_raw(serialize_point_cloud(msg))

    def send_layer_meta(self, msg: LayerMetaMsg) -> bool:
        """Send a layer meta message."""
        return self._send_raw(serialize_layer_meta(msg))

    def send_transformed_points(self, msg: TransformedPointsMsg) -> bool:
        """Send transformed points message."""
        return self._send_raw(serialize_transformed_points(msg))
    
    def send_transformed_points_batch(self, msg: TransformedPointsBatchMsg) -> bool:
        """Send transformed points message."""
        return self._send_raw(serialize_transformed_points_batch(msg))
    
    def send_vector_list(self, msg: VectorListMsg) -> bool:
        """Send a vector list message."""
        return self._send_raw(serialize_vector_list(msg))
    
    def send_metrics(self, msg: MetricsMsg) -> bool:
        """Send a metrics message."""
        return self._send_raw(serialize_metrics(msg))

    def send_image(self, msg: ImageMsg) -> bool:
        """Send an image message."""
        return self._send_raw(serialize_image(msg))

    def send_images(self, msg: ImageBatchMsg) -> bool:
        """Send an image batch message."""
        return self._send_raw(serialize_image_batch(msg))
    
    def send_layout(self, msg: LayoutMsg) -> bool:
        """Send a layout mode with the point offsets"""
        return self._send_raw(serialize_layout(msg))

    def send_ack(self) -> bool:
        """Send an acknowledgment."""
        return self._send_raw(serialize_ack())
    
    # Receive Methods
    
    def poll(self, timeout_ms: int = 0) -> int:
        """
        Process incoming messages (non-blocking by default).
        
        Args:
            timeout_ms: Maximum time to wait for first message (0 = non-blocking)
        
        Returns:
            Number of messages processed.
        """
        if self._socket is None:
            return 0
        
        count = 0
        self._socket.recv_timeout = timeout_ms if timeout_ms > 0 else 1  # Minimum 1ms
        
        while True:
            try:
                data = self._socket.recv()
                self._handle_message(data)
                count += 1
                # After first message, make subsequent receives non-blocking
                self._socket.recv_timeout = 1
            except pynng.Timeout:
                break
            except pynng.TryAgain:
                break
            except Exception as e:
                print(f"[IPC] poll error: {e}")
                break
        
        return count
    
    def receive_blocking(self, timeout_ms: int = 0) -> bool:
        """
        Block until a message is received.
        
        Args:
            timeout_ms: Maximum time to wait (0 = infinite)
        
        Returns:
            True if a message was received, False on timeout.
        """
        if self._socket is None:
            return False
        
        if timeout_ms > 0:
            self._socket.recv_timeout = timeout_ms
        else:
            self._socket.recv_timeout = None  # Infinite
        
        try:
            data = self._socket.recv()
            self._handle_message(data)
            return True
        except pynng.Timeout:
            return False
        except Exception as e:
            print(f"[IPC] receive_blocking error: {e}")
            return False
    
    def _handle_message(self, data: bytes) -> None:
        """Dispatch received message to appropriate callback."""
        if len(data) < 1:
            return
        
        msg_type = data[0]
        payload = data[1:]
        
        try:
            if msg_type == MsgType.POINT_CLOUD:
                if self.on_point_cloud:
                    self.on_point_cloud(deserialize_point_cloud(payload))
            
            elif msg_type == MsgType.TRANSFORMED_POINTS:
                if self.on_transformed_points:
                    self.on_transformed_points(deserialize_transformed_points(payload))

            elif msg_type == MsgType.TRANSFORMED_POINTS_BATCH:
                if self.on_transformed_points_batch:
                    self.on_transformed_points_batch(deserialize_transformed_points_batch(payload))
            
            elif msg_type == MsgType.VECTOR_LIST:
                if self.on_vector_list:
                    self.on_vector_list(deserialize_vector_list(payload))
            
            elif msg_type == MsgType.METRICS:
                if self.on_metrics:
                    self.on_metrics(deserialize_metrics(payload))

            elif msg_type == MsgType.ACK:
                if self.on_ack:
                    self.on_ack()
            
            else:
                print(f"[IPC] Unknown message type: {msg_type}")
        
        except Exception as e:
            print(f"[IPC] Error handling message type {msg_type}: {e}")
    
    def __del__(self):
        """Clean up on destruction."""
        self.close()


# Convenience factory function
def create_nng_ipc() -> NngIPC:
    """Create an NNG-based IPC instance."""
    return NngIPC()
