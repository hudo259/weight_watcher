#pragma once

#include "ipc_messages.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <string>

// Forward declaration (defined in scene.hpp)
class Scene;

namespace ipc {

    /**
     * @brief Abstract IPC interface for communication with Python backend.
     * 
     * This interface allows for different transport implementations:
     * - NngIPC: Uses NNG (nanomsg next generation) over TCP
     * - Future: ShmIPC for shared memory transport
     * 
     * ## Usage Pattern (Renderer as Client)
     * 
     * The renderer connects to a running backend:
     * 
     * ```cpp
     * auto ipc = create_nng_ipc();
     * ipc->connect("tcp://localhost:9999");
     * 
     * // Set up callbacks for receiving data
     * ipc->on_point_cloud([](PointCloudMsg msg) {
     *     // Handle received point cloud
     * });
     * 
     * ipc->on_vector_list([](VectorListMsg msg) {
     *     if (msg.id == VectorListID::LOSS) {
     *         // Handle loss
     *     } else if (msg.id == VectorListID::FEATURES) {
     *         // Handle features
     *     }
     * });
     * 
     * ipc->on_metrics([](MetricsMsg msg) {
     *     // Handle metrics
     * });
     * 
     * // In render loop:
     * ipc->poll();  // Process incoming messages
     * 
     * // Send transformed points when user manipulates them
     * ipc->send_transformed_points(cloud_id, indices, positions);
     * ```
     */
    class IIPC {
    public:
        virtual ~IIPC() = default;
        
        // Connection
        
        /// Server mode: listen for incoming connections
        virtual bool bind(const std::string& address) = 0;
        
        /// Client mode: connect to a server
        virtual bool connect(const std::string& address) = 0;
        
        /// Close the connection
        virtual void close() = 0;
        
        /// Check if connection is open
        virtual bool is_open() const = 0;
        
        // Send Methods
        
        /// Send a full point cloud
        virtual bool send_point_cloud(
            uint32_t cloud_id,
            const std::vector<glm::vec3>& points,
            const std::vector<glm::vec3>* colors = nullptr,
            const std::vector<glm::vec3>* normals = nullptr) = 0;
        
        /// Send transformed points (subset with new positions)
        virtual bool send_transformed_points(
            uint32_t cloud_id,
            const std::vector<uint32_t>& indices,
            const std::vector<glm::vec3>& positions) = 0;

        virtual bool send_transformed_points_batch(const TransformedPointsBatchMsg& msg) = 0;
        
        /// Send n-dimensional vectors (features, loss, etc.)
        virtual bool send_vector_list(
            uint32_t id,
            uint32_t dimensions,
            const std::vector<float>& data) = 0;
        
        /// Convenience: send a single loss value
        bool send_loss(float loss) {
            return send_vector_list(VectorListID::LOSS, 1, {loss});
        }
        
        /// Send metrics (name-value pairs)
        virtual bool send_metrics(
            const std::vector<std::string>& names,
            const std::vector<float>& values) = 0;
        
        /// Send acknowledgment
        virtual bool send_ack() = 0;
        
        // Receive Callbacks
        
        using PointCloudCallback = std::function<void(PointCloudMsg)>;
        using LayerMetaCallback = std::function<void(LayerMetaMsg)>;
        using TransformedPointsCallback = std::function<void(TransformedPointsMsg)>;
        using TransformedPointsBatchCallback = std::function<void(TransformedPointsBatchMsg)>;
        using VectorListCallback = std::function<void(VectorListMsg)>;
        using MetricsCallback = std::function<void(MetricsMsg)>;
        using ImageCallback = std::function<void(ImageMsg)>;
        using ImageBatchCallback = std::function<void(ImageBatchMsg)>;
        using LayoutCallback = std::function<void(LayoutMsg)>;
        using AckCallback = std::function<void()>;
        
        virtual void on_point_cloud(PointCloudCallback cb) = 0;
        virtual void on_layer_meta(LayerMetaCallback cb) = 0;
        virtual void on_transformed_points(TransformedPointsCallback cb) = 0;
        virtual void on_transformed_points_batch(TransformedPointsBatchCallback cb) = 0;
        virtual void on_vector_list(VectorListCallback cb) = 0;
        virtual void on_metrics(MetricsCallback cb) = 0;
        virtual void on_image(ImageCallback cb) = 0;
        virtual void on_image_batch(ImageBatchCallback cb) = 0;
        virtual void on_layout(LayoutCallback cb) = 0;
        virtual void on_ack(AckCallback cb) = 0;
        
        // Polling
        
        /// Process incoming messages (non-blocking). Returns number processed.
        virtual int poll() = 0;
        
        /// Block until a message is received (with optional timeout in ms, 0 = infinite)
        virtual bool receive_blocking(uint32_t timeout_ms = 0) = 0;
    };

    

    // Factory Functions

    /// Create NNG-based IPC implementation
    std::unique_ptr<IIPC> create_nng_ipc();

    // Future implementations:
    // std::unique_ptr<IIPC> create_shm_ipc();

}