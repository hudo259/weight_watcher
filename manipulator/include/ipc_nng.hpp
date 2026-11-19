#pragma once

#include "iipc.hpp"

#include <nng/nng.h>
#include <nng/protocol/pair0/pair.h>

#include <atomic>
#include <mutex>

namespace ipc {

    /**
     * This implementation uses NNG (nanomsg next generation) for reliable
     * bidirectional communication. The PAIR protocol provides exactly one
     * connection between two endpoints.
     * 
     * Regarding thread safety:
     * - send_* methods are thread-safe (protected by mutex)
     * - poll() and receive_blocking() should be called from a single thread
     * - Callbacks are invoked on the polling thread
     */
    class NngIPC : public IIPC {
    public:
        NngIPC();
        ~NngIPC() override;
        
        // Non-copyable, non-movable
        NngIPC(const NngIPC&) = delete;
        NngIPC& operator=(const NngIPC&) = delete;
        NngIPC(NngIPC&&) = delete;
        NngIPC& operator=(NngIPC&&) = delete;
        
        // Connection
        bool bind(const std::string& address) override;
        bool connect(const std::string& address) override;
        void close() override;
        bool is_open() const override;
        
        // Send
        bool send_point_cloud(
            uint32_t cloud_id,
            const std::vector<glm::vec3>& points,
            const std::vector<glm::vec3>* colors = nullptr,
            const std::vector<glm::vec3>* normals = nullptr) override;
        
        bool send_transformed_points(
            uint32_t cloud_id,
            const std::vector<uint32_t>& indices,
            const std::vector<glm::vec3>& positions) override;

        bool send_transformed_points_batch(const TransformedPointsBatchMsg& msg) override;
        
        bool send_vector_list(
            uint32_t id,
            uint32_t dimensions,
            const std::vector<float>& data) override;
        
        bool send_metrics(
            const std::vector<std::string>& names,
            const std::vector<float>& values) override;
        
        bool send_ack() override;
        
        // Callbacks
        void on_point_cloud(PointCloudCallback cb) override;
        void on_layer_meta(LayerMetaCallback cb) override;
        void on_transformed_points(TransformedPointsCallback cb) override;
        void on_transformed_points_batch(TransformedPointsBatchCallback cb) override;
        void on_vector_list(VectorListCallback cb) override;
        void on_metrics(MetricsCallback cb) override;
        void on_image(ImageCallback cb) override;
        void on_image_batch(ImageBatchCallback cb) override;
        void on_layout(LayoutCallback cb) override;
        void on_ack(AckCallback cb) override;
        
        // Polling
        int poll() override;
        bool receive_blocking(uint32_t timeout_ms = 0) override;

    private:
        bool send_raw(const std::vector<uint8_t>& data);
        bool receive_one(int flags);  // NNG_FLAG_NONBLOCK or 0
        void handle_message(const uint8_t* data, size_t size);
        
        nng_socket socket_;
        std::atomic<bool> open_{false};
        
        std::mutex send_mutex_;
        
        PointCloudCallback point_cloud_cb_;
        LayerMetaCallback layer_meta_cb_;
        TransformedPointsCallback transformed_points_cb_;
        TransformedPointsBatchCallback transformed_points_batch_cb_;
        VectorListCallback vector_list_cb_;
        MetricsCallback metrics_cb_;
        ImageCallback image_cb_;
        ImageBatchCallback image_batch_cb_;
        LayoutCallback layout_cb_;
        AckCallback ack_cb_;
    };

}