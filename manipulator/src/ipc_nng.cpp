#include "ipc_nng.hpp"

#include <iostream>

namespace ipc {

    NngIPC::NngIPC() {
        socket_ = NNG_SOCKET_INITIALIZER;
    }

    NngIPC::~NngIPC() {
        close();
    }

    bool NngIPC::bind(const std::string& address) {
        if (open_) close();
        
        int rv = nng_pair0_open(&socket_);
        if (rv != 0) {
            std::cerr << "[IPC] nng_pair0_open failed: " << nng_strerror(rv) << "\n";
            return false;
        }
        
        rv = nng_listen(socket_, address.c_str(), nullptr, 0);
        if (rv != 0) {
            std::cerr << "[IPC] nng_listen failed on " << address << ": " << nng_strerror(rv) << "\n";
            nng_close(socket_);
            return false;
        }
        
        std::cout << "[IPC] Listening on " << address << "\n";
        open_ = true;
        return true;
    }

    bool NngIPC::connect(const std::string& address) {
        if (open_) close();
        
        int rv = nng_pair0_open(&socket_);
        if (rv != 0) {
            std::cerr << "[IPC] nng_pair0_open failed: " << nng_strerror(rv) << "\n";
            return false;
        }
        
        rv = nng_dial(socket_, address.c_str(), nullptr, 0);
        if (rv != 0) {
            std::cerr << "[IPC] nng_dial failed on " << address << ": " << nng_strerror(rv) << "\n";
            nng_close(socket_);
            return false;
        }
        
        std::cout << "[IPC] Connected to " << address << "\n";
        open_ = true;
        return true;
    }

    void NngIPC::close() {
        if (open_) {
            nng_close(socket_);
            open_ = false;
            std::cout << "[IPC] Connection closed\n";
        }
    }

    bool NngIPC::is_open() const {
        return open_;
    }

    // Send Methods

    bool NngIPC::send_raw(const std::vector<uint8_t>& data) {
        if (!open_) return false;
        
        std::lock_guard<std::mutex> lock(send_mutex_);
        
        nng_msg* msg;
        int rv = nng_msg_alloc(&msg, data.size());
        if (rv != 0) {
            std::cerr << "[IPC] nng_msg_alloc failed: " << nng_strerror(rv) << "\n";
            return false;
        }
        
        std::memcpy(nng_msg_body(msg), data.data(), data.size());
        
        rv = nng_sendmsg(socket_, msg, 0);
        if (rv != 0) {
            std::cerr << "[IPC] nng_sendmsg failed: " << nng_strerror(rv) << "\n";
            nng_msg_free(msg);
            return false;
        }
        
        return true;
    }

    bool NngIPC::send_point_cloud(
        uint32_t cloud_id,
        const std::vector<glm::vec3>& points,
        const std::vector<glm::vec3>* colors,
        const std::vector<glm::vec3>* normals) {
        auto data = serialize_point_cloud(cloud_id, points, colors, normals);
        return send_raw(data);
    }

    bool NngIPC::send_transformed_points(
        uint32_t cloud_id,
        const std::vector<uint32_t>& indices,
        const std::vector<glm::vec3>& positions) {
        auto data = serialize_transformed_points(cloud_id, indices, positions);
        return send_raw(data);
    }

    bool NngIPC::send_transformed_points_batch(const TransformedPointsBatchMsg& msg) {
        auto data = serialize_transformed_points_batch(msg);
        return send_raw(data);
    }

    bool NngIPC::send_vector_list(
        uint32_t id,
        uint32_t dimensions,
        const std::vector<float>& data) {
        auto msg = serialize_vector_list(id, dimensions, data);
        return send_raw(msg);
    }

    bool NngIPC::send_metrics(
        const std::vector<std::string>& names,
        const std::vector<float>& values) {
        auto data = serialize_metrics(names, values);
        return send_raw(data);
    }

    bool NngIPC::send_ack() {
        auto data = serialize_ack();
        return send_raw(data);
    }

    // Callbacks

    void NngIPC::on_point_cloud(PointCloudCallback cb) {
        point_cloud_cb_ = std::move(cb);
    }

    void NngIPC::on_layer_meta(LayerMetaCallback cb) {
        layer_meta_cb_ = std::move(cb);
    }

    void NngIPC::on_transformed_points(TransformedPointsCallback cb) {
        transformed_points_cb_ = std::move(cb);
    }

    void NngIPC::on_transformed_points_batch(TransformedPointsBatchCallback cb) {
        transformed_points_batch_cb_ = std::move(cb);
    }

    void NngIPC::on_vector_list(VectorListCallback cb) {
        vector_list_cb_ = std::move(cb);
    }

    void NngIPC::on_metrics(MetricsCallback cb) {
        metrics_cb_ = std::move(cb);
    }

    void NngIPC::on_image(ImageCallback cb) {
        image_cb_ = std::move(cb);
    }

    void NngIPC::on_image_batch(ImageBatchCallback cb) {
        image_batch_cb_ = std::move(cb);
    }

    void NngIPC::on_layout(LayoutCallback cb) {
        layout_cb_ = std::move(cb);
    }

    void NngIPC::on_ack(AckCallback cb) {
        ack_cb_ = std::move(cb);
    }

    // Receiving

    bool NngIPC::receive_one(int flags) {
        if (!open_) return false;
        
        nng_msg* msg;
        int rv = nng_recvmsg(socket_, &msg, flags);
        
        if (rv != 0) {
            return false;  // No message or error (including NNG_ETIMEDOUT, NNG_EAGAIN)
        }
        
        handle_message(
            static_cast<const uint8_t*>(nng_msg_body(msg)),
            nng_msg_len(msg)
        );
        
        nng_msg_free(msg);
        return true;
    }

    int NngIPC::poll() {
        int count = 0;
        while (receive_one(NNG_FLAG_NONBLOCK)) {
            count++;
        }
        return count;
    }

    bool NngIPC::receive_blocking(uint32_t timeout_ms) {
        if (!open_) return false;
        
        // Set receive timeout
        if (timeout_ms > 0) {
            nng_duration dur = static_cast<nng_duration>(timeout_ms);
            nng_socket_set_ms(socket_, NNG_OPT_RECVTIMEO, dur);
        } else {
            nng_socket_set_ms(socket_, NNG_OPT_RECVTIMEO, NNG_DURATION_INFINITE);
        }
        
        return receive_one(0);
    }

    void NngIPC::handle_message(const uint8_t* data, size_t size) {
        if (size < 1) return;
        
        try {
            MsgType type = static_cast<MsgType>(data[0]);
            BinaryReader reader(data + 1, size - 1);
           
            //TODO: If this thing gets to long, make a dispatch table instead
            switch (type) {
                case MsgType::POINT_CLOUD:
                    if (point_cloud_cb_) {
                        point_cloud_cb_(deserialize_point_cloud(reader));
                    }
                    break;
                    
                case MsgType::TRANSFORMED_POINTS:
                    if (transformed_points_cb_) {
                        transformed_points_cb_(deserialize_transformed_points(reader));
                    }
                    break;
                
                case MsgType::TRANSFORMED_POINTS_BATCH:
                    if (transformed_points_batch_cb_) {
                        transformed_points_batch_cb_(deserialize_transformed_points_batch(reader));
                    }    
                    break;
                    
                case MsgType::VECTOR_LIST:
                    if (vector_list_cb_) {
                        vector_list_cb_(deserialize_vector_list(reader));
                    }
                    break;
                    
                case MsgType::METRICS:
                    if (metrics_cb_) {
                        metrics_cb_(deserialize_metrics(reader));
                    }
                    break;
                
                case MsgType::IMAGE:
                    if (image_cb_) {
                        image_cb_(deserialize_image(reader));
                    }
                    break;
                
                case MsgType::IMAGE_BATCH:
                    if (image_batch_cb_) {
                        image_batch_cb_(deserialize_image_batch(reader));
                    }
                    break;
                
                case MsgType::LAYER_META:
                    if(layer_meta_cb_) {
                        layer_meta_cb_(deserialize_layer_meta(reader));
                    }
                    break;

                case MsgType::LAYOUT:
                    if(layout_cb_) {
                        layout_cb_(deserialize_layout(reader));
                    }
                    break;
                    
                case MsgType::ACK:
                    if (ack_cb_) {
                        ack_cb_();
                    }
                    break;
                    
                default:
                    std::cerr << "[IPC] Unknown message type: " 
                            << static_cast<int>(type) << "\n";
                    break;
            }
        } catch (const std::exception& e) {
            std::cerr << "[IPC] Error handling message: " << e.what() << "\n";
        }
    }

    // Factory Function

    std::unique_ptr<IIPC> create_nng_ipc() {
        return std::make_unique<NngIPC>();
    }

}