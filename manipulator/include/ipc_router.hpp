#pragma once

#include "iipc.hpp"
#include "ipc_messages.hpp"
#include "job_tracker.hpp"
#include "point_cloud.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration
class Scene;

namespace ipc {

// Key encoding helpers.
// We abuse a single uint64_t to pack two uint32_t IDs into one map key:
//   high 32 bits = cloud_id  (which layer/cloud this point belongs to)
//   low  32 bits = point_idx (the original point index within that cloud)
// This lets us use a single flat unordered_map instead of a nested one,
// avoiding double hashing and separate heap allocations per cloud.
// Safe because cloud counts and point counts are both far below UINT32_MAX
// in any realistic model (dozens of layers, hundreds of PCA components).
inline uint64_t make_transform_key(uint32_t cloud_id, uint32_t point_idx) {
    return (static_cast<uint64_t>(cloud_id) << 32) | point_idx;
}
inline uint32_t transform_key_cloud(uint64_t key) { return static_cast<uint32_t>(key >> 32); }
inline uint32_t transform_key_point(uint64_t key) { return static_cast<uint32_t>(key); }

struct AccumulatedTransform {
    // Both maps are keyed by the packed uint64_t (cloud_id | point_idx).
    std::unordered_map<uint64_t, glm::vec3> index_to_pos;
    std::unordered_map<uint64_t, glm::vec3> index_to_original;
    bool has_data = false;

    void merge(uint32_t cid,
               const std::vector<uint32_t>&  indices,
               const std::vector<glm::vec3>& positions,
               const std::vector<glm::vec3>& positions_before);

    void clear();

    // Calls f(cloud_id, indices, positions) once per cloud that has pending data.
    template<typename F>
    void for_each_cloud(F&& f) const {
        std::unordered_map<uint32_t,
            std::pair<std::vector<uint32_t>, std::vector<glm::vec3>>> by_cloud;
        for (auto& [k, pos] : index_to_pos) {
            auto& [idxs, pts] = by_cloud[transform_key_cloud(k)];
            idxs.push_back(transform_key_point(k));
            pts.push_back(pos);
        }
        for (auto& [cid, payload] : by_cloud)
            f(cid, payload.first, payload.second);
    }

    // Same as for_each_cloud but yields the original (pre-transform) positions.
    template<typename F>
    void for_each_cloud_original(F&& f) const {
        std::unordered_map<uint32_t,
            std::pair<std::vector<uint32_t>, std::vector<glm::vec3>>> by_cloud;
        for (auto& [k, pos] : index_to_original) {
            auto& [idxs, pts] = by_cloud[transform_key_cloud(k)];
            idxs.push_back(transform_key_point(k));
            pts.push_back(pos);
        }
        for (auto& [cid, payload] : by_cloud)
            f(cid, payload.first, payload.second);
    }
};

// IPCRouter
//
// Components call on_*() to subscribe to incoming messages, and the send_*()
// methods to send outgoing messages.  Each component registers its own needs
// in an attach(IPCRouter&) method (which is what we should call in main. Nothing else)
//
// # Notes so that I don't forget later:
// - Multiple subscribers per message typename (vectors of callbacks).
// - Per-ID dispatch for VECTOR_LIST via unordered_map.
// - send_*() are safe no-ops when not connected.
class IPCRouter {
public:
    IPCRouter();
    ~IPCRouter();

    // Must be called once after the GL context exists (creates the IIPC impl).
    void init();

    // Shut down: call before GL context is destroyed.
    void shutdown();

    // Call once per frame from the main loop.
    void poll();

    // Draw the connection / send panel ImGui window.
    void draw_ui(Scene& scene);

    bool is_connected() const { return connected_; }

    // Outgoing sends (safe no-ops when not connected)
    bool send_vector_list(uint32_t id, uint32_t dimensions,
                          std::vector<float> data);

    bool send_transformed_points(uint32_t cloud_id,
                                 const std::vector<uint32_t>&  indices,
                                 const std::vector<glm::vec3>& positions);
    bool send_transformed_points_batch(const TransformedPointsBatchMsg& msg);

    bool send_point_cloud(uint32_t cloud_id,
                          const std::vector<glm::vec3>& points);

    // Subscriptions
    //
    // Components call these from their attach(IPCRouter&) method.
    // Multiple subscribers per type are all called in registration order.

    void on_point_cloud (std::function<void(PointCloudMsg)>  cb);
    void on_layer_meta  (std::function<void(LayerMetaMsg)>   cb);
    void on_image_batch (std::function<void(ImageBatchMsg)>  cb);
    void on_metrics     (std::function<void(MetricsMsg)>     cb);
    void on_layout      (std::function<void(LayoutMsg)>      cb);
    void on_ack         (std::function<void()>               cb);

    // Subscribe to a specific VectorList ID.  Only one handler per ID.
    // Use VectorListID::LOSS, FEATURES, CONCEPT_SCORES, etc.
    void on_vector_list(uint32_t id, std::function<void(VectorListMsg)> cb);

    // Pending backend jobs, read by UI sections to show busy spinners
    const JobTracker& jobs() const {return jobs_;}

    //state that needs to be public so that it can be accessed by draw_ui and draw_gizmo
    char    address_buf[256]      = "tcp://localhost:9999";
    bool    auto_send_transformed = false;
    AccumulatedTransform pending_transform;

private:
    void connect();
    void disconnect();

    void wire_ipc_callbacks();
    void dispatch_vector_list(VectorListMsg msg);

    // State
    std::unique_ptr<IIPC> ipc_;
    bool   connected_ = false;
    std::string address_;
    JobTracker jobs_;

    // Subscriber lists
    std::vector<std::function<void(PointCloudMsg)>>  point_cloud_cbs_;
    std::vector<std::function<void(LayerMetaMsg)>>   layer_meta_cbs_;
    std::vector<std::function<void(ImageBatchMsg)>>  image_batch_cbs_;
    std::vector<std::function<void(MetricsMsg)>>     metrics_cbs_;
    std::vector<std::function<void(LayoutMsg)>>      layout_cbs_;
    std::vector<std::function<void()>>               ack_cbs_;

    // Per-ID vector list dispatch table
    std::unordered_map<uint32_t, std::function<void(VectorListMsg)>>
        vector_list_handlers_;
};

}
