#include "ipc_router.hpp"
#include "ipc_nng.hpp"
#include "scene.hpp"
#include "point_cloud.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

#include <iostream>

namespace ipc {

void AccumulatedTransform::merge(uint32_t cid,
                                  const std::vector<uint32_t>&  indices,
                                  const std::vector<glm::vec3>& positions,
                                  const std::vector<glm::vec3>& positions_before) {
    has_data = true;
    for (size_t i = 0; i < indices.size(); ++i) {
        uint64_t k = make_transform_key(cid, indices[i]);
        // Only record the original position the first time we see this point
        // so that repeated drags don't overwrite the true pre-transform origin.
        if (index_to_original.find(k) == index_to_original.end())
            index_to_original[k] = positions_before[i];
        index_to_pos[k] = positions[i];
    }
}

void AccumulatedTransform::clear() {
    index_to_pos.clear();
    index_to_original.clear();
    has_data = false;
}

IPCRouter::IPCRouter() = default;

IPCRouter::~IPCRouter() {
    shutdown();
}

void IPCRouter::init() {
    ipc_ = create_nng_ipc();
    wire_ipc_callbacks();

    // Backend job lifecycle: row = [job_type, status]
    on_vector_list(VectorListID::JOB_STATUS, [this](VectorListMsg msg) {
            if (msg.data.size() < 2) {
                return;
            }
            jobs_.on_status(static_cast<JobType>(msg.data[0]), static_cast<JobStatus>(msg.data[1]));
    });
}

void IPCRouter::shutdown() {
    if (connected_) disconnect();
    // Clear all subscriber lists so no dangling references survive
    point_cloud_cbs_.clear();
    layer_meta_cbs_.clear();
    image_batch_cbs_.clear();
    metrics_cbs_.clear();
    ack_cbs_.clear();
    vector_list_handlers_.clear();
    layout_cbs_.clear();
}

void IPCRouter::poll() {
    if (ipc_ && connected_)
        ipc_->poll();
}

void IPCRouter::connect() {
    if (!ipc_) return;
    if (connected_) {
        ipc_->close();
        connected_ = false;
    }
    address_ = std::string(address_buf);
    connected_ = ipc_->connect(address_);
    if (connected_) {
        std::cout << "[IPCRouter] Connected to " << address_ << "\n";
        ipc_->send_ack();
    }
}

void IPCRouter::disconnect() {
    if (!ipc_ || !connected_) return;
    ipc_->close();
    connected_ = false;
    std::cout << "[IPCRouter] Disconnected\n";
}

// Outgoing sends
bool IPCRouter::send_vector_list(uint32_t id, uint32_t dimensions,
                                  std::vector<float> data) {
    if (!ipc_ || !connected_) return false;
    return ipc_->send_vector_list(id, dimensions, data);
}

bool IPCRouter::send_transformed_points(uint32_t cloud_id,
                                         const std::vector<uint32_t>&  indices,
                                         const std::vector<glm::vec3>& positions) {
    if (!ipc_ || !connected_) return false;
    return ipc_->send_transformed_points(cloud_id, indices, positions);
}

bool IPCRouter::send_transformed_points_batch(const TransformedPointsBatchMsg& msg) {
    if (!ipc_ || !connected_) return false;
    return ipc_->send_transformed_points_batch(msg);
}

bool IPCRouter::send_point_cloud(uint32_t cloud_id,
                                  const std::vector<glm::vec3>& points) {
    if (!ipc_ || !connected_) return false;
    return ipc_->send_point_cloud(cloud_id, points);
}

// Subscriptions
void IPCRouter::on_point_cloud(std::function<void(PointCloudMsg)> cb) {
    point_cloud_cbs_.push_back(std::move(cb));
}

void IPCRouter::on_layer_meta(std::function<void(LayerMetaMsg)> cb) {
    layer_meta_cbs_.push_back(std::move(cb));
}

void IPCRouter::on_image_batch(std::function<void(ImageBatchMsg)> cb) {
    image_batch_cbs_.push_back(std::move(cb));
}

void IPCRouter::on_metrics(std::function<void(MetricsMsg)> cb) {
    metrics_cbs_.push_back(std::move(cb));
}

void IPCRouter::on_layout(std::function<void(LayoutMsg)> cb) {
    layout_cbs_.push_back(std::move(cb));
}

void IPCRouter::on_ack(std::function<void()> cb) {
    ack_cbs_.push_back(std::move(cb));
}

void IPCRouter::on_vector_list(uint32_t id,
                                std::function<void(VectorListMsg)> cb)
{
    vector_list_handlers_[id] = std::move(cb);
}

// Internal dispatch
void IPCRouter::dispatch_vector_list(VectorListMsg msg) {
    auto it = vector_list_handlers_.find(msg.id);
    if (it != vector_list_handlers_.end())
        it->second(std::move(msg));
    else
        std::cout << "[IPCRouter] Unhandled VectorList id=" << msg.id << "\n";
}

void IPCRouter::wire_ipc_callbacks() {
    if (!ipc_) return;

    ipc_->on_point_cloud([this](PointCloudMsg msg) {
        for (auto& cb : point_cloud_cbs_) cb(msg);
    });

    ipc_->on_layer_meta([this](LayerMetaMsg msg) {
        for (auto& cb : layer_meta_cbs_) cb(msg);
    });

    ipc_->on_vector_list([this](VectorListMsg msg) {
        dispatch_vector_list(std::move(msg));
    });

    ipc_->on_metrics([this](MetricsMsg msg) {
        for (auto& cb : metrics_cbs_) cb(msg);
    });

    ipc_->on_image_batch([this](ImageBatchMsg msg) {
        for (auto& cb : image_batch_cbs_) cb(std::move(msg));
    });

    ipc_->on_layout([this](LayoutMsg msg) {
        for (auto& cb : layout_cbs_) cb(std::move(msg));
    });

    ipc_->on_ack([this]() {
        for (auto& cb : ack_cbs_) cb();
    });
}

// draw_ui: connection panel + pending transforms
void IPCRouter::draw_ui(Scene& scene) {
    ImGui::Begin("Backend Connection");

    // Connection
    if (ImGui::CollapsingHeader("Connection", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Status:");
        ImGui::SameLine();
        if (connected_)
            ImGui::TextColored({0.2f, 1.0f, 0.2f, 1.0f}, "Connected");
        else
            ImGui::TextColored({1.0f, 0.3f, 0.3f, 1.0f}, "Disconnected");

        ImGui::Text("Address:");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##address", address_buf, sizeof(address_buf));

        if (connected_) {
            if (ImGui::Button("Disconnect", {-1, 0})) disconnect();
        } else {
            if (ImGui::Button("Connect", {-1, 0})) connect();
        }
    }

    if (!connected_) { ImGui::End(); return; }

    ImGui::Separator();

    // Send
    if (ImGui::CollapsingHeader("Send to Backend", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Auto-send on transform", &auto_send_transformed);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Automatically send transformed points when using the gizmo");

        ImGui::Separator();

        // Pending transforms
        // pending point count across all clouds
        size_t pending_count = pending_transform.index_to_pos.size();

        ImGui::BeginDisabled(!pending_transform.has_data);
        if (ImGui::Button("Send Pending Transforms", {-1, 0})) {
            TransformedPointsBatchMsg batch;
            pending_transform.for_each_cloud([&batch](uint32_t cid,
                                                    const std::vector<uint32_t>&  idx,
                                                    const std::vector<glm::vec3>& pos) {
                batch.clouds.push_back({cid, idx, pos});
            });
            send_transformed_points_batch(batch);
            pending_transform.clear();
        }
        ImGui::EndDisabled();
        if (pending_count > 0) {
            ImGui::SameLine();
            ImGui::TextDisabled("(%zu pts)", pending_count);
        }

        ImGui::BeginDisabled(!pending_transform.has_data);
        if (ImGui::Button("Reset Pending (restore positions)", {-1, 0})) {
            pending_transform.for_each_cloud_original([this, &scene](uint32_t cid,
                                                                    const std::vector<uint32_t>&  idx,
                                                                    const std::vector<glm::vec3>& orig) {
                PointCloud* cloud = scene.cloud(cid);
                if (!cloud) return;
                for (size_t i = 0; i < idx.size(); ++i) {
                    glm::vec3 delta = orig[i] - cloud->get_point_by_original(idx[i]);
                    cloud->select(idx[i], false);
                    cloud->apply_transform(glm::translate(glm::mat4(1.0f), delta), orig[i]);
                }
                cloud->clear_selection();
            });
            pending_transform.clear();
        }
        ImGui::EndDisabled();

        ImGui::Separator();

        // Send full cloud
        ImGui::Text("Send Full Cloud:");
        for (uint32_t i = 0; i < scene.cloud_count(); i++) {
            PointCloud* cloud = scene.cloud(i);
            if (!cloud) continue;
            char label[64];
            snprintf(label, sizeof(label), "Cloud %u (%zu pts)##full", i, cloud->size());
            if (ImGui::Button(label, {-1, 0}))
                send_point_cloud(i, cloud->points());
        }
    }

    ImGui::End();
}

}
