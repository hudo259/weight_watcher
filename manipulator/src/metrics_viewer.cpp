#include "metrics_viewer.hpp"
#include "busy_section.hpp"

#include <algorithm>

void MetricsViewer::set_loss(float loss) {
    loss_     = loss;
    has_loss_ = true;
}

void MetricsViewer::set_metrics(const std::vector<std::string>& names,
                                const std::vector<float>& values) {
    metrics_.clear();
    size_t n = std::min(names.size(), values.size());
    metrics_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        metrics_.emplace_back(names[i], values[i]);
    }
}

void MetricsViewer::clear() {
    has_loss_ = false;
    loss_     = 0.0f;
    metrics_.clear();
}

void MetricsViewer::attach(ipc::IPCRouter& router) {
    jobs_ = &router.jobs();

    router.on_metrics([this](ipc::MetricsMsg msg) {
        set_metrics(msg.names, msg.values);
    });

    router.on_vector_list(ipc::VectorListID::LOSS,
        [this](ipc::VectorListMsg msg) { set_loss(msg.value()); });
}

void MetricsViewer::draw() {
    if (!has_data()) return;

    ImGui::Begin("Backend Metrics");

    // Loss display
    if (has_loss_) {
        ImGui::Text("Loss:");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "%.6f", loss_);

        ImGui::ProgressBar(std::min(loss_, 1.0f), ImVec2(-1, 0), "");
        ImGui::Spacing();
    }

    // Metrics table
    if (!metrics_.empty()) {
        // Metrics are recomputed after every weight change...
        const bool busy = jobs_ && (jobs_->busy(ipc::JobType::TRANSFORMED_POINTS) || jobs_->busy(ipc::JobType::POINT_CLOUD));
        ui::BusySection section(busy, "Updating metrics...", ui::BusyInput::ALLOW);

        if (has_loss_) {
            ImGui::Separator();
        }

        ImGui::Text("Metrics:");
        ImGui::Spacing();

        if (ImGui::BeginTable("MetricsTable", 2,
                              ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 100.0f);
            ImGui::TableHeadersRow();

            for (const auto& [name, value] : metrics_) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", name.c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", value);
            }

            ImGui::EndTable();
        }
    }

    ImGui::Separator();
    if (ImGui::Button("Clear", ImVec2(-1, 0))) {
        clear();
    }

    ImGui::End();
}
