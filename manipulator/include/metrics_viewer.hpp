#pragma once

#include "ipc_router.hpp"

#include <imgui.h>

#include <string>
#include <vector>
#include <cstdint>

/**
 * @brief Displays loss and named metrics received from the backend.
 *
 * Owns the data. IPC callbacks write directly into it via
 * set_loss() and set_metrics(). Draws its own ImGui window.
 *
 * Usage:
 *   MetricsViewer metrics;
 *   // In IPC callback:
 *   metrics.set_loss(value);
 *   metrics.set_metrics(names, values);
 *   // In render loop:
 *   metrics.draw();
 */
class MetricsViewer {
public:
    /// Set the current loss value.
    void set_loss(float loss);

    /// Replace all metrics with new name-value pairs.
    void set_metrics(const std::vector<std::string>& names,
                     const std::vector<float>& values);

    void attach(ipc::IPCRouter& router);

    /// Draw the ImGui window. Only shows if there's data to display.
    void draw();

    /// Clear all data.
    void clear();

    bool has_data() const { return has_loss_ || !metrics_.empty(); }

private:
    // Backend job state for busy spinners, set in attach()
    const ipc::JobTracker* jobs_ = nullptr;

    float loss_     = 0.0f;
    bool  has_loss_ = false;

    std::vector<std::pair<std::string, float>> metrics_;
};
