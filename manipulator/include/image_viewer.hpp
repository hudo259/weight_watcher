#pragma once

#include "image_modes.hpp"
#include "ipc_messages.hpp"
#include "ipc_router.hpp"

#include <glad/gl.h>
#include <imgui.h>

#include <memory>
#include <vector>
#include <string>
#include <cstdint>

/**
 * @brief ImGui window that displays anchor images in selectable overlay modes.
 *
 * Stores the raw RGBA8 anchor images (received once at connect). Owns three IImageMode instances (Raw, Predictions, Heatmap).
 * Draws the ImGui window: toolbar, single-image view, grid view. Routes incoming ImageBatchMsg batches to all modes.
 *
 * ## Use like this:
 *
 *   viewer.init();                 // after GL context — calls mode.init() on all modes
 *   viewer.add_image_batch(msg);   // forwards to all modes; they filter by batch_type
 *   viewer.draw();                 // once per frame inside ImGui frame
 */
class ImageViewer {
public:
    enum class ViewMode { Single, Grid };

    ImageViewer();
    ~ImageViewer();

    ImageViewer(const ImageViewer&)            = delete;
    ImageViewer& operator=(const ImageViewer&) = delete;

    // Compile shaders and initialise all modes.
    // Must be called once after the GL context exists.
    void init();

    // Forward the batch to all modes, each mode picks up its own batch_type.
    // RAW_IMAGES batches are also stored in entries_raw_ by ImageViewer itself.
    void add_image_batch(ipc::ImageBatchMsg batch);

    void attach(ipc::IPCRouter& router);

    // Draw the ImGui window.  Call once per frame inside the ImGui frame.
    void draw();

    // Number of raw anchor images stored.
    size_t sample_count() const { return entries_raw_.size(); }

    // Free all textures (raw + all modes).
    void clear();

    // Direct access to the prediction mode for palette configuration.
    PredictionImageMode& prediction_mode() { return pred_mode_; }

private:
    // Backend job state for busy spinners, set in attach()
    const ipc::JobTracker* jobs_ = nullptr;

    // Stored by ImageViewer directly so they are available to all modes without
    // any mode needing to hold a reference to another mode.
    struct RawEntry {
        uint32_t    id      = 0;
        uint32_t    width   = 0;
        uint32_t    height  = 0;
        std::string name;
        GLuint      texture = 0;
    };

    std::vector<RawEntry> entries_raw_;

    static GLuint upload_rgba(const ipc::ImageMsg& msg, GLuint existing = 0);
    static void   free_tex(GLuint& tex);
    void          upload_raw_batch(const ipc::ImageBatchMsg& batch);

    uint32_t raw_width (int idx) const;
    uint32_t raw_height(int idx) const;

    // Modes
    RawImageMode        raw_mode_;
    PredictionImageMode pred_mode_;
    HeatmapImageMode    heat_mode_;

    // All modes in a fixed array for iteration
    static constexpr int MODE_COUNT = 3;
    IImageMode* modes_[MODE_COUNT];   // raw, pred, heat — set in constructor

    IImageMode* active_mode_ = nullptr;

    // UI stuff
    ViewMode view_mode_    = ViewMode::Single;
    int      current_idx_  = 0;
    int      grid_columns_ = 3;
    bool     show_window_  = true;
    bool     initialized_  = false;

    void draw_toolbar();
    void draw_single_view();
    void draw_grid_view();

    // Return the GL texture for sample idx in the active mode.
    // Falls back to raw texture if the mode has no data for this sample.
    GLuint resolve(int idx, int display_w, int display_h);

    int active_count() const;
};
