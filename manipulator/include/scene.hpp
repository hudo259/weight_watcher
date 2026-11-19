#pragma once

#include "gpu_point_buffer.hpp"
#include "ipc_messages.hpp"
#include "ipc_router.hpp"
#include "point_cloud.hpp"
#include "pick_buffer.hpp"
#include "shader.hpp"
#include "frustum.hpp"
#include "colormaps.hpp"
#include "app_state.hpp"

#include <glm/glm.hpp>
#include <glad/gl.h>

#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

/**
 * # Notes so that I don't forget later:
 *
 * ## GPU ownership
 * All point data lives in a single GpuPointBuffer (VBO, heatmap VBO, VAO,
 * selection SSBO).  Each cloud owns a contiguous sub-range tracked in slots_.
 *
 * ## Dirty-flag stuff
 * PointCloud never touches the GPU.  render() polls positions_dirty_ and
 * selection_dirty_ and delegates uploads to buffer_.
 *
 * ## Cloud indices
 * remove_cloud() leaves a tombstone so existing indices in undo entries and
 * IPC messages stay valid.  cloud() returns nullptr for tombstones.
 */

class Scene {
public:

    struct LightSettings {
        glm::vec3 direction = glm::normalize(glm::vec3(0.5f, 0.8f, 1.0f));
        glm::vec3 color     = glm::vec3(1.0f);
        float ambient   = 0.25f;
        float diffuse   = 0.65f;
        float specular  = 0.4f;
        float shininess = 32.0f;
    };

    struct RenderSettings {
        //color of selected points
        glm::vec3 color_selected      = glm::vec3(1.0f, 0.8f, 0.0f);

        float     point_size          = 8.0f;
        bool      fake_sphere_shading = true;
        bool      soft_edges          = false;
        bool      use_frustum_culling = true;

        //color of background
        glm::vec3 bg_color = glm::vec3(255.0f, 255.0f, 255.0f);

        //color of coordinate axes and labels for the current layout
        glm::vec3 axis_color = glm::vec3(0.0f, 0.0f, 0.0f);
    };

    struct LayoutSettings {
        std::unordered_map<uint32_t, std::array<std::string, 3>> layout_axis_labels_;
        std::unordered_map<uint32_t, std::vector<glm::vec3>> received_layouts_;
        std::unordered_map<uint32_t, std::string> layout_names_;
        uint32_t active_layout_id_ = ipc::LayoutID::DEPTH;
        glm::vec3 layout_scale_ = glm::vec3(1.0f);
        // Layer isolation
        std::optional<uint32_t> isolated_layer_idx_;   // absent = show all
        bool layer_isolation_active_ = false;
        bool reframe_pending_ = false;
        glm::vec3 reframe_target_ = glm::vec3(0);
    };

    struct Stats {
        size_t total_points    = 0;
        size_t selected_points = 0;
        size_t visible_points  = 0;
        size_t total_chunks    = 0;
        size_t visible_chunks  = 0;
    };

    struct ResolvedPick {
        bool        valid       = false;
        uint32_t    cloud_index = 0;
        uint32_t    point_index = 0;
        PointCloud* cloud       = nullptr;
    };

    struct UndoEntry {
        uint32_t               cloud_id;
        std::vector<uint32_t>  indices;
        std::vector<glm::vec3> positions;
    };

    enum class ActivationAggregation { Average = 0, Maximum = 1 };

    LightSettings  light;
    RenderSettings render_settings;
    LayoutSettings layout_settings;
    mutable Stats  stats;

    uint32_t defrag_threshold = 0;

    // Callbacks set by attach(), kept public so main.cpp can clear them to
    // nullptr during ordered shutdown before the IPCRouter destructs.
    std::function<void(int)>               on_concept_select;
    std::function<void(std::vector<float>&&)> on_activation_request;

    //Non-IPC related callback.
    std::function<void(glm::vec3)> on_reframe_request_;

    static constexpr glm::vec3 LAYER_PALETTE[] = {
        { 0.4f, 0.6f, 0.9f }, { 0.9f, 0.5f, 0.3f }, { 0.4f, 0.8f, 0.5f },
        { 0.9f, 0.4f, 0.7f }, { 0.7f, 0.5f, 0.9f }, { 0.9f, 0.8f, 0.3f },
    };

    const std::array<std::string, 3>& active_axis_labels() const {
        static const std::array<std::string, 3> empty = {"", "", ""};
        auto it = layout_settings.layout_axis_labels_.find(layout_settings.active_layout_id_);
        return it != layout_settings.layout_axis_labels_.end() ? it->second : empty;
    }

    // Lifecycle
    void init(int width, int height);
    void resize(int width, int height);
    ~Scene();

    // Wire IPC subscriptions and outgoing callbacks.
    // Call once after IPCRouter::init().
    void attach(ipc::IPCRouter& router);

    // Cloud management
    void load_layers(std::vector<glm::vec3>  points,
                     std::vector<std::string> layer_names,
                     std::vector<uint32_t>    layer_counts,
                     std::vector<glm::vec3>   layer_colors);

    uint32_t add_cloud(std::vector<glm::vec3> points,
                       std::string            name  = "",
                       glm::vec3              color = glm::vec3(0));

    void remove_cloud(uint32_t cloud_idx);
    void defrag();

    // Concept heatmap

    // Upload raw scores [-1, 1] to the heatmap VBO.
    // Colormap is applied per-vertex in the shader via u_Colormap uniform.
    void set_concept_scores(const ipc::VectorListMsg& msg);

    void clear_concept_scores();

    // Rendering
    void render(const glm::mat4& view, const glm::mat4& proj,
                const glm::vec3& camera_pos, const Frustum* frustum = nullptr);

    void render_pick_pass(const glm::mat4& vp, const glm::mat4& view,
                          float point_size, const Frustum* frustum = nullptr);

    ResolvedPick pick_at(int x, int y);
    std::unordered_map<uint32_t, std::vector<uint32_t>> box_select(
        int x0, int y0, int x1, int y1);

    // Selection
    PointCloud* cloud(uint32_t index);
    size_t      cloud_count()      const;
    size_t      live_cloud_count() const;

    void      clear_all_selections();
    bool      has_any_selection()  const;
    glm::vec3 selection_centroid() const;
    void      apply_transform_to_selection(const glm::mat4& delta,
                                           const glm::vec3& pivot);

    // UI
    // Draw all scene panels inside the main ImGui window.
    // (all state lives in ui_).
    void draw_ui_contents();

    // Draw the selection buttons panel.
    // Returns true if the selection changed (caller should update gizmo matrix).
    bool draw_selection_ui_contents();

    void set_layer_isolation(std::optional<uint32_t> cloud_idx);
    glm::vec3 cloud_aabb_center(uint32_t cloud_idx) const;
    bool has_pending_reframe() const { return layout_settings.reframe_pending_; }
    glm::vec3 consume_reframe_target() { layout_settings.reframe_pending_ = false; return layout_settings.reframe_target_; }

    // Undo / Redo
    void push_undo(UndoEntry entry);
    void undo();
    void redo();

    //extract and apply session states from when reading and saving a session file
    void apply_state(const AppState& state);
    void extract_state(AppState& state) const;

private:
    struct CloudSlot {
        GpuPointBuffer::Slot gpu;
        glm::vec3 layout_offset{0.0f};
        bool alive = false;
    };

    // All UI variables in one place.
    // Initialised/reset with load_layers() so stale values never persist
    // across reconnects or scene reloads.
    struct UIState {
        // Concept heatmap panel
        int            selected_class = 0;
        ColorMaps::ColorMapType colormap      = ColorMaps::ColorMapType::Viridis;

        // Spatial activation panel
        ActivationAggregation aggregation = ActivationAggregation::Average;

        // Generate panel
        int   num_points  = 500000;
        float point_range = 10.0f;
        int   random_seed = 42;
    };

    struct LegendEntry { std::string name; glm::vec3 color; };
    std::vector<LegendEntry> legend_entries_;
    // Private helpers

    static uint32_t encode_group(uint32_t cloud_idx, uint32_t chunk_idx) {
        return ((cloud_idx & 0x3FFF) << 18) | (chunk_idx & 0x3FFFF);
    }
    static void decode_group(uint32_t group_id,
                              uint32_t& cloud_idx, uint32_t& chunk_idx) {
        cloud_idx = (group_id >> 18) & 0x3FFF;
        chunk_idx =  group_id        & 0x3FFFF;
    }

    ResolvedPick resolve(const PickResult& pick);
    void         setup_main_shader(const glm::mat4& view,
                                   const glm::mat4& proj,
                                   const glm::vec3& camera_pos);
    void         compile_shaders();

    void apply_layout(uint32_t layout_id);

    void upload_cloud_positions(uint32_t cloud_idx);
    void sync_cloud_selection  (uint32_t cloud_idx);
    void rebuild_selection_ssbo();

    // UI sub-panels (each one CollapsingHeader block)
    void draw_ui_stats();
    void draw_ui_rendering();
    void draw_ui_lighting();
    void draw_ui_layout();
    void draw_ui_concept_heatmap();
    void draw_ui_spatial_activation();
    void draw_ui_buffer();
    void draw_ui_generate();
    void draw_ui_legend();
    void draw_ui_layers();
    void draw_ui_controls();

    // Private data

    // GPU resources
    GpuPointBuffer buffer_;
    GLuint         color_ssbo_     = 0;
    GLuint layout_offset_ssbo_ = 0;
    bool           heatmap_active_ = false;

    // Shaders 
    std::unique_ptr<Shader> main_frag_;
    std::unique_ptr<Shader> main_vert_;
    ShaderPipeline          main_shader_pipeline_;

    std::unique_ptr<Shader> pick_frag_;
    std::unique_ptr<Shader> pick_vert_;
    ShaderPipeline          pick_shader_pipeline_;

    // Cloud data 
    std::vector<CloudSlot>                   slots_;
    std::vector<std::unique_ptr<PointCloud>> clouds_;
    uint32_t                                 removal_count_ = 0;

    PickBuffer pick_buffer_;

    // Undo / redo 
    std::vector<UndoEntry> undo_stack_;
    std::vector<UndoEntry> redo_stack_;

    // IPC buffering
    // Stores the LayerMetaMsg received before the PointCloud arrives.
    std::optional<ipc::LayerMetaMsg> pending_layer_meta_;

    // Number of classes, set from LayerMetaMsg.num_classes in attach().
    // Only used by draw_ui_concept_heatmap() (intentionally private!)
    int num_classes_ = 0;

    // UI state
    UIState ui_;

    //Backend job state for busy spinners, set in attach()
    const ipc::JobTracker* jobs_ = nullptr;
};
