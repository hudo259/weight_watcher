#pragma once

#include "ipc_messages.hpp"
#include "ipc_router.hpp"
#include "orbit_camera.hpp"
#include "shader.hpp"

#include <glad/gl.h>
#include <glm/glm.hpp>

#include <vector>
#include <string>
#include <cstdint>

/**
 * @brief Renders feature vectors as a 3D point cloud in its own ImGui window.
 *
 * Receives VectorListMsg from IPC, maps 3 chosen dimensions to XYZ,
 * colors points by index (rainbow gradient), and renders into an FBO
 * displayed as an ImGui::Image(). Has its own OrbitCamera with input
 * handled when the window is hovered.
 *
 * Use like this:
 *   FeatureViewer features;
 *   features.init();           // After GL context is ready
 *   // In IPC callback:
 *   features.set_features(std::move(msg));
 *   // In render loop:
 *   features.draw();
 */
class FeatureViewer {
public:
    ~FeatureViewer();

    /// Must be called after GL context is created (compiles shaders).
    void init();

    /// Replace current features with new data.
    void set_features(ipc::VectorListMsg msg);

    /// Draw the ImGui window (renders point cloud, handles input). 
    void draw();

    /// Clear all data and free GL resources.
    void clear();

    bool has_data() const { return point_count_ > 0; }

    void attach(ipc::IPCRouter& router);
private:
    // Backend job state for busy spinners, set in attach()
    const ipc::JobTracker* jobs_ = nullptr;

    // Convert VectorListMsg to 3D points using current dimension mapping
    void rebuild_points();

    // Upload points_ and colors_ to GPU
    void upload_to_gpu();

    // Ensure FBO matches the given size
    void ensure_fbo(int width, int height);

    // Render the point cloud into the FBO
    void render_to_fbo(int width, int height);

    // Handle orbit camera input when window is hovered
    void process_input();

    // Generate rainbow color from normalized value [0, 1]
    static glm::vec3 rainbow(float t);

    // Compile the simple point cloud shader
    void compile_shader();

    void destroy_gpu();
    void destroy_fbo();

    // Raw feature data
    ipc::VectorListMsg features_;

    // Dimension mapping: which feature dimensions map to X, Y, Z
    int dim_x_ = 0;
    int dim_y_ = 1;
    int dim_z_ = 2;

    // 3D point data (derived from features_ + dimension mapping)
    std::vector<glm::vec3> points_;
    std::vector<glm::vec3> colors_;
    size_t point_count_ = 0;

    // Bounds (for auto-focus)
    glm::vec3 bounds_min_{0};
    glm::vec3 bounds_max_{0};

    // Camera
    OrbitCamera camera_;
    bool needs_focus_ = true;
    glm::vec2 last_mouse_{0};
    bool first_hover_ = true;

    // Render settings
    float point_size_ = 6.0f;
    glm::vec3 bg_color_{0.12f, 0.12f, 0.15f};

    // GL resources: point cloud
    GLuint vao_ = 0;
    GLuint vbo_pos_ = 0;
    GLuint vbo_color_ = 0;

    // GL resources: FBO
    GLuint fbo_ = 0;
    GLuint color_tex_ = 0;
    GLuint depth_rbo_ = 0;
    int fbo_width_ = 0;
    int fbo_height_ = 0;

    // Shader
    std::unique_ptr<Shader> shader_vert_;
    std::unique_ptr<Shader> shader_frag_;
    ShaderPipeline pipeline_;
    bool initialized_ = false;

    // UI state
    bool show_window_ = true;
};
