#include "feature_viewer.hpp"
#include "busy_section.hpp"

#include <imgui.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

// Lifecycle stuff
FeatureViewer::~FeatureViewer() {
    clear();
    destroy_fbo();
}

void FeatureViewer::init() {
    compile_shader();
    camera_.distance = 20.0f;
    camera_.target = glm::vec3(0.0f);
    initialized_ = true;
}

void FeatureViewer::clear() {
    features_ = {};
    points_.clear();
    colors_.clear();
    point_count_ = 0;
    destroy_gpu();
    needs_focus_ = true;
}

// Data
void FeatureViewer::set_features(ipc::VectorListMsg msg) {
    features_ = std::move(msg);

    // Clamp dimension indices to valid range
    int max_dim = static_cast<int>(features_.dimensions) - 1;
    if (max_dim < 0) {
        clear();
        return;
    }
    dim_x_ = std::min(dim_x_, max_dim);
    dim_y_ = std::min(dim_y_, max_dim);
    dim_z_ = std::min(dim_z_, max_dim);

    // For 1D data, map to X only
    if (features_.dimensions == 1) {
        dim_x_ = 0; dim_y_ = 0; dim_z_ = 0;
    }
    // For 2D data, map to XY
    else if (features_.dimensions == 2) {
        dim_x_ = 0; dim_y_ = 1; dim_z_ = 0;
    }
    // For 3D+, default first time
    else {
        dim_x_ = std::min(dim_x_, max_dim);
        dim_y_ = std::min(dim_y_, max_dim);
        dim_z_ = std::min(dim_z_, max_dim);
    }

    rebuild_points();
    upload_to_gpu();
    show_window_ = true;
    needs_focus_ = true;
}

void FeatureViewer::rebuild_points() {
    size_t count = features_.count();
    if (count == 0) {
        points_.clear();
        colors_.clear();
        point_count_ = 0;
        return;
    }

    uint32_t dims = features_.dimensions;
    points_.resize(count);
    colors_.resize(count);

    bounds_min_ = glm::vec3(std::numeric_limits<float>::max());
    bounds_max_ = glm::vec3(std::numeric_limits<float>::lowest());

    for (size_t i = 0; i < count; ++i) {
        const float* v = features_.vector(i);

        float x = (dim_x_ < static_cast<int>(dims)) ? v[dim_x_] : 0.0f;
        float y = (dim_y_ < static_cast<int>(dims)) ? v[dim_y_] : 0.0f;
        float z = (dim_z_ < static_cast<int>(dims)) ? v[dim_z_] : 0.0f;

        points_[i] = glm::vec3(x, y, z);
        colors_[i] = rainbow(static_cast<float>(i) / static_cast<float>(std::max(count - 1, size_t(1))));

        bounds_min_ = glm::min(bounds_min_, points_[i]);
        bounds_max_ = glm::max(bounds_max_, points_[i]);
    }

    point_count_ = count;
}

glm::vec3 FeatureViewer::rainbow(float t) {
    // HSV-like rainbow: red -> yellow -> green -> cyan -> blue -> magenta
    float h = t * 5.0f;
    float r = std::clamp(std::abs(h - 3.0f) - 1.0f, 0.0f, 1.0f);
    float g = std::clamp(2.0f - std::abs(h - 2.0f), 0.0f, 1.0f);
    float b = std::clamp(2.0f - std::abs(h - 4.0f), 0.0f, 1.0f);
    return glm::vec3(r, g, b);
}

// GL resources: point cloud
void FeatureViewer::upload_to_gpu() {
    destroy_gpu();

    if (points_.empty()) return;

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_pos_);
    glGenBuffers(1, &vbo_color_);

    glBindVertexArray(vao_);

    // Positions -> attribute 0
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(points_.size() * sizeof(glm::vec3)), points_.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);

    // Colors -> attribute 1
    glBindBuffer(GL_ARRAY_BUFFER, vbo_color_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(colors_.size() * sizeof(glm::vec3)), colors_.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);

    glBindVertexArray(0);
}

void FeatureViewer::destroy_gpu() {
    if (vao_) { glDeleteVertexArrays(1, &vao_); vao_ = 0; }
    if (vbo_pos_) { glDeleteBuffers(1, &vbo_pos_); vbo_pos_ = 0; }
    if (vbo_color_) { glDeleteBuffers(1, &vbo_color_); vbo_color_ = 0; }
}

// GL resources: FBO
void FeatureViewer::ensure_fbo(int width, int height) {
    if (width <= 0 || height <= 0) return;
    if (fbo_ && fbo_width_ == width && fbo_height_ == height) return;

    destroy_fbo();

    fbo_width_ = width;
    fbo_height_ = height;

    // Color texture
    glGenTextures(1, &color_tex_);
    glBindTexture(GL_TEXTURE_2D, color_tex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Depth renderbuffer
    glGenRenderbuffers(1, &depth_rbo_);
    glBindRenderbuffer(GL_RENDERBUFFER, depth_rbo_);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);

    // FBO
    glGenFramebuffers(1, &fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex_, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rbo_);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "[FeatureViewer] FBO incomplete: 0x" << std::hex << status << std::dec << "\n";
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FeatureViewer::destroy_fbo() {
    if (fbo_) { glDeleteFramebuffers(1, &fbo_); fbo_ = 0; }
    if (color_tex_) { glDeleteTextures(1, &color_tex_); color_tex_ = 0; }
    if (depth_rbo_) { glDeleteRenderbuffers(1, &depth_rbo_); depth_rbo_ = 0; }
    fbo_width_ = fbo_height_ = 0;
}

// Rendering
void FeatureViewer::render_to_fbo(int width, int height) {
    ensure_fbo(width, height);
    if (!fbo_ || !vao_ || point_count_ == 0) return;

    // Auto-focus camera on first render after new data
    if (needs_focus_) {
        float radius = glm::length(bounds_max_ - bounds_min_) * 0.5f;
        camera_.focus_on((bounds_min_ + bounds_max_) * 0.5f, std::max(radius, 0.1f));
        needs_focus_ = false;
    }

    // Save current GL state
    GLint prev_fbo;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
    GLint prev_viewport[4];
    glGetIntegerv(GL_VIEWPORT, prev_viewport);

    // Bind our FBO
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glViewport(0, 0, width, height);
    glClearColor(bg_color_.r, bg_color_.g, bg_color_.b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);

    float aspect = static_cast<float>(width) / static_cast<float>(height);
    glm::mat4 view = camera_.view_matrix();
    glm::mat4 proj = camera_.projection_matrix(aspect);

    pipeline_.bind();
    shader_vert_->set_mat4("u_View", view);
    shader_vert_->set_mat4("u_Proj", proj);
    shader_vert_->set_float("u_PointSize", point_size_);
    shader_frag_->set_vec3("u_LightDir", glm::normalize(glm::vec3(0.5f, 0.8f, 1.0f)));
    shader_frag_->set_vec3("u_CameraPos", camera_.position());

    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(point_count_));
    glBindVertexArray(0);

    // Restore previous GL state
    glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(prev_fbo));
    glViewport(prev_viewport[0], prev_viewport[1], prev_viewport[2], prev_viewport[3]);
}

// Input
void FeatureViewer::process_input() {
    if (!ImGui::IsWindowHovered()) {
        first_hover_ = true;
        return;
    }

    ImGuiIO& io = ImGui::GetIO();
    glm::vec2 mouse(io.MousePos.x, io.MousePos.y);

    // Reset delta tracking on first hover frame to avoid jump
    if (first_hover_) {
        last_mouse_ = mouse;
        first_hover_ = false;
    }

    glm::vec2 delta = mouse - last_mouse_;
    last_mouse_ = mouse;

    // Scroll to zoom
    if (io.MouseWheel != 0.0f) {
        camera_.on_scroll(io.MouseWheel);
    }

    bool middle = ImGui::IsMouseDown(ImGuiMouseButton_Middle);
    bool right = ImGui::IsMouseDown(ImGuiMouseButton_Right);
    bool left = ImGui::IsMouseDown(ImGuiMouseButton_Left);
    bool shift = io.KeyShift;
    bool alt = io.KeyAlt;

    // Orbit: middle drag, or alt+left drag
    if (middle || (alt && left)) {
        camera_.on_mouse_move(delta.x, delta.y, left, middle, right, shift);
    }
    // Right drag also orbits (matches main viewport)
    else if (right) {
        camera_.on_mouse_move(delta.x, delta.y, false, false, true, shift);
    }
}

void FeatureViewer::attach(ipc::IPCRouter& router) {
    jobs_ = &router.jobs();

    router.on_vector_list(ipc::VectorListID::FEATURES,
        [this](ipc::VectorListMsg msg) { set_features(std::move(msg)); });
}

// ImGui window
void FeatureViewer::draw() {
    if (!initialized_) return;
    if (!has_data()) return;
    if (!show_window_) return;

    ImGui::Begin("Feature Viewer", &show_window_);

    // Settings bar
    bool dim_changed = false;
    int max_dim = static_cast<int>(features_.dimensions) - 1;

    if (features_.dimensions > 3) {
        ImGui::Text("Dims: %u  Points: %zu", features_.dimensions, point_count_);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(50);
        dim_changed |= ImGui::InputInt("X", &dim_x_, 0);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(50);
        dim_changed |= ImGui::InputInt("Y", &dim_y_, 0);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(50);
        dim_changed |= ImGui::InputInt("Z", &dim_z_, 0);

        dim_x_ = std::clamp(dim_x_, 0, max_dim);
        dim_y_ = std::clamp(dim_y_, 0, max_dim);
        dim_z_ = std::clamp(dim_z_, 0, max_dim);

        if (dim_changed) {
            rebuild_points();
            upload_to_gpu();
            needs_focus_ = true;
        }
    } else {
        ImGui::Text("Dims: %u  Points: %zu", features_.dimensions, point_count_);
    }

    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    ImGui::SliderFloat("Size", &point_size_, 1.0f, 20.0f, "%.0f");

    ImGui::SameLine();
    if (ImGui::Button("Reset View")) {
        needs_focus_ = true;
    }

    // Render area: fill remaining window space
    ImVec2 avail = ImGui::GetContentRegionAvail();
    int w = static_cast<int>(avail.x);
    int h = static_cast<int>(avail.y);

    if (w > 0 && h > 0) {
        //Features are recomputed after every weight change...
        const bool busy = jobs_ && (jobs_->busy(ipc::JobType::TRANSFORMED_POINTS) || jobs_->busy(ipc::JobType::POINT_CLOUD));
        ui::BusySection section(busy, "Updating features...", ui::BusyInput::ALLOW);

        render_to_fbo(w, h);

        if (color_tex_) {
            // Display FBO texture. UV flipped vertically because FBO origin is bottom-left.
            ImGui::Image(
                static_cast<ImTextureID>(static_cast<uintptr_t>(color_tex_)),
                avail,
                ImVec2(0, 1), ImVec2(1, 0)  // flip Y
            );
        }
    }

    // Handle input (must be after Image so IsWindowHovered works correctly)
    process_input();

    ImGui::End();
}

// Shader
// TODO: should be moved into its own dedicated file (and maybe do a runtime loader / compile)?
void FeatureViewer::compile_shader() {
    const char* vert = R"(
        #version 430 core

        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aColor;

        uniform mat4 u_View;
        uniform mat4 u_Proj;
        uniform float u_PointSize;

        layout(location = 0) out vec3 vColor;
        out gl_PerVertex {
            vec4 gl_Position;
            float gl_PointSize;
        };

        void main() {
            vColor = aColor;
            vec4 viewPos = u_View * vec4(aPos, 1.0);
            gl_Position = u_Proj * viewPos;

            float dist = -viewPos.z;
            gl_PointSize = u_PointSize * (10.0 / max(dist, 0.1));
        }
    )";

    const char* frag = R"(
        #version 430 core

        layout(location = 0) in vec3 vColor;
        out vec4 FragColor;

        uniform vec3 u_LightDir;
        uniform vec3 u_CameraPos;

        void main() {
            // Circular point sprite
            vec2 coord = gl_PointCoord * 2.0 - 1.0;
            float dist_sq = dot(coord, coord);
            if (dist_sq > 1.0) discard;

            // Fake sphere normal
            float z = sqrt(1.0 - dist_sq);
            vec3 normal = vec3(coord.x, -coord.y, z);

            // Simple lighting: ambient + diffuse
            float ambient = 0.3;
            float diff = max(dot(normal, u_LightDir), 0.0);
            vec3 lit = vColor * (ambient + 0.7 * diff);

            FragColor = vec4(lit, 1.0);
        }
    )";
    shader_vert_ = std::make_unique<Shader>(Shader::from_string(GL_VERTEX_SHADER, vert));
    shader_frag_ = std::make_unique<Shader>(Shader::from_string(GL_FRAGMENT_SHADER, frag));
    pipeline_.attach(*shader_vert_).attach(*shader_frag_);
}
