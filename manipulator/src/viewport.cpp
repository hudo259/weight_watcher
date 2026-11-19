#include "viewport.hpp"
#include "scene.hpp"
#include "point_cloud.hpp"

#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <iostream>

void Viewport::init(GLFWwindow* window) {

    // Compute DPI scaling offset (HiDPI / UI scaling support)
    // TODO: This shit is buggy when this app runs on a system with multiple monitors and each monitor has its own fractional scaling applied.
    // GLFW will provide the resolution of the monitor that is set to main in the OS.
    // I don't know if there is a reliable way (works both on Wayland and X11, and maybe on Windows as well) to poll "which screen am I currently on" and then
    // ask which resolution the screen's framebuffer has. I would probably need to write class for this that listens to "monitor changes" and somehow notifies or updates
    // the framebuffer size (which then need to be stored somewhere public) and access those at all times for mouse movement etc.
    int win_w, win_h, fb_w, fb_h;
    glfwGetWindowSize(window, &win_w, &win_h);
    glfwGetFramebufferSize(window, &fb_w, &fb_h);

    scaling_x_ = static_cast<float>(fb_w) / static_cast<float>(win_w);
    scaling_y_ = static_cast<float>(fb_h) / static_cast<float>(win_h);

    std::cout << "[Viewport] win=" << win_w << "x" << win_h << " fb=" << fb_w << "x" << fb_h << " scaling=" << scaling_x_ << "x" << scaling_y_ << "\n";

    camera_ = std::make_unique<OrbitCamera>();
}

void Viewport::resize(int width, int height) {
    width_  = width;
    height_ = height;
}

// Per-frame
void Viewport::update(float dt) {
    camera_->update(dt);
}

void Viewport::reframe_to(glm::vec3 target, float distance) {
    if (auto* orbit = dynamic_cast<OrbitCamera*>(camera_.get())) {
        orbit->target   = target;
        orbit->distance = distance;
    } else if (auto* fly = dynamic_cast<FlyCamera*>(camera_.get())) {
        fly->pos = target + glm::vec3(0, 0, distance);
    }
}

void Viewport::draw_gizmo(Scene& scene, ipc::IPCRouter& router) {
    if (!scene.has_any_selection()) return;

    float     asp  = aspect();
    glm::mat4 view = camera_->view_matrix();
    glm::mat4 proj = camera_->projection_matrix(asp);

    ImGuizmo::SetRect(0, 0,
                      static_cast<float>(width_) / scaling_x_,
                      static_cast<float>(height_) / scaling_y_);

    glm::mat4 old_matrix = gizmo_matrix_;

    if (ImGuizmo::Manipulate(
            &view[0][0], &proj[0][0],
            gizmo_op_, gizmo_mode_,
            &gizmo_matrix_[0][0]))
    {
        glm::mat4 delta = gizmo_matrix_ * glm::inverse(old_matrix);
        glm::vec3 pivot = glm::vec3(old_matrix[3]);

        ipc::TransformedPointsBatchMsg batch;

        for (uint32_t i = 0; i < scene.cloud_count(); i++) {
            PointCloud* cloud = scene.cloud(i);
            if (!cloud || !cloud->has_selection()) continue;

            const auto& indices          = cloud->selected_indices();
            auto        positions_before = cloud->selected_positions(indices);
            cloud->apply_transform(delta, pivot);
            auto positions_after = cloud->selected_positions(indices);

            scene.push_undo({i, indices, positions_before});
            router.pending_transform.merge(i, indices,
                                           positions_after,
                                           positions_before);

            if (router.is_connected() && router.auto_send_transformed)
                batch.clouds.push_back({i, indices, positions_after});
        }

        if (!batch.clouds.empty()) {
            router.send_transformed_points_batch(batch);
        }

        gizmo_matrix_ = glm::translate(glm::mat4(1.0f),
                                        scene.selection_centroid());
    }
}

void Viewport::draw_box_select_overlay() const {
    if (!box_selecting_) return;

    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    dl->AddRect(
        ImVec2(box_start_.x / scaling_x_, box_start_.y / scaling_y_),
        ImVec2(box_end_.x / scaling_x_,   box_end_.y / scaling_y_),
        IM_COL32(255, 200, 0, 255), 0, 0, 2.0f
    );
    dl->AddRectFilled(
        ImVec2(box_start_.x / scaling_x_, box_start_.y / scaling_y_),
        ImVec2(box_end_.x / scaling_x_,   box_end_.y / scaling_y_),
        IM_COL32(255, 200, 0, 30)
    );
}

void Viewport::process_input(GLFWwindow* window, Scene& scene) {
    ImGuiIO& io = ImGui::GetIO();
    handle_camera_input(window, io);
    handle_selection(scene, io);
    handle_keyboard(scene, io);
}

void Viewport::draw_ui() {
    // Camera 
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        static const char* MODES[] = { "Orbit (F1)", "Fly (F2)" };
        int mode = static_cast<int>(camera_mode_);
        if (ImGui::Combo("Mode", &mode, MODES, 2))
            switch_camera(static_cast<CameraMode>(mode));

        glm::vec3 pos = camera_->position();
        ImGui::Text("Position: (%.1f, %.1f, %.1f)", pos.x, pos.y, pos.z);

        if (camera_mode_ == CameraMode::Orbit) {
            auto* orbit = static_cast<OrbitCamera*>(camera_.get());
            ImGui::SliderFloat("Distance", &orbit->distance,
                               0.1f, 500.0f, "%.1f",
                               ImGuiSliderFlags_Logarithmic);
            ImGui::DragFloat3("Target", &orbit->target.x, 0.1f);

            //Depending on the camera mode, we can copy and set camera extrinsics and intrinsics
            ImGui::DragFloat("Yaw",   &orbit->yaw,   0.001f, -3.14159f, 3.14159f, "%.3f");            
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                orbit->set_yaw(orbit->yaw);
            }

            ImGui::DragFloat("Pitch", &orbit->pitch, 0.001f, -1.5f,     1.5f,     "%.3f");
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                orbit->set_pitch(orbit->pitch);
            }

            // Copy camera state to clipboard as a pasteable block
            if (ImGui::Button("Copy camera state")) {
                char buf[256];
                snprintf(buf, sizeof(buf),
                    "target=(%.2f,%.2f,%.2f) dist=%.2f yaw=%.3f pitch=%.3f",
                    orbit->target.x, orbit->target.y, orbit->target.z,
                    orbit->distance, orbit->yaw, orbit->pitch);
                ImGui::SetClipboardText(buf);
            }
        } else {
            auto* fly = static_cast<FlyCamera*>(camera_.get());
            ImGui::SliderFloat("Speed", &fly->move_speed,
                               0.1f, 100.0f, "%.1f",
                               ImGuiSliderFlags_Logarithmic);

            //Depending on the camera mode, we can copy and set camera extrinsics and intrinsics
            ImGui::DragFloat3("Position", &fly->pos.x, 0.1f);

            ImGui::DragFloat("Yaw",   &fly->yaw,   0.001f, -3.14159f, 3.14159f, "%.3f");
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                fly->set_yaw(fly->yaw);
            }

            ImGui::DragFloat("Pitch", &fly->pitch, 0.001f, -1.5f,     1.5f,     "%.3f");
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                fly->set_pitch(fly->pitch);
            }

            if (ImGui::Button("Copy camera state")) {
                char buf[256];
                snprintf(buf, sizeof(buf),
                    "pos=(%.2f,%.2f,%.2f) yaw=%.3f pitch=%.3f",
                    fly->pos.x, fly->pos.y, fly->pos.z,
                    fly->yaw, fly->pitch);
                ImGui::SetClipboardText(buf);
            }
        }
    }

    // Gizmo mode 
    if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::RadioButton("Translate (W)", gizmo_op_ == ImGuizmo::TRANSLATE))
            gizmo_op_ = ImGuizmo::TRANSLATE;
        ImGui::SameLine();
        if (ImGui::RadioButton("Rotate (E)", gizmo_op_ == ImGuizmo::ROTATE))
            gizmo_op_ = ImGuizmo::ROTATE;
        ImGui::SameLine();
        if (ImGui::RadioButton("Scale (R)", gizmo_op_ == ImGuizmo::SCALE))
            gizmo_op_ = ImGuizmo::SCALE;

        if (ImGui::RadioButton("World", gizmo_mode_ == ImGuizmo::WORLD))
            gizmo_mode_ = ImGuizmo::WORLD;
        ImGui::SameLine();
        if (ImGui::RadioButton("Local", gizmo_mode_ == ImGuizmo::LOCAL))
            gizmo_mode_ = ImGuizmo::LOCAL;
    }
}

void Viewport::draw_axis_labels(const Scene& scene) const {
    const auto& labels = scene.active_axis_labels();  // {"X label", "Y label", "Z label"}
    if (labels[0].empty() && labels[1].empty() && labels[2].empty()) return;

    float asp  = aspect();
    glm::mat4 view = camera_->view_matrix();
    glm::mat4 proj = camera_->projection_matrix(asp);
    glm::mat4 vp   = proj * view;

    float W = static_cast<float>(width_)  / scaling_x_;
    float H = static_cast<float>(height_) / scaling_y_;

    // Origin in world space,  use scene center
    glm::vec3 origin(0.0f);
    float axis_len = 4.0f;  // world-space length of each axis arrow

    // World-space axis endpoints
    glm::vec3 ax = origin + glm::vec3(axis_len, 0, 0);
    glm::vec3 ay = origin + glm::vec3(0, axis_len, 0);
    glm::vec3 az = origin + glm::vec3(0, 0, axis_len);

    // Project to screen
    auto project = [&](glm::vec3 p) -> ImVec2 {
        glm::vec4 clip = vp * glm::vec4(p, 1.0f);
        if (std::abs(clip.w) < 1e-6f) return {-9999, -9999};
        glm::vec3 ndc = glm::vec3(clip) / clip.w;
        return ImVec2(
            (ndc.x * 0.5f + 0.5f) * W,
            (1.0f - (ndc.y * 0.5f + 0.5f)) * H
        );
    };

    ImVec2 o  = project(origin);
    ImVec2 px = project(ax);
    ImVec2 py = project(ay);
    ImVec2 pz = project(az);

    ImDrawList* dl = ImGui::GetBackgroundDrawList();

    // Convert the glm::vec3 color to an ImGui 32-bit color
    glm::vec3 col_vec = scene.render_settings.axis_color;
    ImU32 axis_color = IM_COL32(
        static_cast<int>(col_vec.r * 255.0f),
        static_cast<int>(col_vec.g * 255.0f),
        static_cast<int>(col_vec.b * 255.0f),
        255
    );

    // Draw arrows
    dl->AddLine(o, px, axis_color, 4.0f);  // X 
    dl->AddLine(o, py, axis_color, 4.0f);  // Y 
    dl->AddLine(o, pz, axis_color, 4.0f);  // Z

    // Draw labels at arrow tips
    float pad = 6.0f;
    auto tip_label = [&](ImVec2 tip, const std::string& label, ImU32 col) {
    if (!label.empty())
        dl->AddText(
            ImGui::GetFont(),   // current font
            38.0f,              // font size
            ImVec2(tip.x + pad, tip.y - pad),
            col,
            label.c_str()
        );
    };

    tip_label(px, labels[0], axis_color);
    tip_label(py, labels[1], axis_color);
    tip_label(pz, labels[2], axis_color);
}

glm::mat4 Viewport::view_matrix()       const { return camera_->view_matrix(); }
glm::mat4 Viewport::projection_matrix() const { return camera_->projection_matrix(aspect()); }
glm::mat4 Viewport::vp_matrix()         const { return projection_matrix() * view_matrix(); }
glm::vec3 Viewport::camera_position()   const { return camera_->position(); }
Frustum   Viewport::frustum()           const { return camera_->frustum(aspect()); }
float     Viewport::aspect()            const {
    return static_cast<float>(width_) / static_cast<float>(height_);
}

// input sub-handlers
void Viewport::handle_camera_input(GLFWwindow* window, ImGuiIO& io) {
    // Feed WASD keys into fly camera
    if (camera_mode_ == CameraMode::Fly) {
        auto* fly = static_cast<FlyCamera*>(camera_.get());
        fly->move_forward  = glfwGetKey(window, GLFW_KEY_W)             == GLFW_PRESS;
        fly->move_backward = glfwGetKey(window, GLFW_KEY_S)             == GLFW_PRESS;
        fly->move_left     = glfwGetKey(window, GLFW_KEY_A)             == GLFW_PRESS;
        fly->move_right    = glfwGetKey(window, GLFW_KEY_D)             == GLFW_PRESS;
        fly->move_up       = glfwGetKey(window, GLFW_KEY_SPACE)         == GLFW_PRESS;
        fly->move_down     = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL)  == GLFW_PRESS;
        fly->sprinting     = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)    == GLFW_PRESS;
    }

    if (io.WantCaptureMouse) {
        box_selecting_ = false;
        return;
    }

    //Wayland should deliver mouse coordinates already scaled. 
    //So the following might be necessary for X11
    glm::vec2 mouse(io.MousePos.x * scaling_x_,
                    io.MousePos.y * scaling_y_);
    //glm::vec2 mouse(io.MousePos.x, io.MousePos.y);
    glm::vec2 delta = mouse - last_mouse_;
    last_mouse_     = mouse;

    bool shift  = io.KeyShift;
    bool alt    = io.KeyAlt;
    //For some reason wayland doesn't seem to report middle mouse button?
    bool middle = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    bool right  = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)  == GLFW_PRESS;
    bool left   = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)   == GLFW_PRESS;
    //std::cout << "middle=" << middle << " right=" << right << " left=" << left
    //          << " WantCaptureMouse=" << io.WantCaptureMouse
    //          << " IsUsing=" << ImGuizmo::IsUsing()
    //          << " box_selecting=" << box_selecting_
    //          << " delta=" << delta.x << "," << delta.y << "\n";

    if (io.MouseWheel != 0.0f && !ImGuizmo::IsUsing())
        camera_->on_scroll(io.MouseWheel);

    // Orbit: middle drag or alt+left drag or right drag
    bool right_drag_orbit = right && !alt 
                        && camera_mode_ == CameraMode::Orbit
                        && !box_selecting_;
    if ((middle || (alt && left) || right_drag_orbit) && !ImGuizmo::IsUsing() && !box_selecting_)
        camera_->on_mouse_move(delta.x, delta.y, left, middle, right, shift);

    // Fly: right drag to look
    if (camera_mode_ == CameraMode::Fly && right && !alt)
        camera_->on_mouse_move(delta.x, delta.y, left, middle, right, shift);
}

void Viewport::handle_selection(Scene& scene, ImGuiIO& io) {
    if (io.WantCaptureMouse) return;

    bool shift  = io.KeyShift;
    bool ctrl   = io.KeyCtrl;
    bool alt    = io.KeyAlt;
    bool middle = ImGui::IsMouseDown(ImGuiMouseButton_Middle);

    //Wayland should deliver mouse coordinates already scaled. 
    //So the following might be necessary for X11
    glm::vec2 mouse(io.MousePos.x * scaling_x_,
                    io.MousePos.y * scaling_y_);
    //glm::vec2 mouse(io.MousePos.x, io.MousePos.y);

    // Box selection (right drag)
    if (alt && !middle && !ImGuizmo::IsUsing())
    {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
            box_selecting_ = true;
            box_start_     = mouse;
            box_end_       = box_start_;
        }

        if (box_selecting_) {
            box_end_ = mouse;

            if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
                box_selecting_ = false;
                
                Frustum fr = frustum();
                scene.render_pick_pass(
                    vp_matrix(), view_matrix(),
                    scene.render_settings.point_size,
                    scene.render_settings.use_frustum_culling
                        ? &fr : nullptr
                );

                auto selections = scene.box_select(
                    static_cast<int>(box_start_.x),
                    static_cast<int>(box_start_.y),
                    static_cast<int>(box_end_.x),
                    static_cast<int>(box_end_.y)
                );

                for (auto& [cloud_idx, indices] : selections)
                    scene.cloud(cloud_idx)->select_multiple(indices, shift);

                if (scene.has_any_selection())
                    gizmo_matrix_ = glm::translate(glm::mat4(1.0f),
                                                   scene.selection_centroid());
            }
        }
    }

    // Single click selection 

    bool can_select = camera_mode_ == CameraMode::Fly 
                  ? true          // in fly mode always allow left click select
                  : !alt;         // in orbit mode block when alt is held (used for orbit)
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)
        && can_select && !ImGuizmo::IsOver() && !ImGuizmo::IsUsing())
    {

        //std::cout << "[pick] mouse=" << io.MousePos.x << "," << io.MousePos.y
        //  << " viewport=" << width_ << "x" << height_
        //  << " scaling=" << scaling_x_ << "x" << scaling_y_ << "\n";

        // Need a temporary Frustum for pick pass
        Frustum fr = frustum();
        scene.render_pick_pass(
            vp_matrix(), view_matrix(),
            scene.render_settings.point_size,
            scene.render_settings.use_frustum_culling ? &fr : nullptr
        );

        auto pick = scene.pick_at(static_cast<int>(mouse.x),
                                  static_cast<int>(mouse.y));

        if (pick.valid && pick.cloud) {
            if (ctrl) {
                pick.cloud->toggle_selection(pick.point_index);
            } else {
                if (!shift) scene.clear_all_selections();
                pick.cloud->select(pick.point_index, true);
            }
            if (scene.has_any_selection())
                gizmo_matrix_ = glm::translate(glm::mat4(1.0f),
                                               scene.selection_centroid());
        } else if (!shift && !ctrl) {
            scene.clear_all_selections();
        }
    }
}

void Viewport::handle_keyboard(Scene& scene, ImGuiIO& io) {
    if (io.WantCaptureKeyboard) return;

    bool ctrl = io.KeyCtrl;

    // Gizmo mode shortcuts (W only in non-fly mode to avoid WASD conflict)
    if (camera_mode_ != CameraMode::Fly) {
        if (ImGui::IsKeyPressed(ImGuiKey_W)) gizmo_op_ = ImGuizmo::TRANSLATE;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_E)) gizmo_op_ = ImGuizmo::ROTATE;
    if (ImGui::IsKeyPressed(ImGuiKey_R)) gizmo_op_ = ImGuizmo::SCALE;

    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) scene.clear_all_selections();

    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_A)) {
        for (size_t i = 0; i < scene.cloud_count(); i++)
            if (scene.cloud(i)) scene.cloud(i)->select_all();
        if (scene.has_any_selection())
            gizmo_matrix_ = glm::translate(glm::mat4(1.0f),
                                           scene.selection_centroid());
    }

    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_I)) {
        for (size_t i = 0; i < scene.cloud_count(); i++)
            if (scene.cloud(i)) scene.cloud(i)->invert_selection();
        if (scene.has_any_selection())
            gizmo_matrix_ = glm::translate(glm::mat4(1.0f),
                                           scene.selection_centroid());
    }

    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_Z)) scene.undo();
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_Y)) scene.redo();

    // Camera mode shortcuts
    if (ImGui::IsKeyPressed(ImGuiKey_F1)) switch_camera(CameraMode::Orbit);
    if (ImGui::IsKeyPressed(ImGuiKey_F2)) switch_camera(CameraMode::Fly);
}

void Viewport::switch_camera(CameraMode mode) {
    if (mode == camera_mode_) return;

    glm::vec3 old_pos     = camera_->position();
    glm::vec3 old_forward = camera_->forward();
    camera_mode_          = mode;

    if (mode == CameraMode::Orbit) {
        auto cam      = std::make_unique<OrbitCamera>();
        cam->target   = old_pos + old_forward * 20.0f;
        cam->distance = 20.0f;
        camera_       = std::move(cam);
    } else {
        auto cam  = std::make_unique<FlyCamera>();
        cam->pos  = old_pos;
        cam->yaw  = std::atan2(old_forward.z, old_forward.x);
        cam->pitch = std::asin(old_forward.y);
        camera_   = std::move(cam);
    }
}

void Viewport::apply_state(const AppState& state) {
    switch_camera(static_cast<CameraMode>(state.camera_mode));
    //camera_->set_yaw(state.yaw);
    //camera_->set_pitch(state.pitch);
    camera_->yaw = state.yaw;
    camera_->pitch = state.pitch;
    
    if (camera_mode_ == CameraMode::Orbit) {
        auto* orbit = static_cast<OrbitCamera*>(camera_.get());
        orbit->target = state.target;
        orbit->distance = state.distance;
    } else {
        auto* fly = static_cast<FlyCamera*>(camera_.get());
        fly->pos = state.pos;
    }
}

void Viewport::extract_state(AppState& state) const {
    state.camera_mode = static_cast<int>(camera_mode_);
    state.yaw = camera_->yaw;
    state.pitch = camera_->pitch;
    
    if (camera_mode_ == CameraMode::Orbit) {
        auto* orbit = static_cast<OrbitCamera*>(camera_.get());
        state.target = orbit->target;
        state.distance = orbit->distance;
    } else {
        auto* fly = static_cast<FlyCamera*>(camera_.get());
        state.pos = fly->pos;
    }

    state.fb_width = width_;
    state.fb_height = height_;
}
