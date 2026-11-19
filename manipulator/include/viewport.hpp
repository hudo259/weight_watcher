#pragma once

#include "icamera.hpp"
#include "orbit_camera.hpp"
#include "fly_camera.hpp"
#include "frustum.hpp"
#include "ipc_router.hpp"
#include "app_state.hpp"

#include <imgui.h>
#include <ImGuizmo.h>

//We already load gl.h via GLAD in scene later
#define GLFW_INCLUDE_NONE 
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <memory>

// Forward declaration
class Scene;

/**
 * Usage in main loop:
 *   viewport.update(dt);                        // camera physics
 *   scene.render(viewport.view(), viewport.proj(), viewport.camera_pos(), ...);
 *   viewport.draw_gizmo(scene, router);         // ImGuizmo overlay
 *   // ... draw ImGui windows ...
 *   viewport.process_input(window, scene);      // AFTER ImGui draws
 *   viewport.draw_box_select_overlay();         // background draw list rect
 */
class Viewport {
public:
    enum class CameraMode { Orbit, Fly };

    void init(GLFWwindow* window);
    void resize(int width, int height);

    // Per-frame

    // Update camera physics (call at the start of the frame).
    void update(float dt);

    // Render the ImGuizmo gizmo and apply transform to scene selection.
    // Also merges into router.pending_transform and auto-sends if enabled.
    void draw_gizmo(Scene& scene, ipc::IPCRouter& router);

    void focus_gizmo_on(const glm::vec3& point) {
        gizmo_matrix_ = glm::translate(glm::mat4(1.0f), point);
    }

    // Draw the box-selection rectangle overlay (call after draw_gizmo).
    void draw_box_select_overlay() const;

    // Handle all mouse/keyboard input (call AFTER all ImGui windows are drawn).
    void process_input(GLFWwindow* window, Scene& scene);

    // Camera panel + gizmo mode panel, drawn inside the main ImGui window.
    void draw_ui();

    void draw_axis_labels(const Scene& scene) const;

    //change camera to
    void reframe_to(glm::vec3 target, float distance = 15.0f);

    //extract and apply session states from when reading and saving a session file
    void apply_state(const AppState& state);
    void extract_state(AppState& state) const;

    // Matrix accessors (used by scene render call) 

    glm::mat4 view_matrix()             const;
    glm::mat4 projection_matrix()       const;
    glm::mat4 vp_matrix()               const;
    glm::vec3 camera_position()         const;
    Frustum   frustum()                 const;
    float     aspect()                  const;

    int width()  const { return width_; }
    int height() const { return height_; }

    // Gizmo state (read by draw_gizmo_ui)

    ImGuizmo::OPERATION gizmo_op()   const { return gizmo_op_; }
    ImGuizmo::MODE      gizmo_mode() const { return gizmo_mode_; }

private:
    // Input sub-handlers

    void handle_camera_input(GLFWwindow* window, ImGuiIO& io);
    void handle_selection   (Scene& scene,        ImGuiIO& io);
    void handle_keyboard    (Scene& scene,        ImGuiIO& io);

    // Camera switch

    void switch_camera(CameraMode mode);

    // State

    int   width_  = 1920;
    int   height_ = 1080;

    // DPI scaling applied to mouse coordinates
    float scaling_x_ = 1.0f;
    float scaling_y_ = 1.0f;

    std::unique_ptr<ICamera> camera_;
    CameraMode               camera_mode_ = CameraMode::Orbit;

    // Gizmo
    ImGuizmo::OPERATION gizmo_op_   = ImGuizmo::TRANSLATE;
    ImGuizmo::MODE      gizmo_mode_ = ImGuizmo::WORLD;
    glm::mat4           gizmo_matrix_{1.0f};

    // Box selection
    bool      box_selecting_ = false;
    glm::vec2 box_start_{0};
    glm::vec2 box_end_{0};

    // Mouse tracking (for delta computation)
    glm::vec2 last_mouse_{0};
};