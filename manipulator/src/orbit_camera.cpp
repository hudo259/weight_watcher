#include "orbit_camera.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

OrbitCamera::OrbitCamera() : ICamera() {
    yaw = 0.0f; //azimuthal angle, aka yaw
    pitch = 0.3f; //polar angle, aka pitch
}

glm::mat4 OrbitCamera::view_matrix() const {
    return glm::lookAt(position(), target, glm::vec3(0, 1, 0));
}

glm::mat4 OrbitCamera::projection_matrix(float aspect) const {
    return glm::perspective(glm::radians(fov), aspect, near_plane, far_plane);
}

glm::vec3 OrbitCamera::position() const {
    return target + glm::vec3(
        distance * std::cos(pitch) * std::sin(yaw),
        distance * std::sin(pitch),
        distance * std::cos(pitch) * std::cos(yaw)
    );
}

glm::vec3 OrbitCamera::forward() const {
    return glm::normalize(target - position());
}

void OrbitCamera::on_mouse_move(float dx, float dy, bool left, bool middle, bool right, bool shift) {
    (void)left;
    
    if (middle || right) {
        if (shift) {
            pan(dx, dy);
        } else {
            rotate(dx, dy);
        }
    }
}

void OrbitCamera::on_scroll(float delta) {
    zoom(delta);
}

void OrbitCamera::rotate(float dx, float dy) {
    //mouse moves horizontally, e.g. right -> dx positive -> yaw (yaw) decreases -> camera orbits left (counterclockwise from above)
    yaw -= dx * rotate_speed;

    //clamp at [-pi, pi] so the slider stays in sync to the mouse movement
    yaw = std::fmod(yaw + glm::pi<float>(), glm::two_pi<float>()) - glm::pi<float>();

    //mouse moves verticall, e.g. down -> dy positive -> pitch (pitch) decreases -> camera moves down (looking down at target)
    pitch += dy * rotate_speed;

    //clamp at ~86 degrees radians to avoid gimbal lock at 90 degrees and to prevent flipping over the poles
    pitch = std::clamp(pitch, -1.5f, 1.5f);
}

void OrbitCamera::pan(float dx, float dy) {
    //Camera is on Z+ axis, looking toward target at origin

    //The right vector is -x, which is coplanar to XZ-plane. Imagine the unit circle with Z being up and X. -X is then just simply (cos(ANGLE), -sin(ANGLE)).
    //ANGLE is then yaw (aka yaw) and we rotate around y axis since we are looking at XZ-plane => right = (cos(yaw), 0, -sin(yaw)) 
    glm::vec3 right = glm::normalize(glm::vec3(std::cos(yaw), 0, -std::sin(yaw)));
    glm::vec3 up(0, 1, 0);
    
    // Scale by distance so pan feels consistent at any zoom
    float scale = pan_speed * distance * 0.01f;
    target += right * dx * scale;
    target -= up * dy * scale;
}

void OrbitCamera::zoom(float delta) {
    // Multiplicative zoom for consistent feel
    distance *= (delta > 0) ? (1.0f / zoom_speed) : zoom_speed;
    distance = std::clamp(distance, 0.1f, 500.0f);
}
void OrbitCamera::focus_on(const glm::vec3& center, float radius) {
    target = center;
    distance = radius * 1.5f;  // Pull back enough to see the whole object
}

void OrbitCamera::set_yaw(float yaw) {
    //clamp at [-pi, pi] so the slider stays in sync to the mouse movement
    yaw = std::fmod(yaw + glm::pi<float>(), glm::two_pi<float>()) - glm::pi<float>();
}

void OrbitCamera::set_pitch(float pitch) {
    //clamp at ~86 degrees radians to avoid gimbal lock at 90 degrees and to prevent flipping over the poles
    pitch = glm::clamp(pitch, -1.5f, 1.5f);
}


