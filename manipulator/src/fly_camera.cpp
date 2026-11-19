#include "fly_camera.hpp"

FlyCamera::FlyCamera() : ICamera() {
    yaw = -glm::half_pi<float>();  // Looking toward -Z
    pitch = 0.0f;
}

glm::mat4 FlyCamera::view_matrix() const {
    return glm::lookAt(pos, pos + forward(), glm::vec3(0, 1, 0));
}

glm::mat4 FlyCamera::projection_matrix(float aspect) const {
    return glm::perspective(glm::radians(fov), aspect, near_plane, far_plane);
}

glm::vec3 FlyCamera::position() const {
    return pos;
}

glm::vec3 FlyCamera::forward() const {
    //Spherical coordinates to cartesian. Radius (distance) is 1 in this case (wich we omit) since the forward vector is
    //the unit vector pointing where camera looks
    return glm::normalize(glm::vec3(
        std::cos(pitch) * std::cos(yaw),
        std::sin(pitch),
        std::cos(pitch) * std::sin(yaw)
    ));
}

glm::vec3 FlyCamera::right() const {
    return glm::normalize(glm::cross(forward(), glm::vec3(0, 1, 0)));
}

glm::vec3 FlyCamera::up() const{
    return glm::normalize(glm::cross(right(), forward()));
}

void FlyCamera::on_mouse_move(float dx, float dy, bool left, bool middle, bool right, bool shift) {
    (void)left; (void)middle; (void)shift;
    
    // Look only when right mouse held
    if (right) {
        yaw += dx * look_speed;

        //clamp at [-pi, pi] so the slider stays in sync to the mouse movement
        yaw = std::fmod(yaw + glm::pi<float>(), glm::two_pi<float>()) - glm::pi<float>();

        pitch -= dy * look_speed;

        //clamp at ~86 degrees radians to avoid gimbal lock at 90 degrees and to prevent flipping over the poles
        pitch = std::clamp(pitch, -1.5f, 1.5f);
    }
}

void FlyCamera::on_scroll(float delta) {
    // Adjust move speed with scroll
    move_speed *= (delta > 0) ? 1.2f : (1.0f / 1.2f);
    move_speed = std::clamp(move_speed, 0.1f, 100.0f);
}

void FlyCamera::update(float dt) {
    glm::vec3 move_dir{0.0f};
    
    if (move_forward)  move_dir += forward();
    if (move_backward) move_dir -= forward();
    if (move_right)    move_dir += right();
    if (move_left)     move_dir -= right();
    if (move_up)       move_dir += glm::vec3(0, 1, 0);
    if (move_down)     move_dir -= glm::vec3(0, 1, 0);
    
    if (glm::length(move_dir) > 0.001f) {
        move_dir = glm::normalize(move_dir);
        float speed = move_speed * (sprinting ? sprint_multiplier : 1.0f);
        pos += move_dir * speed * dt;
    }
}

void FlyCamera::focus_on(const glm::vec3& center, float radius) {
    // Position camera to look at center from a reasonable distance
    float distance = radius * 2.5f;
    
    // Place camera along current view direction, or default if uninitialized
    glm::vec3 dir = glm::vec3(
        cos(pitch) * sin(yaw),
        sin(pitch),
        cos(pitch) * cos(yaw)
    );
    
    pos = center - dir * distance;
    
    // Adjust speed based on scene size
    move_speed = radius * 0.5f;
    
    // Adjust near/far
    //near_plane = std::max(0.01f, radius * 0.01f);
    //far_plane = std::max(100.0f, radius * 20.0f);
}

void FlyCamera::set_yaw(float yaw) {
    yaw = std::fmod(yaw + glm::pi<float>(), glm::two_pi<float>()) - glm::pi<float>();
}

void FlyCamera::set_pitch(float pitch) {
    pitch = std::clamp(pitch, -1.5f, 1.5f);
}