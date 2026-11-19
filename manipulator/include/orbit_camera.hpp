#pragma once

#include "icamera.hpp"

class OrbitCamera : public ICamera {
public:
    glm::vec3 target{0.0f};

    float distance = 50.0f; //radial distance
    
    float rotate_speed = 0.005f;
    float pan_speed = 0.01f;
    float zoom_speed = 1.1f;
   
    OrbitCamera();
    glm::mat4 view_matrix() const override;
    glm::mat4 projection_matrix(float aspect) const override;
    glm::vec3 position() const override;
    glm::vec3 forward() const override;
    void set_yaw(float yaw) override;
    void set_pitch(float pitch) override;
    
    void on_mouse_move(float dx, float dy, bool left, bool middle, bool right, bool shift) override;
    void on_scroll(float delta) override;
    void focus_on(const glm::vec3& center, float radius) override;

private:
    void rotate(float dx, float dy);
    void pan(float dx, float dy);
    void zoom(float delta);
};
