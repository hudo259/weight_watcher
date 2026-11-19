#pragma once

#include "icamera.hpp"
#include <glm/gtc/type_ptr.hpp>

class FlyCamera : public ICamera {
public:
    glm::vec3 pos{0.0f, 0.0f, 50.0f};
    
    float move_speed = 10.0f;
    float look_speed = 0.002f;
    float sprint_multiplier = 3.0f;
    
    FlyCamera();
    glm::mat4 view_matrix() const override;
    glm::mat4 projection_matrix(float aspect) const override;
    glm::vec3 position() const override;
    glm::vec3 forward() const override;
    glm::vec3 right() const;
    glm::vec3 up() const;
    void set_yaw(float yaw) override;
    void set_pitch(float pitch) override;
    
    void on_mouse_move(float dx, float dy, bool left, bool middle, bool right, bool shift) override;
    void on_scroll(float delta) override;
    void update(float dt) override;
    void focus_on(const glm::vec3& center, float radius) override;

    bool move_forward = false;
    bool move_backward = false;
    bool move_left = false;
    bool move_right = false;
    bool move_up = false;
    bool move_down = false;
    bool sprinting = false;
};