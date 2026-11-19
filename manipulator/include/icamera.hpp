#pragma once

#include "frustum.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>

// Abstract camera interface
class ICamera {
public:
    virtual ~ICamera() = default;
    
    virtual glm::mat4 view_matrix() const = 0;
    virtual glm::mat4 projection_matrix(float aspect) const = 0;
    virtual glm::vec3 position() const = 0;
    virtual glm::vec3 forward() const = 0;
    
    glm::mat4 vp_matrix(float aspect) const {
        return projection_matrix(aspect) * view_matrix();
    }
    
    Frustum frustum(float aspect) const {
        return Frustum(vp_matrix(aspect));
    }
    
    // Input handlers - override as needed
    virtual void on_mouse_move(float dx, float dy, bool left, bool middle, bool right, bool shift) {
        (void)dx; (void)dy; (void)left; (void)middle; (void)right; (void)shift;
    }
    
    virtual void on_scroll(float delta) {
        (void)delta;
    }
    
    virtual void update(float dt) {
        (void)dt;
    }

    virtual void focus_on(const glm::vec3& center, float radius) {
        (void)center; (void)radius;
    }
   
    virtual void set_yaw(float yaw) {
        (void)yaw;
    }

    virtual void set_pitch(float pitch) {
        (void)pitch;
    }

    // Shared projection parameters
    float yaw = 0.0f;
    float pitch = 0.0f;
    float fov = 45.0f;
    float near_plane = 0.1f;
    float far_plane = 1000.0f;
};