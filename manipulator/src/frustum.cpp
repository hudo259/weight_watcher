#include "frustum.hpp"

void Frustum::update(const glm::mat4& vp) {
    // Extract planes from view-projection matrix
    // Plane equation: ax + by + cz + d = 0
    // Point is in front if ax + by + cz + d > 0
    
    planes_[0] = normalize_plane(row(vp, 3) + row(vp, 0)); // Left
    planes_[1] = normalize_plane(row(vp, 3) - row(vp, 0)); // Right
    planes_[2] = normalize_plane(row(vp, 3) + row(vp, 1)); // Bottom
    planes_[3] = normalize_plane(row(vp, 3) - row(vp, 1)); // Top
    planes_[4] = normalize_plane(row(vp, 3) + row(vp, 2)); // Near
    planes_[5] = normalize_plane(row(vp, 3) - row(vp, 2)); // Far
}

bool Frustum::contains_point(const glm::vec3& point) const {
    for (const auto& plane : planes_) {
        if (signed_distance(plane, point) < 0) {
            return false;
        }
    }
    return true;
}

bool Frustum::intersects_sphere(const glm::vec3& center, float radius) const {
    for (const auto& plane : planes_) {
        if (signed_distance(plane, center) < -radius) {
            return false;
        }
    }
    return true;
}

bool Frustum::intersects_aabb(const glm::vec3& min, const glm::vec3& max) const {

    // AABB test: find the corner most aligned with plane normal
    // If that corner is outside, the whole box is outside
    for (const auto& plane : planes_) {
        glm::vec3 p = min;
        if (plane.x >= 0) p.x = max.x;
        if (plane.y >= 0) p.y = max.y;
        if (plane.z >= 0) p.z = max.z;
        
        if (signed_distance(plane, p) < 0) {
            return false;
        }
    }
    return true;
}
