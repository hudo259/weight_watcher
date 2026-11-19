#pragma once

#include <glm/glm.hpp>
#include <array>

// Frustum culling. Extracts 6 planes from view-projection matrix. 
// Tests points, spheres, and AABBs against frustum. Used to skip rendering chunks outside camera view.
class Frustum {
public:
    Frustum() = default;
    
    explicit Frustum(const glm::mat4& vp) {
        update(vp);
    }
    
    void update(const glm::mat4& vp);
    bool contains_point(const glm::vec3& point) const;
    bool intersects_sphere(const glm::vec3& center, float radius) const;
    bool intersects_aabb(const glm::vec3& min, const glm::vec3& max) const;

private:
    std::array<glm::vec4, 6> planes_;
    
    static glm::vec4 row(const glm::mat4& m, int i) {
        return {m[0][i], m[1][i], m[2][i], m[3][i]};
    }
    
    static glm::vec4 normalize_plane(const glm::vec4& p) {
        float len = glm::length(glm::vec3(p));
        return p / len;
    }
    
    static float signed_distance(const glm::vec4& plane, const glm::vec3& point) {
        return glm::dot(glm::vec3(plane), point) + plane.w;
    }
};