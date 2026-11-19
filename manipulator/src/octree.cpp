#include "octree.hpp"

void OctreeBuilder::insert(const glm::vec3& point, uint32_t original_index) {
        if (!contains(point)) return;
        
        if (is_leaf_) {
            points_.push_back({point, original_index});
            
            if (points_.size() > settings_.max_points_per_leaf &&
                depth_ < settings_.max_depth) {
                split();
            }
        } else {
            int child_idx = get_child_index(point);
            children_[child_idx]->insert(point, original_index);
        }
    }
    
void OctreeBuilder::collect_reordered(std::vector<glm::vec3>& out_points,
                        std::vector<uint32_t>& out_new_to_original,
                        std::vector<ChunkInfo>& out_chunks) {
    if (is_leaf_) {
        if (points_.empty()) return;
        
        ChunkInfo chunk;
        chunk.offset = static_cast<uint32_t>(out_points.size());
        chunk.count = static_cast<uint32_t>(points_.size());
        
        chunk.aabb_min = points_[0].position;
        chunk.aabb_max = points_[0].position;
        
        for (const auto& p : points_) {
            out_points.push_back(p.position);
            out_new_to_original.push_back(p.original_index);
            chunk.aabb_min = glm::min(chunk.aabb_min, p.position);
            chunk.aabb_max = glm::max(chunk.aabb_max, p.position);
        }
        
        out_chunks.push_back(chunk);
        
        // Free memory as we go
        points_.clear();
        points_.shrink_to_fit();
    } else {
        for (auto& child : children_) {
            if (child) {
                child->collect_reordered(out_points, out_new_to_original, out_chunks);
                child.reset();
            }
        }
    }
}

bool OctreeBuilder::contains(const glm::vec3& point) const {
    return point.x >= aabb_min_.x && point.x <= aabb_max_.x &&
            point.y >= aabb_min_.y && point.y <= aabb_max_.y &&
            point.z >= aabb_min_.z && point.z <= aabb_max_.z;
}

void OctreeBuilder::split() {
    is_leaf_ = false;
    glm::vec3 center = (aabb_min_ + aabb_max_) * 0.5f;
    
    // Create 8 children (octants)
    for (int i = 0; i < 8; i++) {
        glm::vec3 child_min, child_max;
        
        child_min.x = (i & 1) ? center.x : aabb_min_.x;
        child_max.x = (i & 1) ? aabb_max_.x : center.x;
        child_min.y = (i & 2) ? center.y : aabb_min_.y;
        child_max.y = (i & 2) ? aabb_max_.y : center.y;
        child_min.z = (i & 4) ? center.z : aabb_min_.z;
        child_max.z = (i & 4) ? aabb_max_.z : center.z;
        
        children_[i] = std::make_unique<OctreeBuilder>(
            child_min, child_max, settings_, depth_ + 1);
    }
    
    // Redistribute points to children
    for (auto& p : points_) {
        int child_idx = get_child_index(p.position);
        children_[child_idx]->insert(p.position, p.original_index);
    }
    
    points_.clear();
    points_.shrink_to_fit();
}

int OctreeBuilder::get_child_index(const glm::vec3& point) const {
    glm::vec3 center = (aabb_min_ + aabb_max_) * 0.5f;
    int index = 0;
    if (point.x > center.x) index |= 1;
    if (point.y > center.y) index |= 2;
    if (point.z > center.z) index |= 4;
    return index;
}