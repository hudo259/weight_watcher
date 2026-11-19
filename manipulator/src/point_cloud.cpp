#include "point_cloud.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <random>
#include <algorithm>

// Static helpers

std::vector<glm::vec3> PointCloud::make_random(size_t count, float range, uint32_t seed) {
    std::vector<glm::vec3> points(count);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-range, range);
    for (auto& p : points)
        p = glm::vec3(dist(rng), dist(rng), dist(rng));
    return points;
}

// Set points

void PointCloud::set_points(std::vector<glm::vec3> original_points) {
    original_count_ = original_points.size();

    auto settings    = OctreeSettings::for_point_count(original_points.size());
    octree_data_     = build_octree(original_points, points_, new_to_original_, settings);
    original_to_new_ = build_original_to_new(new_to_original_, original_count_);

    selection_.resize(original_count_);
    selection_.clear();
    selected_original_indices_.clear();
    selection_centroid_sum_ = glm::vec3(0);

    update_bounds();

    // Both position and selection data are new; Scene must upload both
    positions_dirty_ = true;
    selection_dirty_ = true;
}

// Selection  (all indices are in original space)

void PointCloud::select(uint32_t original_idx, bool add_to_selection) {
    if (original_idx >= original_count_) return;
    if (!add_to_selection) clear_selection();

    if (!selection_[original_idx]) {
        selection_.set(original_idx, true);
        selected_original_indices_.push_back(original_idx);
        selection_centroid_sum_ += get_point_by_original(original_idx);
    }
    selection_dirty_ = true;
}

void PointCloud::toggle_selection(uint32_t original_idx) {
    if (original_idx >= original_count_) return;

    if (selection_[original_idx]) {
        selection_.set(original_idx, false);
        selected_original_indices_.erase(
            std::remove(selected_original_indices_.begin(),
                        selected_original_indices_.end(), original_idx),
            selected_original_indices_.end());
        selection_centroid_sum_ -= get_point_by_original(original_idx);
    } else {
        selection_.set(original_idx, true);
        selected_original_indices_.push_back(original_idx);
        selection_centroid_sum_ += get_point_by_original(original_idx);
    }
    selection_dirty_ = true;
}

void PointCloud::select_multiple(const std::vector<uint32_t>& original_indices,
                                  bool add_to_selection) {
    if (!add_to_selection) clear_selection();

    for (uint32_t orig_idx : original_indices) {
        if (orig_idx < original_count_ && !selection_[orig_idx]) {
            selection_.set(orig_idx, true);
            selected_original_indices_.push_back(orig_idx);
            selection_centroid_sum_ += get_point_by_original(orig_idx);
        }
    }
    selection_dirty_ = true;
}

void PointCloud::select_range(uint32_t offset, uint32_t count, bool add_to_selection) {
    if (!add_to_selection) clear_selection();

    for (uint32_t i = offset; i < offset + count && i < original_count_; ++i)
        selection_.set(i, true);

    // Rebuild the list and centroid from the bitmask (simpler than incremental update)
    selected_original_indices_.clear();
    selection_centroid_sum_ = glm::vec3(0);
    for (uint32_t i = 0; i < original_count_; ++i) {
        if (selection_[i]) {
            selected_original_indices_.push_back(i);
            selection_centroid_sum_ += get_point_by_original(i);
        }
    }
    selection_dirty_ = true;
}

void PointCloud::clear_selection() {
    selection_.clear();
    selected_original_indices_.clear();
    selection_centroid_sum_ = glm::vec3(0);
    selection_dirty_ = true;
}

void PointCloud::select_all() {
    selection_centroid_sum_ = glm::vec3(0);
    selected_original_indices_.clear();
    selected_original_indices_.reserve(original_count_);

    for (uint32_t i = 0; i < original_count_; i++) {
        selection_.set(i, true);
        selected_original_indices_.push_back(i);
    }
    for (const auto& p : points_)
        selection_centroid_sum_ += p;

    selection_dirty_ = true;
}

void PointCloud::invert_selection() {
    selection_centroid_sum_ = glm::vec3(0);
    selected_original_indices_.clear();

    for (uint32_t i = 0; i < original_count_; i++) {
        bool was = selection_[i];
        selection_.set(i, !was);
        if (!was) {
            selected_original_indices_.push_back(i);
            selection_centroid_sum_ += get_point_by_original(i);
        }
    }
    selection_dirty_ = true;
}

// Transforms

void PointCloud::apply_transform(const glm::mat4& delta, const glm::vec3& pivot) {
    if (selected_original_indices_.empty()) return;

    for (uint32_t orig_idx : selected_original_indices_) {
        uint32_t  new_idx = original_to_new_[orig_idx];
        glm::vec3 local   = points_[new_idx] - pivot;
        points_[new_idx]  = glm::vec3(delta * glm::vec4(local, 1.0f)) + pivot;
    }

    // Recompute centroid from moved points
    selection_centroid_sum_ = glm::vec3(0);
    for (uint32_t orig_idx : selected_original_indices_)
        selection_centroid_sum_ += get_point_by_original(orig_idx);

    update_bounds();
    update_chunk_aabbs();
    positions_dirty_ = true;   // Scene will upload the updated range next frame
}

void PointCloud::translate_all(const glm::vec3& offset) {
    for (auto& p : points_) p += offset;
    bounds_min_ += offset;
    bounds_max_ += offset;
    for (auto& chunk : octree_data_.chunks) {
        chunk.aabb_min += offset;
        chunk.aabb_max += offset;
    }
    positions_dirty_ = true;
}

// Index conversion

uint32_t PointCloud::new_to_original(uint32_t new_idx) const {
    if (new_idx >= new_to_original_.size()) return UINT32_MAX;
    return new_to_original_[new_idx];
}

uint32_t PointCloud::original_to_new(uint32_t orig_idx) const {
    if (orig_idx >= original_to_new_.size()) return UINT32_MAX;
    return original_to_new_[orig_idx];
}

glm::vec3 PointCloud::get_point_by_original(uint32_t orig_idx) const {
    return points_[original_to_new_[orig_idx]];
}

std::vector<glm::vec3> PointCloud::selected_positions(
    const std::vector<uint32_t>& indices) const
{
    std::vector<glm::vec3> positions;
    positions.reserve(indices.size());
    for (uint32_t orig_idx : indices)
        positions.push_back(get_point_by_original(orig_idx));
    return positions;
}

// GPU interface (called by Scene)

// Returns a BitVector in reordered (GPU) index space: bit i is set when the
// point at reordered position i is selected.  Scene writes this into the
// global selection SSBO at the word range [vbo_base_/32 ...].
BitVector PointCloud::build_reordered_selection() const {
    BitVector reordered(points_.size());
    for (uint32_t orig_idx : selected_original_indices_) {
        uint32_t new_idx = original_to_new_[orig_idx];
        reordered.set(new_idx, true);
    }
    return reordered;
}

// Private helpers

void PointCloud::update_bounds() {
    if (points_.empty()) {
        bounds_min_ = bounds_max_ = glm::vec3(0);
        return;
    }
    bounds_min_ = bounds_max_ = points_[0];
    for (const auto& p : points_) {
        bounds_min_ = glm::min(bounds_min_, p);
        bounds_max_ = glm::max(bounds_max_, p);
    }
}

void PointCloud::update_chunk_aabbs() {
    for (auto& chunk : octree_data_.chunks) {
        if (chunk.count == 0) continue;
        chunk.aabb_min = chunk.aabb_max = points_[chunk.offset];
        for (uint32_t i = 1; i < chunk.count; i++) {
            const auto& p  = points_[chunk.offset + i];
            chunk.aabb_min = glm::min(chunk.aabb_min, p);
            chunk.aabb_max = glm::max(chunk.aabb_max, p);
        }
    }
}