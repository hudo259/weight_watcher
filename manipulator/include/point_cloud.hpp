#pragma once

#include "bit_vector.hpp"
#include "octree.hpp"

#include <glm/glm.hpp>

#include <vector>
#include <cstdint>
#include <string>

/**
 * @brief Point cloud CPU storage, selection, and spatial indexing.
 *
 * Notes so I don't forget later:
 * PointCloud owns NO GPU resources.  All OpenGL buffers (shared VBO, global
 * selection SSBO) are managed by Scene.  PointCloud signals CPU-side changes
 * through two dirty flags:
 *   - positions_dirty_: set whenever points_ are mutated (transform / translate)
 *   - selection_dirty_: set whenever the selection bitmask changes
 * Scene polls these flags in render() and uploads the affected sub-ranges of
 * the shared buffers.
 *
 * Points live in two orderings:
 *   1. Original order: stable user-facing indices; selection is stored here.
 *   2. Reordered order: spatially coherent (octree leaf order) for the GPU.
 *
 * Mappings:
 *   new_to_original_[reordered_idx] -> original_idx   (picking)
 *   original_to_new_[original_idx]  -> reordered_idx  (selection SSBO)
 */
class PointCloud {
public:
    // Layer identity: set by Scene after construction
    std::string name;
    glm::vec3   color{0.4f, 0.6f, 0.9f};
    bool        visible = true;

    PointCloud()  = default;
    ~PointCloud() = default;

    PointCloud(const PointCloud&)            = delete;
    PointCloud& operator=(const PointCloud&) = delete;

    // Generate random test data
    static std::vector<glm::vec3> make_random(size_t   count,
                                               float    range = 10.0f,
                                               uint32_t seed  = 42);

    // Set points and build octree on the CPU
    void set_points(std::vector<glm::vec3> original_points);

    // Selection (all indices are in original space) 
    void select(uint32_t original_idx, bool add_to_selection);
    void toggle_selection(uint32_t original_idx);
    void select_multiple(const std::vector<uint32_t>& original_indices,
                         bool add_to_selection);
    void select_range(uint32_t offset, uint32_t count, bool add_to_selection);
    void clear_selection();
    void select_all();
    void invert_selection();

    // Transforms (mark positions_dirty_) 
    void apply_transform(const glm::mat4& delta, const glm::vec3& pivot);
    void translate_all(const glm::vec3& offset);

    // Index conversion 
    uint32_t  new_to_original(uint32_t new_idx)        const;
    uint32_t  original_to_new(uint32_t orig_idx)       const;
    glm::vec3 get_point_by_original(uint32_t orig_idx) const;

    // Accessors 
    size_t    size()            const { return original_count_; }
    size_t    selection_count() const { return selected_original_indices_.size(); }
    bool      has_selection()   const { return !selected_original_indices_.empty(); }

    glm::vec3 selection_centroid() const {
        if (selected_original_indices_.empty()) return glm::vec3(0);
        return selection_centroid_sum_ /
               static_cast<float>(selected_original_indices_.size());
    }

    const glm::vec3& bounds_min() const { return bounds_min_; }
    const glm::vec3& bounds_max() const { return bounds_max_; }
    glm::vec3        center()     const { return (bounds_min_ + bounds_max_) * 0.5f; }

    const std::vector<glm::vec3>& points()           const { return points_; }
    const std::vector<uint32_t>&  selected_indices() const { return selected_original_indices_; }

    std::vector<glm::vec3> selected_positions(
        const std::vector<uint32_t>& indices) const;

    const std::vector<ChunkInfo>& chunks()       const { return octree_data_.chunks; }
    size_t                        chunk_count()  const { return octree_data_.chunks.size(); }
    const OctreeData::Stats&      octree_stats() const { return octree_data_.stats; }

    // GPU interface (used by Scene) 

    // Build a BitVector of selection bits in reordered (GPU) index space.
    // Scene splices this into the global selection SSBO at bits
    // [vbo_base_, vbo_base_ + size()).
    BitVector build_reordered_selection() const;

    // VBO slot: base index assigned by Scene; stored here for reference
    uint32_t vbo_base()               const { return vbo_base_; }
    void     set_vbo_base(uint32_t b)       { vbo_base_ = b; }

    // Dirty flags: set on CPU mutations, cleared by Scene after GPU upload
    bool is_positions_dirty() const { return positions_dirty_; }
    void clear_positions_dirty()    { positions_dirty_ = false; }

    bool is_selection_dirty()  const { return selection_dirty_; }
    void clear_selection_dirty()     { selection_dirty_ = false; }

private:
    void update_bounds();
    void update_chunk_aabbs();

    // Positions in spatially-reordered order (matches the shared VBO layout)
    std::vector<glm::vec3> points_;
    size_t                 original_count_ = 0;

    // Index-space mappings
    std::vector<uint32_t> new_to_original_;
    std::vector<uint32_t> original_to_new_;

    // Selection in original-index space
    BitVector             selection_;
    std::vector<uint32_t> selected_original_indices_;
    glm::vec3             selection_centroid_sum_{0};

    glm::vec3 bounds_min_{0};
    glm::vec3 bounds_max_{0};

    OctreeData octree_data_;

    // GPU placement within the shared VBO, assigned and managed by Scene
    uint32_t vbo_base_ = 0;

    // Dirty flags: set by CPU mutations, cleared by Scene after upload
    bool positions_dirty_ = false;
    bool selection_dirty_ = false;
};
