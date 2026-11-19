#pragma once

#include <glm/glm.hpp>

#include <vector>
#include <array>
#include <memory>
#include <algorithm>

/**
 * @brief Spatial indexing for point clouds with chunk-based rendering.
 *
 * The octree serves two purposes here:
 * 1. Reorder points for spatial coherence (cache-friendly GPU access)
 * 2. Produce chunk metadata for frustum culling
 *
 * Unlike traditional octrees that store points in nodes, this implementation uses a temporary tree during construction (OctreeBuilder)
 * Then it outputs a flat list of chunks with ranges into reordered array and discards tree structure after building (only chunks remain)
 *
 * Points are NOT duplicated. Each chunk stores {offset, count} into the single reordered point array. This saves max_points_per_leaf memory compared
 * to copying points into each leaf.
 *
 * Also:
 * - Points within a chunk are contiguous -> single glDrawArrays call
 * - Each chunk has tight AABB for accurate culling
 * - Chunk size controlled by max_points_per_leaf setting
 *
 * ## Use like this:
 * std::vector<glm::vec3> original_points = ...;
 * std::vector<glm::vec3> reordered;
 * std::vector<uint32_t> new_to_original;
 *
 * OctreeData data = build_octree(original_points, reordered, new_to_original);
 *
 */

// Chunk metadata - references range in reordered point array
struct ChunkInfo {
    uint32_t offset = 0;      // Start index in reordered array
    uint32_t count = 0;       // Number of points
    glm::vec3 aabb_min{0};    // Tight bounding box
    glm::vec3 aabb_max{0};
};

// Configurable octree parameters
struct OctreeSettings {
    size_t max_points_per_leaf = 8192;
    int max_depth = 10;
    
    // Auto-tune based on point count
    static OctreeSettings for_point_count(size_t num_points) {
        if (num_points < 10000) {
            return {num_points + 1, 1};  // Single chunk
        } else if (num_points < 100000) {
            return {4096, 8};
        } else if (num_points < 1000000) {
            return {8192, 10};
        } else if (num_points < 10000000) {
            return {16384, 12};
        } else {
            return {32768, 14};
        }
    }
};

// Spatial indexing for frustum culling. Reorders points for cache-friendly GPU upload (SoA)
// Stores only chunk metadata (offset, count, AABB). No point duplication 
// Builder uses temporary AoS during construction (since all fields are accessed anyways), then outputs to SoA
class OctreeBuilder {
public:
    struct PointRef {
        glm::vec3 position;
        uint32_t original_index;
    };
    
    OctreeBuilder(const glm::vec3& min, const glm::vec3& max,
                  const OctreeSettings& settings, int depth = 0)
        : aabb_min_(min)
        , aabb_max_(max)
        , settings_(settings)
        , depth_(depth)
    {}
    
    void insert(const glm::vec3& point, uint32_t original_index);
    
    // Collect points in depth-first order -> spatially coherent chunks
    void collect_reordered(std::vector<glm::vec3>& out_points,
                            std::vector<uint32_t>& out_new_to_original,
                            std::vector<ChunkInfo>& out_chunks);

private:
    bool contains(const glm::vec3& point) const;
    void split();
    int get_child_index(const glm::vec3& point) const;
    
    glm::vec3 aabb_min_;
    glm::vec3 aabb_max_;
    OctreeSettings settings_;
    int depth_;
    bool is_leaf_ = true;
    
    std::vector<PointRef> points_;
    std::array<std::unique_ptr<OctreeBuilder>, 8> children_;
};


// Result of octree build, metadata only, no point storage
struct OctreeData {
    std::vector<ChunkInfo> chunks;
    OctreeSettings settings;
    
    struct Stats {
        size_t total_chunks = 0;
        size_t smallest_chunk = 0;
        size_t largest_chunk = 0;
    } stats;
    
    void compute_stats() {
        stats.total_chunks = chunks.size();
        if (!chunks.empty()) {
            stats.smallest_chunk = chunks[0].count;
            stats.largest_chunk = chunks[0].count;
            for (const auto& c : chunks) {
                stats.smallest_chunk = std::min(stats.smallest_chunk, static_cast<size_t>(c.count));
                stats.largest_chunk = std::max(stats.largest_chunk, static_cast<size_t>(c.count));
            }
        }
    }
};


// Build function: original points -> reordered points + chunk metadata
inline OctreeData build_octree(
    const std::vector<glm::vec3>& original_points,
    std::vector<glm::vec3>& out_reordered_points,
    std::vector<uint32_t>& out_new_to_original,
    const OctreeSettings& settings = OctreeSettings{})
{
    OctreeData result;
    result.settings = settings;
    
    out_reordered_points.clear();
    out_new_to_original.clear();
    
    if (original_points.empty()) {
        return result;
    }
    
    out_reordered_points.reserve(original_points.size());
    out_new_to_original.reserve(original_points.size());
    
    // Compute AABB with padding
    glm::vec3 min = original_points[0];
    glm::vec3 max = original_points[0];
    for (const auto& p : original_points) {
        min = glm::min(min, p);
        max = glm::max(max, p);
    }
    
    glm::vec3 padding = (max - min) * 0.001f + glm::vec3(0.001f);
    min -= padding;
    max += padding;
    
    // Build tree
    OctreeBuilder root(min, max, settings);
    for (uint32_t i = 0; i < original_points.size(); i++) {
        root.insert(original_points[i], i);
    }
    
    // Collect reordered points and chunks
    root.collect_reordered(out_reordered_points, out_new_to_original, result.chunks);
    
    result.compute_stats();
    return result;
}


// Build inverse mapping: original_idx -> new_idx
inline std::vector<uint32_t> build_original_to_new(
    const std::vector<uint32_t>& new_to_original,
    size_t original_size)
{
    std::vector<uint32_t> original_to_new(original_size);
    for (uint32_t new_idx = 0; new_idx < new_to_original.size(); new_idx++) {
        original_to_new[new_to_original[new_idx]] = new_idx;
    }
    return original_to_new;
}
