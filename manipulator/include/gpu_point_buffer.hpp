#pragma once

#include "bit_vector.hpp"

#include <glm/glm.hpp>
#include <glad/gl.h>

#include <vector>
#include <cstdint>

/**
 * @brief Shared VBO + heatmap VBO + VAO + selection SSBO for all point clouds.
 *
 * # Some notes so I don't forget this some months later:
 *
 * ## Buffer layout
 *
 * Three parallel GPU buffers, all sharing the same flat index space:
 *
 *   vbo_: vec3 positions,   attribute 0,  always uploaded
 *   heatmap_vbo_: vec3 RGB colors,  attribute 1,  uploaded only when
 *                   concept scores arrive; ignored by the shader when
 *                   u_HeatmapActive == 0
 *   selection_ssbo_: one bit per slot, binding 0
 *
 * ## Allocation
 * Best-fit free list + append.  Holes left by free() are reused or left
 * until defrag() is called.
 *
 * ## Defrag
 * Compacts live slots toward the front using GPU-side glCopyBufferSubData
 * on both vbo_ and heatmap_vbo_.  No CPU round-trip.
 */
class GpuPointBuffer {
public:
    static constexpr uint32_t INITIAL_CAPACITY = 1u << 17; // 128 K points

    struct Slot {
        uint32_t base  = 0;
        uint32_t count = 0;
    };

    GpuPointBuffer()  = default;
    ~GpuPointBuffer() { destroy(); }

    GpuPointBuffer(const GpuPointBuffer&)            = delete;
    GpuPointBuffer& operator=(const GpuPointBuffer&) = delete;

    void init(uint32_t initial_capacity = INITIAL_CAPACITY);
    void destroy();

    // Allocation & Defrag
    Slot allocate(uint32_t count);
    void free(const Slot& slot);
    void defrag(std::vector<Slot>& slots, const std::vector<bool>& alive);

    void upload_positions(const Slot& slot, const glm::vec3* data);

    // Heatmap color upload

    // Upload per-point heatmap scores for one slot.
    // scores must be a float per point.
    // Scene calls this after mapping concept scores through a diverging colormap.
    void upload_heatmap_scores(const Slot& slot, const float* scores);

    // Zero the entire heatmap VBO.
    // Scene calls this when the user turns the heatmap off, so no stale colors
    // are visible if the heatmap is re-enabled without new scores arriving.
    void clear_heatmap_scores();


    void sync_selection(const Slot& slot, const BitVector& reordered_sel);
    void clear_selection();


    GLuint   vao()      const { return vao_; }
    GLuint   sel_ssbo() const { return selection_ssbo_; }
    uint32_t capacity() const { return capacity_; }
    uint32_t used()     const { return used_; }

    float fragmentation() const;

private:
    struct FreeRange { uint32_t base, count; };

    GLuint   vao_            = 0;
    GLuint   vbo_            = 0;    // attribute 0: vec3 position
    GLuint   heatmap_vbo_    = 0;    // attribute 1: vec3 heatmap color
    GLuint   selection_ssbo_ = 0;
    uint32_t capacity_       = 0;
    uint32_t used_           = 0;

    std::vector<FreeRange> free_list_;
    std::vector<uint32_t>  selection_mask_cpu_;

    void grow(uint32_t required);

    // Wire attribute 0 -> vbo_ and attribute 1 -> heatmap_vbo_ into the VAO.
    // Must be called after either VBO handle changes (grow, init).
    void rewire_vao();

    void upload_selection_words(uint32_t first_word, uint32_t last_word);
};
