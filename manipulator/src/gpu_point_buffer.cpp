#include "gpu_point_buffer.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

void GpuPointBuffer::init(uint32_t initial_capacity) {
    destroy();
    capacity_ = initial_capacity;
    used_     = 0;
    
    
    glGenVertexArrays(1, &vao_);

    // Position VBO (attribute 0)
    glGenBuffers(1, &vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(capacity_ * sizeof(glm::vec3)), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Heatmap color VBO (attribute 1)
    // Parallel layout to vbo_: slot i's heatmap color lives at the same base
    // and count as slot i's position.  Initialised to zero (black). the
    // shader only reads it when u_HeatmapActive == 1, so the initial contents
    // are irrelevant for correctness but zero is a safe default.
    glGenBuffers(1, &heatmap_vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, heatmap_vbo_);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(capacity_ * sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Wire both VBOs into the VAO
    rewire_vao();

    // Selection SSBO
    // Used as a compact bit-array: one bit per VBO slot indicates whether that point is selected.  
    // The vertex shader reads it to decide whether to use the selection highlight color.
    selection_mask_cpu_.assign((capacity_ + 31) / 32, 0u);
    glGenBuffers(1, &selection_ssbo_);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, selection_ssbo_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(selection_mask_cpu_.size() * sizeof(uint32_t)), selection_mask_cpu_.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void GpuPointBuffer::destroy() {
    if (vao_)            glDeleteVertexArrays(1, &vao_);
    if (vbo_)            glDeleteBuffers(1, &vbo_);
    if (heatmap_vbo_)    glDeleteBuffers(1, &heatmap_vbo_);
    if (selection_ssbo_) glDeleteBuffers(1, &selection_ssbo_);
    vao_ = vbo_ = heatmap_vbo_ = selection_ssbo_ = 0;
    capacity_ = used_ = 0;
    free_list_.clear();
    selection_mask_cpu_.clear();
}

void GpuPointBuffer::rewire_vao() {
    // Attribute 0: vec3 world-space position (from vbo_)
    // Attribute 1: float raw heatmap score    (from heatmap_vbo_)

    glBindVertexArray(vao_);

    // Attribute 0: position
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
    glEnableVertexAttribArray(0);

    // Attribute 1: heatmap score
    // Bound separately so the VAO records the correct source buffer.
    glBindBuffer(GL_ARRAY_BUFFER, heatmap_vbo_);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), nullptr);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

GpuPointBuffer::Slot GpuPointBuffer::allocate(uint32_t count) {
    // Best-fit: find the smallest free range that fits
    auto best = free_list_.end();
    for (auto it = free_list_.begin(); it != free_list_.end(); ++it) {
        if (it->count >= count) {
            if (best == free_list_.end() || it->count < best->count)
                best = it;
        }
    }

    if (best != free_list_.end()) {
        Slot s{best->base, count};
        if (best->count == count) free_list_.erase(best);
        else { best->base += count; best->count -= count; }
        // Zero selection bits for the reused range
        for (uint32_t i = 0; i < count; i++) {
            uint32_t bit = s.base + i;
            selection_mask_cpu_[bit / 32] &= ~(1u << (bit % 32));
        }
        return s;
    }

    uint32_t base = used_;
    used_ += count;
    grow(used_);
    return {base, count};
}

void GpuPointBuffer::free(const Slot& slot) {
    for (uint32_t i = 0; i < slot.count; i++) {
        uint32_t bit = slot.base + i;
        selection_mask_cpu_[bit / 32] &= ~(1u << (bit % 32));
    }
    free_list_.push_back({slot.base, slot.count});
}

void GpuPointBuffer::defrag(std::vector<Slot>& slots,
                              const std::vector<bool>& alive) {
    struct Entry { size_t idx; uint32_t old_base, count; };
    std::vector<Entry> live;
    live.reserve(slots.size());
    for (size_t i = 0; i < slots.size(); i++) {
        if (alive[i] && slots[i].count > 0)
            live.push_back({i, slots[i].base, slots[i].count});
    }
    std::sort(live.begin(), live.end(),
              [](const Entry& a, const Entry& b){ return a.old_base < b.old_base; });

    uint32_t total = 0;
    for (const auto& e : live) {
        total += e.count;
    }
    if (total == 0) {
        used_ = 0;
        free_list_.clear();
        clear_selection();
        return;
    }

    // Gather live blocks into a scratch buffer, then copy back to [0, total).
    // GL rejects overlapping copies within one buffer, and an in-place left shift overlaps
    // whenever a block moves by less than its length, e.g. [10, 110) -> [0, 100).
    // TODO: we could also do this without the scratch buffer by splitting each move into pieces no longer than
    // the shift distance. But a block that moves by 1 slot would need one glCopyBufferSubData call per point...
    auto compact = [&](GLuint buf, size_t elem) {
        GLuint scratch = 0;
        glGenBuffers(1, &scratch);
        glBindBuffer(GL_COPY_WRITE_BUFFER, scratch);
        glBufferData(GL_COPY_WRITE_BUFFER, (GLsizeiptr)(total * elem), nullptr, GL_STREAM_COPY);
        glBindBuffer(GL_COPY_READ_BUFFER, buf);

        uint32_t head = 0;
        for (const auto& e: live) {
            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                    (GLintptr)(e.old_base * elem),
                    (GLintptr)(head * elem),
                    (GLsizeiptr)(e.count * elem));
            head += e.count;
        }

        glBindBuffer(GL_COPY_READ_BUFFER, scratch);
        glBindBuffer(GL_COPY_WRITE_BUFFER, buf);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, (GLsizeiptr)(total * elem));
        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
        glDeleteBuffers(1, &scratch);
    };

    compact(vbo_, sizeof(glm::vec3)); // positions: 3 floats per point
    compact(heatmap_vbo_, sizeof(float)); //heatmap: 1 raw score per point

    //Rebase slots in the same order the data was packed...
    uint32_t write_head = 0;
    for (const auto& e : live) {
        slots[e.idx].base = write_head;
        write_head += e.count;
    }

    used_ = write_head;
    free_list_.clear();
    clear_selection();
}

void GpuPointBuffer::upload_positions(const Slot& slot, const glm::vec3* data) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferSubData(GL_ARRAY_BUFFER,
        (GLintptr) (slot.base  * sizeof(glm::vec3)),
        (GLsizeiptr)(slot.count * sizeof(glm::vec3)),
        data);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GpuPointBuffer::upload_heatmap_scores(const Slot& slot,
                                             const float* scores) {
    // glBufferSubData updates a sub-range of an existing buffer without
    // reallocating it.  Write exactly slot.count vec3s starting at
    // slot.base, leaving adjacent slots untouched.
    glBindBuffer(GL_ARRAY_BUFFER, heatmap_vbo_);
    glBufferSubData(GL_ARRAY_BUFFER,
        (GLintptr) (slot.base  * sizeof(float)),
        (GLsizeiptr)(slot.count * sizeof(float)),
        scores);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GpuPointBuffer::clear_heatmap_scores() {
    // Zero the entire heatmap VBO.  glBufferData with nullptr reallocates at
    // the same size, it's the fastest way to zero a buffer since the driver
    // can discard the old storage and hand us a fresh zeroed one.
    glBindBuffer(GL_ARRAY_BUFFER, heatmap_vbo_);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(capacity_ * sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Rewire VAO so attribute 1 still points to the (new) heatmap VBO storage.
    // glBufferData does not invalidate the VAO's pointer record, but the old
    // storage may have been orphaned by the driver...
    rewire_vao();
}

void GpuPointBuffer::sync_selection(const Slot& slot,
                                     const BitVector& reordered_sel) {
    if (slot.count == 0) return;
    for (uint32_t i = 0; i < slot.count; i++) {
        uint32_t global = slot.base + i;
        if (reordered_sel[i])
            selection_mask_cpu_[global / 32] |=  (1u << (global % 32));
        else
            selection_mask_cpu_[global / 32] &= ~(1u << (global % 32));
    }
    upload_selection_words(slot.base / 32, (slot.base + slot.count - 1) / 32);
}

void GpuPointBuffer::clear_selection() {
    std::fill(selection_mask_cpu_.begin(), selection_mask_cpu_.end(), 0u);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, selection_ssbo_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(selection_mask_cpu_.size() * sizeof(uint32_t)), selection_mask_cpu_.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

float GpuPointBuffer::fragmentation() const {
    if (used_ == 0) return 0.0f;
    uint32_t wasted = 0;
    for (const auto& r : free_list_) wasted += r.count;
    return (float)wasted / (float)used_;
}

void GpuPointBuffer::grow(uint32_t required) {
    if (required <= capacity_) return;

    uint32_t new_cap = std::max(required, capacity_ * 2);
    new_cap = std::max(new_cap, INITIAL_CAPACITY);

    // Grow position VBO
    GLuint new_vbo;
    glGenBuffers(1, &new_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, new_vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(new_cap * sizeof(glm::vec3)), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Copy the live region GPU-side using capacity_ (the old VBO's actual
    // allocated size), NOT used_.  allocate() increments used_ before calling
    // grow(), so used_ > capacity_ here (using used_ would read past the end
    // of the source buffer (GL_INVALID_VALUE, silent no-op copy)).
    if (capacity_ > 0) {
        glBindBuffer(GL_COPY_READ_BUFFER,  vbo_);
        glBindBuffer(GL_COPY_WRITE_BUFFER, new_vbo);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, (GLsizeiptr)(capacity_ * sizeof(glm::vec3)));
        glBindBuffer(GL_COPY_READ_BUFFER,  0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    }
    glDeleteBuffers(1, &vbo_);
    vbo_ = new_vbo;

    // Grow heatmap VBO
    // Same pattern as the position VBO.  Any existing u_HeatmapActive colors in
    // [0, capacity_) are preserved, new slots start with 0.
    GLuint new_heat;
    glGenBuffers(1, &new_heat);
    glBindBuffer(GL_ARRAY_BUFFER, new_heat);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(new_cap * sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    if (capacity_ > 0) {
        glBindBuffer(GL_COPY_READ_BUFFER,  heatmap_vbo_);
        glBindBuffer(GL_COPY_WRITE_BUFFER, new_heat);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, (GLsizeiptr)(capacity_ * sizeof(float)));
        glBindBuffer(GL_COPY_READ_BUFFER,  0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    }
    glDeleteBuffers(1, &heatmap_vbo_);
    heatmap_vbo_ = new_heat;

    capacity_ = new_cap;

    // Rewire VAO to both new buffer handles
    rewire_vao();

    // Grow selection SSBO
    uint32_t new_word_count = (new_cap + 31) / 32;
    selection_mask_cpu_.resize(new_word_count, 0u);

    GLuint new_ssbo;
    glGenBuffers(1, &new_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, new_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)(selection_mask_cpu_.size() * sizeof(uint32_t)), selection_mask_cpu_.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glDeleteBuffers(1, &selection_ssbo_);
    selection_ssbo_ = new_ssbo;
}

void GpuPointBuffer::upload_selection_words(uint32_t first_word,
                                              uint32_t last_word) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, selection_ssbo_);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER,
        (GLintptr) (first_word * sizeof(uint32_t)),
        (GLsizeiptr)((last_word - first_word + 1) * sizeof(uint32_t)),
        selection_mask_cpu_.data() + first_word);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}
