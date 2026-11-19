#pragma once

#include <glad/gl.h>

#include <cstdint>
#include <vector>
#include <algorithm>

struct PickResult {
    uint32_t group_id = 0xFFFFFFFF;
    uint32_t local_id = 0xFFFFFFFF;
    
    bool valid() const { 
        return group_id != 0xFFFFFFFF || local_id != 0xFFFFFFFF; 
    }
    
    bool operator==(const PickResult& other) const {
        return group_id == other.group_id && local_id == other.local_id;
    }
    
    bool operator<(const PickResult& other) const {
        if (group_id != other.group_id) return group_id < other.group_id;
        return local_id < other.local_id;
    }
};

// GPU-based picking. Renders object/point IDs to RG32UI texture. R = group ID (cloud + chunk), G = local point index. 
// Fast fixed-function depth test determines frontmost point. Reads back on click.
class PickBuffer {
public:
    static constexpr uint32_t INVALID = 0xFFFFFFFF;
    
    PickBuffer() = default;
    
    ~PickBuffer() { 
        destroy(); 
    }
    
    PickBuffer(const PickBuffer&) = delete;
    PickBuffer& operator=(const PickBuffer&) = delete;
    
    void create(int width, int height);
    void resize(int width, int height);
    void bind();
    void unbind();
    void clear();
    PickResult read_pixel(int x, int y) const;
    std::vector<PickResult> read_rect(int x0, int y0, int x1, int y1) const;
    int width() const;
    int height() const;

private:
    void destroy();
    
    GLuint fbo_ = 0;
    GLuint id_tex_ = 0;
    GLuint depth_rbo_ = 0;
    int width_ = 0;
    int height_ = 0;
};