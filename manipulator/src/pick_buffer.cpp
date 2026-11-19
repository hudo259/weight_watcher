#include "pick_buffer.hpp"

#include <algorithm>
#include <iostream>

void PickBuffer::create(int width, int height) {
    destroy();

    width_ = width;
    height_ = height;

    glGenFramebuffers(1, &fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);

    //Create an ID texture (RG32UI) to store the picked points inside the box
    //RG32UI texture for Group+Local ID (see PickResult class in the header file).
    glGenTextures(1, &id_tex_);
    glBindTexture(GL_TEXTURE_2D, id_tex_);

    //   glTexImage2D(                                                            
    //       GL_TEXTURE_2D,     // Target: 2D texture                             
    //       0,                 // Mipmap level (0 = base)                        
    //       GL_RG32UI,          // Internal format: 32-bit unsigned int, 2 channel (so we have 64 bit)
    //       width, height,     // Dimensions                                     
    //       0,                 // Border (must be 0)                             
    //       GL_RED_INTEGER,    // Format of input data (integer, not float)      
    //       GL_UNSIGNED_INT,   // Data type                                      
    //       nullptr            // No initial data (we'll render into it)         
    //   );  
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32UI, width, height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    //attach texture to framebuffer
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, id_tex_, 0);

    // Depth buffer
    //We need depth buffer because otherwise we would pick the point that has been drawn last instead of the point that is actually the frontmost
    glGenRenderbuffers(1, &depth_rbo_);
    glBindRenderbuffer(GL_RENDERBUFFER, depth_rbo_);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rbo_);
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void PickBuffer::resize(int width, int height) {
    if(width != width_ || height != height_) {
        create(width, height);
    }
}

void PickBuffer::bind() {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glViewport(0,0, width_, height_);
}

void PickBuffer::unbind() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void PickBuffer::clear() {
    uint32_t clear_val[2] = {INVALID, INVALID};
    glClearBufferuiv(GL_COLOR, 0, clear_val);
    glClear(GL_DEPTH_BUFFER_BIT);
}

PickResult PickBuffer::read_pixel(int x, int y) const {

    //GLFW, ImGUI and mouse input in general have window coordinates/screen coordinates with the origin on the top left (Y increases downward).
    //OpenGL framebuffer coordinates (textures, glReadPixels, etc.) has origin from bottom left (Y increases upward).
    //Therefore we flip the y axis to openGL
    y = height_ - y - 1; //-1 because valid pixel position/range is from 0 to height-1

    //if outside
    if (x < 0 || x >= width_ || y < 0 || y >= height_) {
        return {};
    }

    uint32_t data[2];
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glReadPixels(x, y, 1, 1, GL_RG_INTEGER, GL_UNSIGNED_INT, data);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    
    if (data[0] == INVALID && data[1] == INVALID) {
        return {};
    }
    
    return {data[0], data[1]};
}

std::vector<PickResult> PickBuffer::read_rect(int x0, int y0, int x1, int y1) const {
    //User can draw a rectangle beginning from each direction...
    if(x0 > x1) std::swap(x0, x1);
    if(y0 > y1) std::swap(y0, y1);

    //GLFW, ImGUI and mouse input in general have window coordinates/screen coordinates with the origin on the top left (Y increases downward).
    //OpenGL framebuffer coordinates (textures, glReadPixels, etc.) has origin from bottom left (Y increases upward).
    //Therefore we flip to openGL
    //-1 because valid pixel position/range is from 0 to height-1
    y0 = height_ - y0 - 1;
    y1 = height_ - y1 - 1;
    if(y0 > y1) std::swap(y0, y1);

    // Clamp to buffer bounds
    x0 = std::clamp(x0, 0, width_ - 1);
    x1 = std::clamp(x1, 0, width_ - 1);
    y0 = std::clamp(y0, 0, height_ - 1);
    y1 = std::clamp(y1, 0, height_ - 1);

    int w = x1 - x0 + 1;
    int h = y1 - y0 + 1;

    //Read RG pairs from buffer
    std::vector<uint32_t> pixels(w * h * 2);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glReadPixels(x0, y0, w, h, GL_RG_INTEGER, GL_UNSIGNED_INT, pixels.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    std::vector<PickResult> results;
    results.reserve(w * h / 4);

    for (int i = 0; i < w * h; i++) {
        uint32_t group = pixels[i * 2];
        uint32_t local = pixels[i * 2 + 1];
        
        if (group != INVALID || local != INVALID) {
            results.push_back({group, local});
        }
    }
    
    return results;
}

int PickBuffer::width() const {
    return width_;
}

int PickBuffer::height() const {
    return height_;
}

void PickBuffer::destroy() {
    if(fbo_) glDeleteFramebuffers(1, &fbo_);
    if(id_tex_) glDeleteTextures(1, &id_tex_);
    if(depth_rbo_) glDeleteRenderbuffers(1, &depth_rbo_);
    fbo_ = id_tex_ = depth_rbo_ = 0;
}