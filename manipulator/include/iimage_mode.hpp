#pragma once
 
#include "ipc_messages.hpp"
#include "shader.hpp"
 
#include <glad/gl.h>
#include <glm/glm.hpp>
 
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
 
/**
 * @brief Interface and implementations for ImageViewer display modes.
 *
 * IImageMode is a pure interface that ImageViewer uses to display images.
 * Each mode is responsible for:
 *   - Storing its own per-sample overlay texture (if any).
 *   - Compiling its own shaders (if any).
 *   - Implementing resolve() which returns a GL texture ready for ImGui::Image().
 *
 * ImageViewer owns the raw RGBA8 images (entries_raw_) and passes the
 * current sample's raw texture into resolve().  This keeps the raw images
 * stored exactly once regardless of how many modes exist.
 *
 * The shared vertex shader (fullscreen-triangle, no VBO) lives in
 * CompositeImageMode and is compiled once per concrete subclass instance.
 */
 
// Per-sample texture entry (used by all concrete modes)
struct ImageEntry {
    uint32_t    id      = 0;
    uint32_t    width   = 0;
    uint32_t    height  = 0;
    std::string name;
    GLuint      texture = 0;
};
 
// ## Mode hierarchy:
//
//   IImageMode                   
//     RawImageMode              
//     CompositeImageMode         
//       PredictionImageMode      
//       HeatmapImageMode         
class IImageMode {
public:
    virtual ~IImageMode() = default;
 
    // Called once after GL context exists.
    virtual void init() = 0;
 
    // Receive image data from an IPC batch.
    // Each mode only uses batches whose batch_type it cares about.
    virtual void upload(const ipc::ImageBatchMsg& batch) = 0;
 
    // Return the GL texture to display for sample idx at the given pixel size.
    //   raw_tex (the RGBA8 raw anchor image for this sample (owned by ImageViewer))
    //   idx (sample index)
    //   w, h (display cell size in pixels (used to size the FBO))
    // Returns 0 if the mode has no data for this sample.
    virtual GLuint resolve(GLuint raw_tex, int idx, int w, int h) = 0;
 
    // Number of samples this mode has data for. Returns 0 if empty.
    virtual int count() const = 0;
 
    // Free all per-sample textures and GL resources.
    virtual void clear() = 0;
 
    // Short label shown in the ImageViewer toolbar radio button.
    virtual const char* label() const = 0;
 
    // Whether this mode has a meaningful alpha blend slider.
    virtual bool needs_alpha() const { return false; }
 
    // Blend weight accessor (when needs_alpha() is true).
    virtual float  get_alpha() const  { return 0.5f; }
    virtual void   set_alpha(float a) { (void)a; }
};
