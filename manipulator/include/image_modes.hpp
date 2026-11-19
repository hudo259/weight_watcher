#pragma once

#include "iimage_mode.hpp"

#include <memory>
#include <vector>

// RawImageMode
// Displays the raw anchor images without any overlay.
// resolve() returns raw_tex directly — no shader, no FBO, no GL work.
class RawImageMode final : public IImageMode {
public:
    void        init()   override {}
    void        upload(const ipc::ImageBatchMsg&) override {}  // raw images stored by ImageViewer
    GLuint      resolve(GLuint raw_tex, int, int, int) override { return raw_tex; }
    int         count()  const override { return 0; }          // ImageViewer uses raw count
    void        clear()  override {}
    const char* label()  const override { return "Raw"; }
};
 
// CompositeImageMode (base for two-texture blending modes)
// Concrete subclasses implement compile_fragment_shader() and
//   set_uniforms_for_sample() to customize the compositing.
class CompositeImageMode : public IImageMode {
public:
    ~CompositeImageMode() override;
 
    void   init()  override;
    void   clear() override;
 
    bool  needs_alpha() const override { return true; }
    float get_alpha()   const override { return alpha_; }
    void  set_alpha(float a) override  { alpha_ = a; }
 
    int count() const override { return (int)entries_.size(); }
 
protected:
    // Upload a GRAYSCALE ImageBatchMsg into entries_ as GL_R8 textures.
    void upload_r8_batch(const ipc::ImageBatchMsg& batch);
    void upload_rgba_batch(const ipc::ImageBatchMsg& batch);

    // Run the compositing pass and return fbo_tex_.
    // Binds raw_tex to unit 0, entries_[idx].texture to unit 1.
    GLuint composite(GLuint raw_tex, int idx, int w, int h);
 
    // Subclasses compile their fragment shader here and attach it to pipeline_.
    virtual void compile_fragment_shader() = 0;
 
    // Subclasses set any additional uniforms beyond u_Raw, u_Overlay, u_Alpha.
    virtual void set_extra_uniforms() {}
 
    // Shared GL resources
    std::unique_ptr<Shader> vert_;
    std::unique_ptr<Shader> frag_;
    ShaderPipeline          pipeline_;
    GLuint                  null_vao_ = 0;
 
    // FBO
    GLuint fbo_     = 0;
    GLuint fbo_tex_ = 0;
    int    fbo_w_   = 0;
    int    fbo_h_   = 0;
 
    // Per-sample overlay textures (R8)
    std::vector<ImageEntry> entries_;
 
    float alpha_ = 0.5f;
    bool is_precomposited_ = false;
 
private:
    void ensure_fbo(int w, int h);
    void destroy_fbo();
 
    static GLuint upload_r8_tex(const ipc::ImageMsg& msg, GLuint existing = 0);
    static void   free_tex(GLuint& tex);
};
 
// PredictionImageMode
// Blends the raw image with a per-pixel class index map (R8).
// The fragment shader performs a palette lookup on u_ClassColors[]
// and blends the result with the raw image at u_Alpha.
// Class colors can be set via class_colors[] (public member).
class PredictionImageMode final : public CompositeImageMode {
public:
    static constexpr int MAX_CLASSES = 16;
 
    // Default palette: 0=black(bg), 1=red, 2=green, 3=blue, rest=magenta
    std::array<glm::vec3, MAX_CLASSES> class_colors;
 
    PredictionImageMode();
 
    void        upload(const ipc::ImageBatchMsg& batch) override;
    GLuint      resolve(GLuint raw_tex, int idx, int w, int h) override;
    const char* label() const override { return "Predictions"; }
 
protected:
    void compile_fragment_shader() override;
    void set_extra_uniforms()      override;
};
 
// HeatmapImageMode
// Blends the raw image with a spatial activation map (R8).
// Encoding: uint8 0 = min activation, 255 = max activation.
// The fragment shader maps through a diverging blue→white→red colormap.
class HeatmapImageMode final : public CompositeImageMode {
public:
    void        upload(const ipc::ImageBatchMsg& batch) override;
    GLuint      resolve(GLuint raw_tex, int idx, int w, int h) override;
    const char* label() const override { return "Heatmap"; }
 
protected:
    void compile_fragment_shader() override;
};
