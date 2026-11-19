#include "image_modes.hpp"

#include <algorithm>
#include <iostream>
#include <cstring>

// Shared vertex shader source
// # Notes so that I don't forget later:
//
// Used by every CompositeImageMode subclass.  Generates a fullscreen triangle
// from gl_VertexID alone, no VBO, no vertex data.
//
//   ID=0 -> (-1,-1)   ID=1 -> (3,-1)   ID=2 -> (-1,3)
//
// The triangle extends beyond the viewport.  The rasterizer clips it at the
// [-1,1]x[-1,1] boundary so every viewport pixel is covered exactly once.
//

static const char* VERT_SRC = R"(
    #version 430 core
    out vec2 vUV;
    // Required by ARB_separate_shader_objects when using program pipelines:
    // gl_PerVertex must be redeclared before any of its members are written.
    out gl_PerVertex { vec4 gl_Position; };
    void main() {
        vec2 pos = vec2(
            (gl_VertexID == 1) ?  3.0 : -1.0,
            (gl_VertexID == 2) ?  3.0 : -1.0
        );
        // UV convention: OpenGL textures have (0,0) at bottom-left. Image data is
        // stored top-row-first.  We flip V once here so fragment shaders sample
        // correctly without needing their own flip.
        //vUV         = vec2(pos.x * 0.5 + 0.5, 1.0 - (pos.y * 0.5 + 0.5));
        
        // UV convention: raw textures are uploaded top-row-first from the backend and the FBO result
        // is displayed by ImGui::Image(), which also treats v=0 as the top row. So we pass through unflipped
        vUV = vec2(pos.x * 0.5 + 0.5, pos.y * 0.5 + 0.5);
        gl_Position = vec4(pos, 0.0, 1.0);
    }
)";

// CompositeImageMode, lifetime
CompositeImageMode::~CompositeImageMode() {
    clear();
    destroy_fbo();
    if (null_vao_) { glDeleteVertexArrays(1, &null_vao_); null_vao_ = 0; }
}

void CompositeImageMode::init() {
    // Compile the shared vertex shader
    vert_ = std::make_unique<Shader>(Shader::from_string(GL_VERTEX_SHADER, VERT_SRC));

    // Compile the subclass-specific fragment shader and attach both to the pipeline
    compile_fragment_shader();
    pipeline_.attach(*vert_).attach(*frag_);

    // The GL spec requires a non-zero VAO to be bound during glDrawArrays,
    // even when the vertex shader reads no per-vertex data at all.
    glGenVertexArrays(1, &null_vao_);
}

void CompositeImageMode::clear() {
    for (auto& e : entries_)
        if (e.texture) { glDeleteTextures(1, &e.texture); e.texture = 0; }
    entries_.clear();
}

// CompositeImageMode, texture upload
GLuint CompositeImageMode::upload_r8_tex(const ipc::ImageMsg& msg, GLuint existing) {
    // GL_R8: single-channel 8-bit normalised texture.
    // Stores one uint8 per pixel, the shader samples it as float in [0,1].
    // Used for both class index maps and activation maps.
    GLuint tex = existing;
    if (!tex) glGenTextures(1, &tex);

    glBindTexture(GL_TEXTURE_2D, tex);
    // GL_NEAREST: never interpolate between class ids or raw activation values.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    // GL_RED as client format: one byte per pixel, stored in the R channel.
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8,
                 (GLsizei)msg.width, (GLsizei)msg.height,
                 0, GL_RED, GL_UNSIGNED_BYTE, msg.data.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

void CompositeImageMode::free_tex(GLuint& tex) {
    if (tex) { glDeleteTextures(1, &tex); tex = 0; }
}

void CompositeImageMode::upload_r8_batch(const ipc::ImageBatchMsg& batch) {
    // Resize entries_ to match batch, reusing existing handles to avoid
    // unnecessary glDeleteTextures / glGenTextures churn.
    while (entries_.size() < batch.images.size()) entries_.push_back({});
    while (entries_.size() > batch.images.size()) {
        free_tex(entries_.back().texture);
        entries_.pop_back();
    }
    for (size_t i = 0; i < batch.images.size(); i++) {
        const auto& msg = batch.images[i];
        auto&       e   = entries_[i];
        e.id      = msg.id;
        e.width   = msg.width;
        e.height  = msg.height;
        e.name    = msg.name;
        e.texture = upload_r8_tex(msg, e.texture);
    }
}

// CompositeImageMode: FBO management
//
// # Notes so that I don't forget later:
//
// Attach a RGBA8 texture as the color output to the FBO. The compositing shader writes into it later. Then pass this texture to ImGui::Image()
// An FBO (Framebuffer Object) is an offscreen render target.  We attach a
//
// The FBO is resized lazily: if the display cell size changes between frames
// (window resize, column count change) it is recreated.

void CompositeImageMode::ensure_fbo(int w, int h) {
    if (w == fbo_w_ && h == fbo_h_ && fbo_) return;
    destroy_fbo();

    glGenTextures(1, &fbo_tex_);
    glBindTexture(GL_TEXTURE_2D, fbo_tex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenFramebuffers(1, &fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    // Attach the texture as color output 0 — fragment shader writes here.
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, fbo_tex_, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "[" << label() << "] FBO not complete\n";
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    fbo_w_ = w;
    fbo_h_ = h;
}

void CompositeImageMode::destroy_fbo() {
    if (fbo_)     { glDeleteFramebuffers(1, &fbo_);     fbo_     = 0; }
    if (fbo_tex_) { glDeleteTextures    (1, &fbo_tex_); fbo_tex_ = 0; }
    fbo_w_ = fbo_h_ = 0;
}

// CompositeImageMode, compositing pass
GLuint CompositeImageMode::composite(GLuint raw_tex, int idx, int w, int h) {
    ensure_fbo(w, h);

    // Save GL state that ImGui depends on.  The compositing pass changes the
    // framebuffer and viewport, ImGui expects to find them unchanged when it
    // submits its own draw calls after our frame returns.
    GLint prev_fbo;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prev_fbo);
    GLint prev_vp[4];
    glGetIntegerv(GL_VIEWPORT, prev_vp);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glViewport(0, 0, w, h);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    pipeline_.bind();

    // Bind raw image to texture unit 0.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, raw_tex);
    frag_->set_int("u_Raw", 0);

    // Bind overlay to unit 1
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, entries_[idx].texture);
    frag_->set_int("u_Overlay", 1);

    frag_->set_float("u_Alpha", alpha_);
    frag_->set_uint("u_IsPrecompsited", uint32_t(is_precomposited_));

    // Let the subclass set any additional uniforms (e.g. class color palette)
    set_extra_uniforms();

    // Draw the fullscreen triangle. Three vertices with positions computed
    // entirely from gl_VertexID in the vertex shader, no vertex data needed.
    glBindVertexArray(null_vao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    pipeline_.unbind();

    // Restore state
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)prev_fbo);
    glViewport(prev_vp[0], prev_vp[1], prev_vp[2], prev_vp[3]);

    return fbo_tex_;
}

void CompositeImageMode::upload_rgba_batch(const ipc::ImageBatchMsg& batch) {
    while (entries_.size() < batch.images.size()) entries_.push_back({});

    while (entries_.size() > batch.images.size()) {
        free_tex(entries_.back().texture);
        entries_.pop_back();
    }

    for (size_t i = 0; i < batch.images.size(); i++) {
        const auto& msg = batch.images[i];
        auto& e   = entries_[i];
        e.id = msg.id; e.width = msg.width;
        e.height = msg.height; e.name = msg.name;

        // Reuse the existing to_rgba conversion from image_viewer.cpp
        // by uploading as GL_RGBA8 directly
        if (!e.texture) glGenTextures(1, &e.texture);

        glBindTexture(GL_TEXTURE_2D, e.texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // Convert to RGBA on CPU (same logic as ImageViewer::to_rgba)
        const size_t N = (size_t)msg.width * msg.height;
        std::vector<uint8_t> rgba(N * 4);
        const uint8_t* s = msg.data.data();
        for (size_t p = 0; p < N; p++) {
            rgba[p*4+0]=s[p*3+0]; rgba[p*4+1]=s[p*3+1];
            rgba[p*4+2]=s[p*3+2]; rgba[p*4+3]=255;
        }

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                     (GLsizei)msg.width, (GLsizei)msg.height,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

// PredictionImageMode
PredictionImageMode::PredictionImageMode() {
    // Default class color palette
    class_colors[0] = {0.0f, 0.0f, 0.0f};   // background — black
    class_colors[1] = {1.0f, 0.0f, 0.0f};   // class 1 — red
    class_colors[2] = {0.0f, 1.0f, 0.0f};   // class 2 — green
    class_colors[3] = {0.0f, 0.0f, 1.0f};   // class 3 — blue
    for (int i = 4; i < MAX_CLASSES; i++)
        class_colors[i] = {1.0f, 0.0f, 1.0f};  // magenta — unset class
}

void PredictionImageMode::upload(const ipc::ImageBatchMsg& batch) {
    if (batch.batch_type != ipc::ImageBatchType::PREDICTIONS) return;

    // Classification path: pre-composited RGB images, store as RGBA, no class-index interpretation
    if (!batch.images.empty() &&
        batch.images[0].format != ipc::ImageFormat::GRAYSCALE) {
        upload_rgba_batch(batch);
        is_precomposited_ = true;
    } else {
        upload_r8_batch(batch);
        is_precomposited_ = false;
    }
    std::cout << "[PredictionImageMode] " << entries_.size()
              << (is_precomposited_ ? " pre-composited RGB" : " class map")
              << " images uploaded\n";
}

GLuint PredictionImageMode::resolve(GLuint raw_tex, int idx, int w, int h) {
    if (idx >= (int)entries_.size() || !entries_[idx].texture) return raw_tex;

    return composite(raw_tex, idx, w, h);
}

//TODO: into a separate file?
void PredictionImageMode::compile_fragment_shader() {
    // Inputs:
    //   u_Raw ((unit 0) RGBA8 raw anchor image)
    //   u_Overlay ((unit 1): R8 class index map
    //     Each pixel stores a class id as uint8 (0..N-1).
    //     GL_R8 normalises on sampling: uint8 k -> float k/255.
    //     We recover the integer: class = round(r * 255).)
    //   u_ClassColors[MAX_CLASSES] (RGB palette, one color per class)
    //   u_Alpha (blend weight: 0 = raw only, 1 = class color only)
    //   u_IsPrecomposited (if the images to display have already been composited before)
    const char* src = R"(
        #version 430 core
        in  vec2 vUV;
        out vec4 FragColor;

        uniform sampler2D u_Raw;
        uniform sampler2D u_Overlay;
        uniform vec3      u_ClassColors[16];
        uniform float     u_Alpha;
        uniform uint      u_IsPrecomposited;

        void main() {
            vec3  raw       = texture(u_Raw,     vUV).rgb;

            if (u_IsPrecomposited > 0) {
                float encoded   = texture(u_Overlay, vUV).r;

                // Recover integer class id.
                // round() prevents floating-point errors at class boundaries
                // (e.g. 1/255.0 * 255.0 might compute to 0.9999...).
                uint  class_idx = uint(round(encoded * 255.0));
                class_idx       = clamp(class_idx, 0u, 15u);

                vec3  cls_color = u_ClassColors[class_idx];
                FragColor       = vec4(mix(raw, cls_color, u_Alpha), 1.0);
            }
            else {
                FragColor = vec4(mix(raw, texture(u_Overlay, vUV).rgb, u_Alpha), 1.0);
            }
        }
    )";
    frag_ = std::make_unique<Shader>(Shader::from_string(GL_FRAGMENT_SHADER, src));
}

void PredictionImageMode::set_extra_uniforms() {
    // Upload the full class color palette every compositing call.
    // 16 vec3 uniforms is negligible — this never shows up in profiling.
    for (int i = 0; i < MAX_CLASSES; i++) {
        char name[32];
        snprintf(name, sizeof(name), "u_ClassColors[%d]", i);
        frag_->set_vec3(name, class_colors[i]);
    }
}

// HeatmapImageMode
void HeatmapImageMode::upload(const ipc::ImageBatchMsg& batch) {
    if (batch.batch_type != ipc::ImageBatchType::ACTIVATION_MAP) return;
    upload_r8_batch(batch);
    // Override to linear — activation values are continuous, interpolation is valid
    for (auto& e : entries_) {
        if (!e.texture) continue;
        glBindTexture(GL_TEXTURE_2D, e.texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

GLuint HeatmapImageMode::resolve(GLuint raw_tex, int idx, int w, int h) {
    if (idx >= (int)entries_.size() || !entries_[idx].texture) return raw_tex;
    return composite(raw_tex, idx, w, h);
}

//TODO: into a separate file?
void HeatmapImageMode::compile_fragment_shader() {
    // Inputs:
    //   u_Raw     ((unit 0), RGBA8 raw anchor image)
    //   u_Overlay ((unit 1), R8 spatial activation map
    //     Encoding: uint8 0 = min activation, 255 = max activation.
    //     GL_R8 maps this to float [0,1] on sampling.)
    //   u_Alpha  (blend weight)
    //
    // The shader maps the activation through a diverging colormap:
    //   heat = r * 2.0 - 1.0  maps [0,1] → [-1,1]
    //   diverging(-1) = blue  (no activation)
    //   diverging( 0) = white (moderate)
    //   diverging(+1) = red   (strong activation)
    //
    // This gives an intuitive cold-to-hot visualization: regions the filter
    // ignores appear blue/cool, regions it responds to strongly appear red/hot.
    const char* src = R"(
        #version 430 core
        in  vec2 vUV;
        out vec4 FragColor;

        uniform sampler2D u_Raw;
        uniform sampler2D u_Overlay;
        uniform float     u_Alpha;

        // Diverging colormap: t in [-1,1]
        //   t = -1 -> blue  (cold, most surpressed activation)
        //   t =  0 -> white (neutral)
        //   t = +1 -> red   (hot, strong activation)
        vec3 diverging(float t) {
            if (t >= 0.0)
                return mix(vec3(1.0), vec3(0.90, 0.18, 0.18),  t);
            else
                return mix(vec3(1.0), vec3(0.18, 0.45, 0.95), -t);
        }

        void main() {
            vec3  raw  = texture(u_Raw,     vUV).rgb;
            float r    = texture(u_Overlay, vUV).r;   // [0,1] from GL_R8
            float heat = r * 2.0 - 1.0;               // remap to [-1,1]
            FragColor  = vec4(mix(raw, diverging(heat), u_Alpha), 1.0);
        }
    )";
    frag_ = std::make_unique<Shader>(Shader::from_string(GL_FRAGMENT_SHADER, src));
}
