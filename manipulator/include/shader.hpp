#pragma once

#include <glad/gl.h>
#include <glm/glm.hpp>
#include <string>

// A single separable shader stage (vertex, fragment, etc.)
// Compiled and linked independently via glCreateShaderProgramv.
// use Shader::from_file, Shader::from_string to create an instance.
class Shader {
public:
    // Factory functions
    static Shader from_file(GLenum type, const char* path);
    static Shader from_string(GLenum type, const char* src);

    ~Shader() { if (program_) glDeleteProgram(program_); }

    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;

    Shader(Shader&& o) noexcept : program_(o.program_), type_(o.type_) { 
        o.program_ = 0; 
    }

    Shader& operator=(Shader&& o) noexcept {
        if (this != &o) {
            if (program_) {
                glDeleteProgram(program_);
            }
            program_ = o.program_; type_ = o.type_;
            o.program_ = 0;
        }
        return *this;
    }

    GLuint id()      const { return program_; }
    GLenum type()    const { return type_; }
    bool   valid()   const { return program_ != 0; }

    // Uniforms set directly on this stage (no glUseProgram needed)
    void set_mat4 (const char* name, const glm::mat4& m) const;
    void set_vec3 (const char* name, const glm::vec3& v) const;
    void set_float(const char* name, float v)             const;
    void set_int  (const char* name, int v)               const;
    void set_uint (const char* name, uint32_t v)          const;

private:
    // Force using the factories
    Shader() = default;
    Shader(GLenum type, const char* src);

    GLuint program_ = 0;
    GLenum type_    = 0;
};


// Assembles separable shader stages into a pipeline object.
// Usage:
//   ShaderPipeline p;
//   p.attach(vert_shader).attach(frag_shader).link();
//   p.bind();
class ShaderPipeline {
public:
    ShaderPipeline()  { glCreateProgramPipelines(1, &pipeline_); }
    ~ShaderPipeline() { if (pipeline_) glDeleteProgramPipelines(1, &pipeline_); }

    ShaderPipeline(const ShaderPipeline&) = delete;
    ShaderPipeline& operator=(const ShaderPipeline&) = delete;

    ShaderPipeline(ShaderPipeline&& o) noexcept : pipeline_(o.pipeline_) { o.pipeline_ = 0; }
    ShaderPipeline& operator=(ShaderPipeline&& o) noexcept {
        if (this != &o) {
            if (pipeline_) glDeleteProgramPipelines(1, &pipeline_);
            pipeline_ = o.pipeline_;
            o.pipeline_ = 0;
        }
        return *this;
    }

    // Attach a stage (maps GL shader type -> pipeline stage bit automatically)
    ShaderPipeline& attach(const Shader& shader);

    void bind()   const { glBindProgramPipeline(pipeline_); }
    void unbind() const { glBindProgramPipeline(0); }

    GLuint id() const { return pipeline_; }

private:
    GLuint pipeline_ = 0;

    static GLbitfield stage_bit(GLenum type);
};