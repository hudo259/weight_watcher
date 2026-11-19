#include "shader.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>

Shader::Shader(GLenum type, const char* src) : type_(type) {
    // glCreateShaderProgramv = compile + glProgramParameteri(SEPARABLE) + link, all in one
    program_ = glCreateShaderProgramv(type, 1, &src);

    GLint status = 0;
    glGetProgramiv(program_, GL_LINK_STATUS, &status);
    if (!status) {
        char log[512];
        glGetProgramInfoLog(program_, 512, nullptr, log);
        glDeleteProgram(program_);
        program_ = 0;
        throw std::runtime_error(std::string("Shader link error: ") + log);
    }
}

Shader Shader::from_file(GLenum type, const char* path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error(std::string("Could not open shader: ") + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return Shader(type, ss.str().c_str());
}

Shader Shader::from_string(GLenum type, const char* src) {
    return Shader(type, src);
}

void Shader::set_mat4(const char* name, const glm::mat4& m) const {
    glProgramUniformMatrix4fv(program_, glGetUniformLocation(program_, name), 1, GL_FALSE, glm::value_ptr(m));
}

void Shader::set_vec3(const char* name, const glm::vec3& v) const {
    glProgramUniform3fv(program_, glGetUniformLocation(program_, name), 1, glm::value_ptr(v));
}

void Shader::set_float(const char* name, float v) const {
    glProgramUniform1f(program_, glGetUniformLocation(program_, name), v);
}

void Shader::set_int(const char* name, int v) const {
    glProgramUniform1i(program_, glGetUniformLocation(program_, name), v);
}

void Shader::set_uint(const char* name, uint32_t v) const {
    glProgramUniform1ui(program_, glGetUniformLocation(program_, name), v);
}

GLbitfield ShaderPipeline::stage_bit(GLenum type) {
    switch (type) {
        case GL_VERTEX_SHADER:          return GL_VERTEX_SHADER_BIT;
        case GL_FRAGMENT_SHADER:        return GL_FRAGMENT_SHADER_BIT;
        case GL_GEOMETRY_SHADER:        return GL_GEOMETRY_SHADER_BIT;
        case GL_TESS_CONTROL_SHADER:    return GL_TESS_CONTROL_SHADER_BIT;
        case GL_TESS_EVALUATION_SHADER: return GL_TESS_EVALUATION_SHADER_BIT;
        case GL_COMPUTE_SHADER:         return GL_COMPUTE_SHADER_BIT;
        default: throw std::runtime_error("Unknown shader stage type");
    }
}

ShaderPipeline& ShaderPipeline::attach(const Shader& shader) {
    glUseProgramStages(pipeline_, stage_bit(shader.type()), shader.id());
    return *this;
}