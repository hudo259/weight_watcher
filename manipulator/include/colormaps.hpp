#pragma once
#include <glm/glm.hpp>

// This is just so that the color map stuff is in one file, so additional maps need only be added here.
// To add a new color, add the color to ColorMapType, add the interpolation function as Shader code in GLSL_SOURCE and 
// Add a C++ host-side equivalent (this is needed to draw the color "legend" for the user in the GUI)
// And add a new case to the apply function
namespace ColorMaps {

enum class ColorMapType {
    None = 0,
    Viridis = 1,
    Inferno,
    Coolwarm,
};

//This is to map the ColorMapType into strings for the GUI
static constexpr const char* NAMES[] = {
        "None", "Viridis", "Inferno", "Coolwarm"
};

// GLSL source (injected into the vertex shader)

inline constexpr const char* GLSL_SOURCE = R"(
vec3 cm_viridis(float t) {
    const vec3 c0 = vec3(0.2777, 0.0054, 0.3342);
    return c0 + t * (vec3(-0.0296,  1.3052,  1.4014)
         + t * (vec3(-0.1003, -3.2036, -2.8431)
         + t * (vec3( 0.1568,  3.3143,  2.7275)
         + t * (vec3(-0.0522, -1.1614, -1.2317)))));
}
vec3 cm_inferno(float t) {
    const vec3 c0 = vec3(0.0002, 0.0016, 0.0139);
    return c0 + t * (vec3(0.1004,  0.0350,  0.7424)
         + t * (vec3(2.2786,  0.5728, -2.4397)
         + t * (vec3(-2.1566,  2.1526,  2.2419)
         + t * (vec3( 0.8683, -3.1998, -1.5660)))));
}
vec3 cm_coolwarm(float t) {
    vec3 cold = vec3(0.230, 0.299, 0.754);
    vec3 mid  = vec3(0.865, 0.865, 0.865);
    vec3 warm = vec3(0.706, 0.016, 0.150);

    float s = step(0.5, t);
    // mix(cold->mid, mid->warm, s)
    return mix(mix(cold, mid, t * 2.0), mix(mid, warm, (t - 0.5) * 2.0), s);
}
vec3 apply_colormap(float score, int cmap) {
    // normalize [-1,1] to [0,1]
    float u = clamp(score * 0.5 + 0.5, 0.0, 1.0);
    if (cmap == 1) return cm_viridis(u);
    if (cmap == 2) return cm_inferno(u);
    if (cmap == 3) return cm_coolwarm(u);

    return vec3(1.0); // None / fallback
}
)";

// C++ equivalents (used only by the UI gradient bar preview)
// METHODS REQUIRE TO HAVE t normalized to [0, 1]
inline glm::vec3 viridis(float t) {
    const glm::vec3 c0(0.2777f, 0.0054f, 0.3342f);
    return c0 + t * (glm::vec3(-0.0296f,  1.3052f,  1.4014f)
         + t * (glm::vec3(-0.1003f, -3.2036f, -2.8431f)
         + t * (glm::vec3( 0.1568f,  3.3143f,  2.7275f)
         + t * (glm::vec3(-0.0522f, -1.1614f, -1.2317f)))));
}

inline glm::vec3 inferno(float t) {
    const glm::vec3 c0(0.0002f, 0.0016f, 0.0139f);
    return c0 + t * (glm::vec3(0.1004f,  0.0350f,  0.7424f)
         + t * (glm::vec3(2.2786f,  0.5728f, -2.4397f)
         + t * (glm::vec3(-2.1566f,  2.1526f,  2.2419f)
         + t * (glm::vec3( 0.8683f, -3.1998f, -1.5660f)))));
}

inline glm::vec3 coolwarm(float t) {
    glm::vec3 cold(0.230f, 0.299f, 0.754f);
    glm::vec3 mid (0.865f, 0.865f, 0.865f);
    glm::vec3 warm(0.706f, 0.016f, 0.150f);
    return t < 0.5f ? glm::mix(cold, mid,  t * 2.0f)
                    : glm::mix(mid,  warm, (t - 0.5f) * 2.0f);
}

inline glm::vec3 apply(float score, int cmap) {
    float u = glm::clamp(score * 0.5f + 0.5f, 0.0f, 1.0f);
    switch (cmap) {
        case 1:  return viridis(u);
        case 2:  return inferno(u);
        case 3:  return coolwarm(u);
        default: return glm::vec3(1.0f);
    }
}

}