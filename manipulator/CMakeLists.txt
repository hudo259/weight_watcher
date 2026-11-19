cmake_minimum_required(VERSION 3.16)
project(WeightWatcher_Manipulator LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release) # Default to Release for performance
endif()

if(MSVC)
    add_compile_options(/W4 /Ox /Oi /Ot) # Maximize speed on Windows
else()
    add_compile_options(-Wall -Wextra -O3 -march=native) # Maximize speed on Linux/Mac
endif()

# ------------------------------------------------------------------------------
# Dependencies

include(FetchContent)

# GLFW
FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw.git GIT_TAG 3.4)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Wayland or X11
set(GLFW_BUILD_WAYLAND ON CACHE BOOL "" FORCE)
set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)

# GLM (Header only)
FetchContent_Declare(glm GIT_REPOSITORY https://github.com/g-truc/glm.git GIT_TAG 1.0.3)

# ImGui
FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG v1.92.8)

# ImGuizmo
FetchContent_Declare(imguizmo GIT_REPOSITORY https://github.com/CedricGuillemet/ImGuizmo.git GIT_TAG 1.10)
FetchContent_Populate(imguizmo)

# NNG (Nano Message Next-Gen)
FetchContent_Declare(nng GIT_REPOSITORY https://github.com/nanomsg/nng.git GIT_TAG v1.11)
set(NNG_TESTS OFF CACHE BOOL "" FORCE)
set(NNG_TOOLS OFF CACHE BOOL "" FORCE)
set(NNG_ENABLE_TLS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(glfw glm imgui imguizmo nng)

# ------------------------------------------------------------------------------

#NOTE: It is recommended to install GLAD via the generator: https://gen.glad.sh/
#and then to include it manually. Which is where the gl.h and gl.c files come from.

set(EXTERNAL
  external/glad/src/gl.c
)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS include/*.hpp)

set(IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

add_executable(${PROJECT_NAME} 
    ${EXTERNAL}
    ${SOURCES} 
    ${HEADERS} 
    ${IMGUI_SOURCES}
    ${imguizmo_SOURCE_DIR}/src/ImGuizmo.cpp
    ${imguizmo_SOURCE_DIR}/src/GraphEditor.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE 
    include
    external/glad/include
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${imguizmo_SOURCE_DIR}/include
    ${imguizmo_SOURCE_DIR}/src
    ${glm_SOURCE_DIR}
)

# GLM is header-only, so we just need the include path if not using their CMake
target_link_libraries(${PROJECT_NAME} PRIVATE 
    glfw
    ${CMAKE_DL_LIBS} # Required for GLAD on Linux
)

find_package(OpenGL REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenGL::GL nng)

if(APPLE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE GL_SILENCE_DEPRECATION)
endif()

# For the AppImage
include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
