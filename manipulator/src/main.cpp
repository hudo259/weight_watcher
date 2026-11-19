#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "viewport.hpp"
#include "icamera.hpp"
#include "orbit_camera.hpp"
#include "fly_camera.hpp"
#include "scene.hpp"
#include "ipc_router.hpp"
#include "ipc_nng.hpp"
#include "image_viewer.hpp"
#include "metrics_viewer.hpp"
#include "feature_viewer.hpp"
#include "app_state.hpp"

#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>

/**
Data Flow:                                                              
                                                                        
1. PointCloud receives points                                           
    Octree builds, reorders points (SoA), creates chunk metadata     
        Single VBO upload (positions only)                           
        Selection SSBO (bitmask in reordered space)                  
                                                                        
2. Scene manages multiple PointClouds                                   
   Compiles shaders (main + pick)                                   
   Owns PickBuffer for GPU-based selection                          
                                                                        
3. Render loop:                                                         
   Camera provides view/proj matrices + frustum                     
   Scene renders each cloud, culling by chunk AABB                  
   Shader reads selection SSBO, applies Blinn-Phong lighting model
                                                                       
4. Selection:                                                           
   Click/box -> render_pick_pass (IDs to texture)                    
   glReadPixels -> decode group_id -> cloud + chunk                   
   Convert: local -> reordered -> original index                      
   Update selection (original space) -> sync to GPU (reordered)      
                                                                        
5. Transform:                                                           
   ImGuizmo provides delta matrix                                   
   PointCloud transforms selected points in reordered array         
   Partial VBO update, recalculate chunk AABBs                      
   Send transformed points to backend via IPC (if auto-send enabled)
                                                                        
6. IPC with Backend:                                                    
   Renderer connects to Python backend                              
   Receives: point clouds, loss, metrics, feature vectors           
   Sends: transformed points, point clouds                          
**/

// Helper function to convert platform ID to string
const char* getPlatformName(int platform) {
    switch (platform) {
        case GLFW_PLATFORM_WIN32:   return "Win32";
        case GLFW_PLATFORM_COCOA:   return "Cocoa";
        case GLFW_PLATFORM_WAYLAND: return "Wayland";
        case GLFW_PLATFORM_X11:     return "X11";
        case GLFW_PLATFORM_NULL:    return "Null";
        default:                    return "Unknown";
    }
}

/// Bundles all application components. Defined here because it's only used in main().
struct App {
    // THE ORDER IS IMPORTANT BECAUSE IT IS USED FOR AUTOMATIC DESTRUCTION LATER!
    std::unique_ptr<ipc::IPCRouter> router;
    std::unique_ptr<Viewport> viewport;
    std::unique_ptr<Scene> scene;
    std::unique_ptr<ImageViewer> image_viewer;
    std::unique_ptr<MetricsViewer> metrics_viewer;
    std::unique_ptr<FeatureViewer> feature_viewer;

    AppState session_state;
    std::string session_file;
    bool gui_visible = true;
};

void draw_main_ui(App& app) {
    ImGui::Begin("Point Cloud Manipulator");
 
    app.scene->draw_ui_contents();           // scene panels (rendering, lighting, etc.)
    ImGui::Separator();
    app.viewport->draw_ui();                  // camera + gizmo mode panels
    ImGui::Separator();

    if (app.scene->draw_selection_ui_contents()) {
        app.viewport->focus_gizmo_on(app.scene->selection_centroid());
    }

    // Save session state
    if (ImGui::CollapsingHeader("Session")) {
        static char save_path[256] = "session_settings.txt";
        ImGui::InputText("File##session", save_path, sizeof(save_path));

        if (ImGui::Button("Save Session")) {
            app.viewport->extract_state(app.session_state);
            app.scene->extract_state(app.session_state);
            if (app.session_state.save(std::string(save_path)))
                ImGui::OpenPopup("##save_ok");
            else
                ImGui::OpenPopup("##save_fail");
        }
        ImGui::SameLine();
        if (ImGui::Button("Load Session")) {
            if (app.session_state.load(std::string(save_path))) {
                app.scene->apply_state(app.session_state);
                app.viewport->apply_state(app.session_state);
                ImGui::OpenPopup("##load_ok");
            } else {
                ImGui::OpenPopup("##load_fail");
            }
        }

        if (ImGui::BeginPopup("##save_ok")) {
            ImGui::Text("Session saved to '%s'", save_path);
            ImGui::EndPopup();
        }
        if (ImGui::BeginPopup("##save_fail")) {
            ImGui::TextColored(ImVec4(1,0.3f,0.3f,1),
                            "Failed to save! Check the path is valid.");
            ImGui::EndPopup();
        }
        if (ImGui::BeginPopup("##load_ok")) {
            ImGui::Text("Session loaded from '%s'", save_path);
            ImGui::EndPopup();
        }
        if (ImGui::BeginPopup("##load_fail")) {
            ImGui::TextColored(ImVec4(1,0.3f,0.3f,1),
                            "Failed to load! File not found or invalid.");
            ImGui::EndPopup();
        }
    }
    ImGui::End();
}

int main(int argc, char** argv) {
    std::cout << "WAYLAND_DISPLAY="
              << (getenv("WAYLAND_DISPLAY") ? getenv("WAYLAND_DISPLAY") : "unset") << "\n";
    std::cout << "DISPLAY="
              << (getenv("DISPLAY") ? getenv("DISPLAY") : "unset") << "\n";
 
    if (!glfwInit()) { std::cerr << "GLFW init failed\n"; return -1; }
 
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
 
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Weight Watcher", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }
 
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
 
    if (!gladLoadGL(glfwGetProcAddress)) { std::cerr << "GLAD init failed\n"; return -1; }
 
    std::cout << "GLFW Session " << glfwGetPlatform() << "\n";
    std::cout << "OpenGL " << glGetString(GL_VERSION) << "\n";
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << "\n";
 
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 460");
    ImGui::StyleColorsDark();
 
    // Determine the settings file from arguments
    std::string settings_file = "session_settings.txt";
    if (argc > 1) {
        settings_file = argv[1];
        std::cout << "Using session settings: " << settings_file << "\n";
    }

    {
        App app;
 
        // Init
        int fb_w, fb_h;
        glfwGetFramebufferSize(window, &fb_w, &fb_h);
        
        app.router = std::make_unique<ipc::IPCRouter>();
        app.viewport = std::make_unique<Viewport>();
        app.scene          = std::make_unique<Scene>();
        app.image_viewer   = std::make_unique<ImageViewer>();
        app.metrics_viewer = std::make_unique<MetricsViewer>();
        app.feature_viewer = std::make_unique<FeatureViewer>();

        // init() calls happen after construction — these are the GL-touching calls
        app.viewport->init(window);    // computes DPI scaling, creates camera
        app.scene->init(fb_w, fb_h);              // creates pick buffer, compiles shaders, allocates VBOs
        app.image_viewer->init();                 // compiles composite shaders, sets up modes
        app.feature_viewer->init();               // compiles point cloud shader
        app.router->init();                        // creates NNG socket, wires callbacks

        //load session state from file and apply
        app.session_state = AppState{};
        app.session_file = (argc > 1) ? argv[1] : "session_file.txt";
        app.session_state.load(app.session_file); // Fails silently to defaults if file doesn't exist
        app.scene->apply_state(app.session_state);
        app.viewport->apply_state(app.session_state);

        // Resize window to saved resolution
        if (app.session_state.fb_width > 0 && app.session_state.fb_height > 0)
            glfwSetWindowSize(window, app.session_state.fb_width, app.session_state.fb_height);

        // Register components to IPC routing
        app.scene->attach(*app.router);
        app.metrics_viewer->attach(*app.router);
        app.feature_viewer->attach(*app.router);
        app.image_viewer->attach(*app.router);
 
        app.scene->add_cloud(PointCloud::make_random(500000, 10.0f, 42), "Test Cloud");
 
        // GL state
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_MULTISAMPLE);
 
        // ---- Main loop ----
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
 
            if (app.router->is_connected())
                app.router->poll();
 
            // Resize
            int w, h;
            glfwGetFramebufferSize(window, &w, &h);
            if (w != app.viewport->width() || h != app.viewport->height()) {
                app.viewport->resize(w, h);
                app.scene->resize(w, h);
            }

            if (ImGui::IsKeyPressed(ImGuiKey_H) && !ImGui::GetIO().WantCaptureKeyboard)
                app.gui_visible = !app.gui_visible;

            // Begin frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            ImGuizmo::BeginFrame();
 
            // Update camera
            app.viewport->update(ImGui::GetIO().DeltaTime);
 
            // Render scene
            glViewport(0, 0, app.viewport->width(), app.viewport->height());
            //glClearColor(0.08f, 0.08f, 0.1f, 1.0f);
            // Use the dynamic background color
            glClearColor(
                app.scene->render_settings.bg_color.r,
                app.scene->render_settings.bg_color.g,
                app.scene->render_settings.bg_color.b,
                1.0f
            );
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
 
            auto fr = app.viewport->frustum();
            app.scene->render(
                app.viewport->view_matrix(),
                app.viewport->projection_matrix(),
                app.viewport->camera_position(),
                app.scene->render_settings.use_frustum_culling ? &fr : nullptr
            );
 
            // Gizmo overlay (before ImGui windows)
            app.viewport->draw_gizmo(*app.scene, *app.router);
            // Gizmo for the layout axes
            app.viewport->draw_axis_labels(*app.scene);

            // Reframe request check
            if (app.scene->has_pending_reframe())
                app.viewport->reframe_to(app.scene->consume_reframe_target());
 
            // Box selection rect (background draw list)
            app.viewport->draw_box_select_overlay();
 
            // ImGui windows
            if (app.gui_visible) {
                draw_main_ui(app);
                app.router->draw_ui(*app.scene);
                app.metrics_viewer->draw();
                app.image_viewer->draw();
                app.feature_viewer->draw();
            }
 
            // Input: AFTER all ImGui windows so WantCaptureMouse is correct
            app.viewport->process_input(window, *app.scene);
 
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window);
        }
    }
 
    // Shutdown ImGui while GL context is still live
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // Destroying GL context here
    glfwDestroyWindow(window);

    //Commenting glfwTerminate out is intentional because there is a bug in GLFW regarding use-after-free and wayland. So we brute force
    //the OS to reclaim resources and clean up stuff by just exiting
    //glfwTerminate();
    exit(0);
}
 
