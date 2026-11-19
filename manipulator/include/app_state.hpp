#pragma once

#include <glm/glm.hpp>
#include <string>
#include <fstream>

//Everything that we want to save from the session and be able to restore should get into here.
//This will be serialzed into a file and then can be deserialized again to restore the session state
struct AppState {
    // Render Settings
    float point_size = 1.0f;

    //color of background
    glm::vec3 bg_color = glm::vec3(255.0f, 255.0f, 255.0f);

    //color of coordinate axes and labels for the current layout
    glm::vec3 axis_color = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 layout_scale = glm::vec3(1.0f);
  
    //framebuffer resolution
    int fb_width = 1280;
    int fb_height = 720;

    // Light Settings
    glm::vec3 direction = glm::normalize(glm::vec3(0.5f, 0.8f, 1.0f));
    glm::vec3 color     = glm::vec3(1.0f);
    float ambient   = 0.25f;
    float diffuse   = 0.65f;
    float specular  = 0.4f;
    float shininess = 32.0f;

    // Camera State
    int       camera_mode      = 0; // 0 for Orbit, 1 for Fly
    float     yaw              = -1.57f;
    float     pitch            = 0.3f;
    glm::vec3 pos              = glm::vec3(0.0f, 0.0f, 50.0f);
    glm::vec3 target           = glm::vec3(0.0f);
    float     distance         = 20.0f;


    bool load(const std::string& filepath) {
        std::ifstream in(filepath);
        if (!in) return false;
        
        in >> bg_color.r >> bg_color.g >> bg_color.b;
        in >> axis_color.r >> axis_color.g >> axis_color.b;
        
        in >> camera_mode;
        in >> yaw >> pitch;
        in >> pos.x >> pos.y >> pos.z;
        in >> target.x >> target.y >> target.z;
        in >> distance;

        in >> point_size;
        in >> layout_scale.x >> layout_scale.y >> layout_scale.z;

        in >> fb_width >> fb_height;

        in >> direction.x >> direction.y >> direction.z;
        in >> color.x >> color.y >> color.z;
        in >> ambient;
        in >> diffuse;
        in >> specular;
        in >> shininess;
        return true;
    }

    bool save(const std::string& filepath) const {
        std::ofstream out(filepath);
        if (!out) return false;
        
        out << bg_color.r << " " << bg_color.g << " " << bg_color.b << "\n";
        out << axis_color.r << " " << axis_color.g << " " << axis_color.b << "\n";
        
        out << camera_mode << "\n";
        out << yaw << " " << pitch << "\n";
        out << pos.x << " " << pos.y << " " << pos.z << "\n";
        out << target.x << " " << target.y << " " << target.z << "\n";
        out << distance << "\n";

        out << point_size << "\n";
        out << layout_scale.x << " " << layout_scale.y << " " << layout_scale.z << "\n";

        out << fb_width << " " << fb_height << "\n";

        out << direction.x << " " << direction.y << " " << direction.z << "\n";
        out << color.x << " " << color.y << " " << color.z << "\n";
        out << ambient << "\n";
        out << diffuse << "\n";
        out << specular << "\n";
        out << shininess << "\n";
        return true;
    }
};