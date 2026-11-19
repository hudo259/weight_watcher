#include "image_viewer.hpp"
#include "busy_section.hpp"

#include <algorithm>
#include <iostream>
#include <cstring>

// CPU-side format conversion
static uint8_t* to_rgba(const ipc::ImageMsg& msg) {
    const size_t   N   = (size_t)msg.width * msg.height;
    uint8_t*       out = new uint8_t[N * 4];
    const uint8_t* s   = msg.data.data();

    switch (msg.format) {
        case ipc::ImageFormat::GRAYSCALE:
            for (size_t i = 0; i < N; i++) {
                uint8_t v = s[i];
                out[i*4+0]=v; out[i*4+1]=v; out[i*4+2]=v; out[i*4+3]=255;
            }
            break;
        case ipc::ImageFormat::RGB:
            for (size_t i = 0; i < N; i++) {
                out[i*4+0]=s[i*3+0]; out[i*4+1]=s[i*3+1];
                out[i*4+2]=s[i*3+2]; out[i*4+3]=255;
            }
            break;
        case ipc::ImageFormat::BGR:
            for (size_t i = 0; i < N; i++) {
                out[i*4+0]=s[i*3+2]; out[i*4+1]=s[i*3+1];
                out[i*4+2]=s[i*3+0]; out[i*4+3]=255;
            }
            break;
        case ipc::ImageFormat::RGBA:
            std::memcpy(out, s, N * 4);
            break;
        case ipc::ImageFormat::BGRA:
            for (size_t i = 0; i < N; i++) {
                out[i*4+0]=s[i*4+2]; out[i*4+1]=s[i*4+1];
                out[i*4+2]=s[i*4+0]; out[i*4+3]=s[i*4+3];
            }
            break;
        default:
            for (size_t i = 0; i < N; i++) {
                out[i*4+0]=255; out[i*4+1]=0; out[i*4+2]=255; out[i*4+3]=255;
            }
    }
    return out;
}

ImageViewer::ImageViewer() {
    modes_[0] = &raw_mode_;
    modes_[1] = &pred_mode_;
    modes_[2] = &heat_mode_;
    active_mode_ = &raw_mode_;
}

ImageViewer::~ImageViewer() {
    clear();
}

void ImageViewer::init() {
    if (initialized_) return;
    for (int i = 0; i < MODE_COUNT; i++)
        modes_[i]->init();
    initialized_ = true;
}

void ImageViewer::clear() {
    for (auto& e : entries_raw_)
        if (e.texture) { glDeleteTextures(1, &e.texture); e.texture = 0; }
    entries_raw_.clear();

    for (int i = 0; i < MODE_COUNT; i++)
        modes_[i]->clear();

    current_idx_ = 0;
}

// Raw image upload
GLuint ImageViewer::upload_rgba(const ipc::ImageMsg& msg, GLuint existing) {
    uint8_t* rgba = to_rgba(msg);

    GLuint tex = existing;
    if (!tex) glGenTextures(1, &tex);

    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                 (GLsizei)msg.width, (GLsizei)msg.height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    glBindTexture(GL_TEXTURE_2D, 0);

    delete[] rgba;
    return tex;
}

void ImageViewer::free_tex(GLuint& tex) {
    if (tex) { glDeleteTextures(1, &tex); tex = 0; }
}

void ImageViewer::upload_raw_batch(const ipc::ImageBatchMsg& batch) {
    while (entries_raw_.size() < batch.images.size()) entries_raw_.push_back({});
    while (entries_raw_.size() > batch.images.size()) {
        free_tex(entries_raw_.back().texture);
        entries_raw_.pop_back();
    }
    for (size_t i = 0; i < batch.images.size(); i++) {
        const auto& msg = batch.images[i];
        auto&       e   = entries_raw_[i];
        e.id      = msg.id;
        e.width   = msg.width;
        e.height  = msg.height;
        e.name    = msg.name;
        e.texture = upload_rgba(msg, e.texture);
    }
}

// IPC dispatch
void ImageViewer::add_image_batch(ipc::ImageBatchMsg batch) {
    // Store raw images in ImageViewer — they are the base for all modes
    if (batch.batch_type == ipc::ImageBatchType::RAW_IMAGES) {
        upload_raw_batch(batch);
        active_mode_ = &raw_mode_;
        show_window_ = true;
        std::cout << "[ImageViewer] Stored " << entries_raw_.size()
                  << " raw images\n";
    }

    // Forward to all modes, each filters by its own batch_type
    for (int i = 0; i < MODE_COUNT; i++)
        modes_[i]->upload(batch);

    // Auto-switch to the mode that just received new data
    if (batch.batch_type == ipc::ImageBatchType::PREDICTIONS) {
        active_mode_ = &pred_mode_;
        show_window_ = true;
    } else if (batch.batch_type == ipc::ImageBatchType::ACTIVATION_MAP) {
        active_mode_ = &heat_mode_;
        show_window_ = true;
    }
}

void ImageViewer::attach(ipc::IPCRouter& router) {
    jobs_ = &router.jobs();

    router.on_image_batch([this](ipc::ImageBatchMsg msg) {
        add_image_batch(std::move(msg));
    });
}

// Texture resolution
uint32_t ImageViewer::raw_width (int idx) const {
    return (idx < (int)entries_raw_.size()) ? entries_raw_[idx].width  : 0;
}
uint32_t ImageViewer::raw_height(int idx) const {
    return (idx < (int)entries_raw_.size()) ? entries_raw_[idx].height : 0;
}

GLuint ImageViewer::resolve(int idx, int display_w, int display_h) {
    GLuint raw_tex = (idx < (int)entries_raw_.size())
                     ? entries_raw_[idx].texture : 0;
    return active_mode_->resolve(raw_tex, idx, display_w, display_h);
}

int ImageViewer::active_count() const {
    int mode_count = active_mode_->count();
    // RawImageMode returns 0 meaning "use raw count"
    return (mode_count > 0) ? mode_count : (int)entries_raw_.size();
}

// ImGui drawing
static void center_cursor_x(float avail_w, float item_w) {
    float off = (avail_w - item_w) * 0.5f;
    if (off > 0.0f)
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + off);
}

void ImageViewer::draw() {
    if (!initialized_ || !show_window_) return;
    if (active_count() == 0) return;

    ImGui::Begin("Image Viewer", &show_window_);
    draw_toolbar();
    ImGui::Separator();
    // Predictions follow weight chanes, activation maps follow requests
    {
        const bool busy = jobs_ && (jobs_->busy(ipc::JobType::ACTIVATION_REQUEST) 
                                    || jobs_->busy(ipc::JobType::TRANSFORMED_POINTS)
                                    || jobs_->busy(ipc::JobType::POINT_CLOUD)
                                   );
        ui::BusySection section(busy, "Waiting for images...", ui::BusyInput::ALLOW);

        if (view_mode_ == ViewMode::Single) { 
            draw_single_view(); 
        }
        else { 
            draw_grid_view(); 
        }
    }
    ImGui::End();
}

void ImageViewer::draw_toolbar() {
    // View layout toggle
    if (ImGui::RadioButton("Single", view_mode_ == ViewMode::Single))
        view_mode_ = ViewMode::Single;
    ImGui::SameLine();
    if (ImGui::RadioButton("Grid", view_mode_ == ViewMode::Grid))
        view_mode_ = ViewMode::Grid;

    ImGui::SameLine();
    ImGui::Text("|");
    ImGui::SameLine();

    // Mode radio buttons, one per mode, disabled if the mode has no data
    for (int i = 0; i < MODE_COUNT; i++) {
        IImageMode* m = modes_[i];
        bool is_active = (m == active_mode_);

        // Raw mode is always available (entries_raw_ is always present).
        // Other modes are disabled until they have received data.
        bool has_data = (m == &raw_mode_) || (m->count() > 0);

        ImGui::BeginDisabled(!has_data);
        if (ImGui::RadioButton(m->label(), is_active))
            active_mode_ = m;
        ImGui::EndDisabled();
        if (i < MODE_COUNT - 1) ImGui::SameLine();
    }

    // Alpha slider (only for modes that blend two images)
    if (active_mode_->needs_alpha()) {
        ImGui::SameLine();
        ImGui::Text("|");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120.0f);
        float a = active_mode_->get_alpha();
        if (ImGui::SliderFloat("Alpha##ov", &a, 0.0f, 1.0f, "%.2f"))
            active_mode_->set_alpha(a);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("0 = raw image only\n1 = overlay only");
    }

    ImGui::SameLine();
    ImGui::Text("|");
    ImGui::SameLine();
    if (ImGui::SmallButton("Clear")) clear();
}

void ImageViewer::draw_single_view() {
    int n = active_count();
    if (n == 0) { ImGui::TextDisabled("(no images)"); return; }

    current_idx_ = std::clamp(current_idx_, 0, n - 1);

    // Navigation
    ImGui::BeginDisabled(current_idx_ == 0);
    if (ImGui::ArrowButton("##prev", ImGuiDir_Left))  current_idx_--;
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(current_idx_ == n - 1);
    if (ImGui::ArrowButton("##next", ImGuiDir_Right)) current_idx_++;
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::Text("%d / %d  (%ux%u)",
                current_idx_ + 1, n,
                raw_width(current_idx_), raw_height(current_idx_));

    ImVec2 avail = ImGui::GetContentRegionAvail();
    uint32_t iw = raw_width (current_idx_);
    uint32_t ih = raw_height(current_idx_);

    float scale = (iw > 0 && ih > 0)
                  ? std::min({avail.x / (float)iw, avail.y / (float)ih, 1.0f})
                  : 1.0f;
    ImVec2 disp((float)iw * scale, (float)ih * scale);
    center_cursor_x(avail.x, disp.x);

    GLuint tex = resolve(current_idx_, (int)disp.x, (int)disp.y);
    if (tex)
        ImGui::Image(static_cast<ImTextureID>((uintptr_t)tex), disp);
}

void ImageViewer::draw_grid_view() {
    ImGui::SliderInt("Columns##gv", &grid_columns_, 1, 8);

    int    n       = active_count();
    ImVec2 avail   = ImGui::GetContentRegionAvail();
    float  spacing = ImGui::GetStyle().ItemSpacing.x;
    float  cell_w  = (avail.x - spacing * (float)(grid_columns_ - 1))
                     / (float)grid_columns_;

    ImGui::BeginChild("##grid", ImVec2(0, 0), ImGuiChildFlags_None,
                      ImGuiWindowFlags_HorizontalScrollbar);

    int col = 0;
    for (int i = 0; i < n; i++) {
        if (col > 0) ImGui::SameLine();
        ImGui::PushID(i);
        ImGui::BeginGroup();

        uint32_t iw     = raw_width (i);
        uint32_t ih     = raw_height(i);
        float    aspect = (ih > 0) ? (float)iw / (float)ih : 1.0f;
        ImVec2   cell(cell_w, cell_w / aspect);

        GLuint tex = resolve(i, (int)cell.x, (int)cell.y);
        if (tex)
            ImGui::Image(static_cast<ImTextureID>((uintptr_t)tex), cell);

        // Label below image — name from active mode's entries, fallback to raw
        const std::string* label = nullptr;
        if (i < (int)entries_raw_.size())
            label = &entries_raw_[i].name;

        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + cell_w);
        if (label && !label->empty()) ImGui::TextWrapped("%s", label->c_str());
        else                          ImGui::TextWrapped("img %d", i);
        ImGui::PopTextWrapPos();

        ImGui::EndGroup();

        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("Image %d  (%ux%u)", i, iw, ih);
            ImGui::EndTooltip();
        }
        if (ImGui::IsItemClicked()) {
            current_idx_ = i;
            view_mode_   = ViewMode::Single;
        }

        ImGui::PopID();
        if (++col >= grid_columns_) col = 0;
    }

    ImGui::EndChild();
}
