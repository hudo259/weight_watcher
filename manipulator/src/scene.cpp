#include "scene.hpp"
#include "busy_section.hpp"
#include "icamera.hpp"

#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <iostream>
#include <string>

//TODO: This class / file is large because it handles multiple things. I can't think of entangling it without introducing additional complexity elsewhere; it doesn't seem like a worthwhile tradeoff.
// However, I still think we could throw the ImGui stuff out into a SceneUI component, which is then used here.

static glm::vec3 resolve_layer_color(glm::vec3 provided, size_t i) {
    static constexpr glm::vec3 PALETTE[] = {
        { 0.4f, 0.6f, 0.9f }, { 0.9f, 0.5f, 0.3f }, { 0.4f, 0.8f, 0.5f },
        { 0.9f, 0.4f, 0.7f }, { 0.7f, 0.5f, 0.9f }, { 0.9f, 0.8f, 0.3f },
    };
    if (provided != glm::vec3(0)) return provided;
    return PALETTE[i % std::size(PALETTE)];
}

void Scene::init(int width, int height) {
    pick_buffer_.create(width, height);
    compile_shaders();
    buffer_.init();
    glGenBuffers(1, &color_ssbo_);
    glGenBuffers(1, &layout_offset_ssbo_);
}

void Scene::resize(int width, int height) {
    pick_buffer_.resize(width, height);
}

Scene::~Scene() {
    on_concept_select = nullptr;
    on_activation_request = nullptr;

    buffer_.destroy();

    if (color_ssbo_) glDeleteBuffers(1, &color_ssbo_);
    color_ssbo_ = 0;

    if (layout_offset_ssbo_) glDeleteBuffers(1, &layout_offset_ssbo_);
    layout_offset_ssbo_ = 0;
}

// Per-cloud GPU upload helpers
void Scene::upload_cloud_positions(uint32_t cloud_idx) {
    buffer_.upload_positions(slots_[cloud_idx].gpu,
                             clouds_[cloud_idx]->points().data());
    clouds_[cloud_idx]->clear_positions_dirty();
}

void Scene::sync_cloud_selection(uint32_t cloud_idx) {
    buffer_.sync_selection(slots_[cloud_idx].gpu,
                           clouds_[cloud_idx]->build_reordered_selection());
    clouds_[cloud_idx]->clear_selection_dirty();
}

void Scene::rebuild_selection_ssbo() {
    buffer_.clear_selection();
    for (size_t i = 0; i < clouds_.size(); i++) {
        if (!slots_[i].alive) continue;
        buffer_.sync_selection(slots_[i].gpu,
                               clouds_[i]->build_reordered_selection());
        clouds_[i]->clear_selection_dirty();
    }
}

// Concept heatmap
void Scene::set_concept_scores(const ipc::VectorListMsg& msg) {
    // msg.data is a flat array: one float per point, all live clouds in the
    // order they appear in clouds_/slots_ (same as the layer order the
    // backend uses when it flattens its _layer_keys).
    uint32_t offset = 0;
    for (size_t i = 0; i < clouds_.size(); i++) {
        if (!slots_[i].alive) continue;
        const uint32_t n = slots_[i].gpu.count;
        
        // upload raw scores directly, the shader handles the colormap
        buffer_.upload_heatmap_scores(slots_[i].gpu, &(msg.data[offset]));
        offset += n;
    }

    heatmap_active_ = true;
    // u_HeatmapActive is set in setup_main_shader() which is called every
    // frame at the top of render(), so no explicit uniform call is needed here.
}

void Scene::clear_concept_scores() {
    buffer_.clear_heatmap_scores();
    heatmap_active_ = false;
}

// load / add / remove / defrag (Cloud management)
void Scene::load_layers(std::vector<glm::vec3>  points,
                        std::vector<std::string> layer_names,
                        std::vector<uint32_t>    layer_counts,
                        std::vector<glm::vec3>   layer_colors)
{
    clouds_.clear();
    slots_.clear();
    undo_stack_.clear();
    redo_stack_.clear();
    removal_count_ = 0;
    heatmap_active_ = false;
    ui_ = {};
    layout_settings.received_layouts_.clear();
    layout_settings.layout_names_.clear();
    layout_settings.active_layout_id_ = ipc::LayoutID::DEPTH;

    buffer_.init(std::max(buffer_.capacity(), GpuPointBuffer::INITIAL_CAPACITY));

    if (layer_names.empty()) {
        layer_names  = { "Cloud 0" };
        layer_counts = { static_cast<uint32_t>(points.size()) };
    }
    layer_colors.resize(layer_names.size(), glm::vec3(0));

    uint32_t offset = 0;
    for (size_t i = 0; i < layer_names.size(); ++i) {
        uint32_t count = layer_counts[i];
        std::vector<glm::vec3> layer_pts(points.begin() + offset,
                                          points.begin() + offset + count);
        auto cloud = std::make_unique<PointCloud>();
        cloud->name    = layer_names[i];
        cloud->color   = resolve_layer_color(layer_colors[i], i);
        cloud->visible = true;
        cloud->set_points(std::move(layer_pts));

        GpuPointBuffer::Slot gpu = buffer_.allocate(count);
        cloud->set_vbo_base(gpu.base);
        slots_.push_back({gpu, glm::vec3(0.f), true});
        clouds_.push_back(std::move(cloud));
        upload_cloud_positions(static_cast<uint32_t>(clouds_.size() - 1));
        offset += count;
    }
}

uint32_t Scene::add_cloud(std::vector<glm::vec3> points,
                           std::string            name,
                           glm::vec3              color)
{
    if (name.empty())
        name = "Cloud " + std::to_string(clouds_.size());

    auto cloud = std::make_unique<PointCloud>();
    cloud->name    = std::move(name);
    cloud->color   = resolve_layer_color(color, clouds_.size());
    cloud->visible = true;
    cloud->set_points(std::move(points));

    uint32_t count = static_cast<uint32_t>(cloud->size());
    GpuPointBuffer::Slot gpu = buffer_.allocate(count);
    cloud->set_vbo_base(gpu.base);

    uint32_t cloud_idx = static_cast<uint32_t>(clouds_.size());
    slots_.push_back({gpu, glm::vec3(0.f), true});
    clouds_.push_back(std::move(cloud));
    upload_cloud_positions(cloud_idx);
    return cloud_idx;
}

void Scene::remove_cloud(uint32_t cloud_idx) {
    if (cloud_idx >= clouds_.size() || !slots_[cloud_idx].alive) return;
    buffer_.free(slots_[cloud_idx].gpu);
    slots_[cloud_idx].alive = false;
    clouds_[cloud_idx].reset();
    if (defrag_threshold > 0 && ++removal_count_ >= defrag_threshold)
        defrag();
}

void Scene::defrag() {
    std::vector<GpuPointBuffer::Slot> gpu_slots(slots_.size());
    std::vector<bool>                  alive(slots_.size());
    for (size_t i = 0; i < slots_.size(); i++) {
        gpu_slots[i] = slots_[i].gpu;
        alive[i]     = slots_[i].alive;
    }
    buffer_.defrag(gpu_slots, alive);
    for (size_t i = 0; i < slots_.size(); i++) {
        if (!slots_[i].alive) continue;
        slots_[i].gpu = gpu_slots[i];
        clouds_[i]->set_vbo_base(gpu_slots[i].base);
    }
    rebuild_selection_ssbo();
    removal_count_ = 0;
}

// Rendering
void Scene::render(const glm::mat4& view, const glm::mat4& proj,
                   const glm::vec3& camera_pos, const Frustum* frustum)
{
    for (size_t i = 0; i < clouds_.size(); i++) {
        if (!slots_[i].alive) continue;
        if (clouds_[i]->is_positions_dirty())
            upload_cloud_positions(static_cast<uint32_t>(i));
        if (clouds_[i]->is_selection_dirty())
            sync_cloud_selection(static_cast<uint32_t>(i));
    }

    setup_main_shader(view, proj, camera_pos);
    stats = {};

    std::vector<GLint>     firsts;
    std::vector<GLsizei>   draw_counts;
    std::vector<glm::vec4> draw_colors;
    std::vector<glm::vec4> draw_offsets;

    for (size_t i = 0; i < clouds_.size(); ++i) {
        if (!slots_[i].alive) continue;
        stats.selected_points += clouds_[i]->selection_count();

        const auto& cloud = clouds_[i];
        const auto& slot  = slots_[i];
        if (!cloud->visible) continue;

        stats.total_points += cloud->size();
        stats.total_chunks += cloud->chunk_count();

        glm::vec4 color(cloud->color, 1.0f);
        glm::vec4 off(slot.layout_offset, 0.f);

        if (render_settings.use_frustum_culling && frustum) {
            for (const auto& chunk : cloud->chunks()) {
                // shift AABBs into world space before culling
                if (!frustum->intersects_aabb(
                        chunk.aabb_min + slot.layout_offset,
                        chunk.aabb_max + slot.layout_offset))
                    continue;

                firsts.push_back((GLint)(slot.gpu.base + chunk.offset));
                draw_counts.push_back((GLsizei)chunk.count);
                draw_colors.push_back(color);
                draw_offsets.push_back(off);

                stats.visible_chunks++;
                stats.visible_points += chunk.count;
            }
        } else {
            firsts.push_back((GLint)slot.gpu.base);
            draw_counts.push_back((GLsizei)slot.gpu.count);
            draw_colors.push_back(color);
            draw_offsets.push_back(off);

            stats.visible_chunks += cloud->chunk_count();
            stats.visible_points += cloud->size();
        }
    }

    if (firsts.empty()) return;

    // Upload SSBOs before the draw call
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, color_ssbo_);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)(draw_colors.size() * sizeof(glm::vec4)),
                 draw_colors.data(), GL_STREAM_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, layout_offset_ssbo_);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)(draw_offsets.size() * sizeof(glm::vec4)),
                 draw_offsets.data(), GL_STREAM_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer_.sel_ssbo());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, color_ssbo_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, layout_offset_ssbo_);

    glBindVertexArray(buffer_.vao());

    glMultiDrawArrays(GL_POINTS,
                      firsts.data(),
                      draw_counts.data(),
                      static_cast<GLsizei>(firsts.size()));

    glBindVertexArray(0);
}

void Scene::render_pick_pass(const glm::mat4& vp, const glm::mat4& view,
                              float point_size, const Frustum* frustum)
{
    pick_buffer_.bind();
    pick_buffer_.clear();

    pick_shader_pipeline_.bind();
    pick_vert_->set_mat4("u_VP",         vp);
    pick_vert_->set_mat4("u_View",       view);
    pick_vert_->set_float("u_PointSize", point_size);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glBindVertexArray(buffer_.vao());

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, layout_offset_ssbo_);

    for (uint32_t ci = 0; ci < (uint32_t)clouds_.size(); ci++) {
        const auto& cloud = clouds_[ci];
        const auto& slot  = slots_[ci];
        if (!slot.alive || !cloud->visible) continue;

        for (uint32_t ki = 0; ki < (uint32_t)cloud->chunks().size(); ki++) {
            const auto& chunk = cloud->chunks()[ki];
            if (frustum && !frustum->intersects_aabb(chunk.aabb_min + slot.layout_offset, chunk.aabb_max + slot.layout_offset))
                continue;

            uint32_t global_chunk_offset = slot.gpu.base + chunk.offset;
            pick_vert_->set_uint("u_ChunkOffset", global_chunk_offset);
            pick_frag_->set_uint("u_GroupID",     encode_group(ci, ki));
            pick_vert_->set_vec3("u_LayoutOffset", slot.layout_offset);
            glDrawArrays(GL_POINTS, (GLint)global_chunk_offset, (GLsizei)chunk.count);
        }
    }

    glBindVertexArray(0);
    pick_buffer_.unbind();
}

// Picking
Scene::ResolvedPick Scene::pick_at(int x, int y) {
    return resolve(pick_buffer_.read_pixel(x, y));
}

std::unordered_map<uint32_t, std::vector<uint32_t>>
Scene::box_select(int x0, int y0, int x1, int y1)
{
    auto picks = pick_buffer_.read_rect(x0, y0, x1, y1);
    std::unordered_map<uint32_t, std::vector<uint32_t>> by_cloud;
    for (const auto& pick : picks) {
        auto r = resolve(pick);
        if (r.valid) by_cloud[r.cloud_index].push_back(r.point_index);
    }
    for (auto& [ci, idx] : by_cloud) {
        std::sort(idx.begin(), idx.end());
        idx.erase(std::unique(idx.begin(), idx.end()), idx.end());
    }
    return by_cloud;
}

Scene::ResolvedPick Scene::resolve(const PickResult& pick) {
    if (!pick.valid()) return {};

    uint32_t cloud_idx, chunk_idx;
    decode_group(pick.group_id, cloud_idx, chunk_idx);

    if (cloud_idx >= clouds_.size() || !slots_[cloud_idx].alive) return {};

    auto& cloud = clouds_[cloud_idx];
    if (chunk_idx >= cloud->chunks().size()) return {};

    const auto& chunk        = cloud->chunks()[chunk_idx];
    uint32_t    reordered    = chunk.offset + pick.local_id;
    uint32_t    original     = cloud->new_to_original(reordered);

    if (original == UINT32_MAX) return {};

    return {true, cloud_idx, original, cloud.get()};
}

// Selection accessors
PointCloud* Scene::cloud(uint32_t index) {
    if (index >= clouds_.size() || !slots_[index].alive) return nullptr;
    return clouds_[index].get();
}
size_t Scene::cloud_count()      const { return clouds_.size(); }
size_t Scene::live_cloud_count() const {
    size_t n = 0;
    for (const auto& s : slots_) n += s.alive ? 1 : 0;
    return n;
}

void Scene::clear_all_selections() {
    for (size_t i = 0; i < clouds_.size(); i++)
        if (slots_[i].alive) clouds_[i]->clear_selection();
}
bool Scene::has_any_selection() const {
    for (size_t i = 0; i < clouds_.size(); i++)
        if (slots_[i].alive && clouds_[i]->has_selection()) return true;
    return false;
}
glm::vec3 Scene::selection_centroid() const {
    glm::vec3 sum{0}; size_t count = 0;
    for (size_t i = 0; i < clouds_.size(); i++) {
        if (!slots_[i].alive) continue;
        const auto& c = *clouds_[i];
        if (c.has_selection()) {
            //the layout offset is only applied to the rendered points via the vertex shader, 
            //we don't really offset them to avoid adding and removing offsets continuously
            glm::vec3 world_centroid = c.selection_centroid() + slots_[i].layout_offset;

            sum += world_centroid * (float)c.selection_count();
            count += c.selection_count();
        }
    }
    return count ? sum / (float)count : glm::vec3(0);
}
void Scene::apply_transform_to_selection(const glm::mat4& delta,
                                          const glm::vec3& pivot)
{
    for (size_t i = 0; i < clouds_.size(); i++) {
        if (slots_[i].alive && clouds_[i]->has_selection()) {
            glm::vec3 local_pivot = pivot - slots_[i].layout_offset;
            clouds_[i]->apply_transform(delta, local_pivot);
        }
    }

}

void Scene::set_layer_isolation(std::optional<uint32_t> cloud_idx) {
    layout_settings.isolated_layer_idx_ = cloud_idx;
    layout_settings.layer_isolation_active_ = cloud_idx.has_value();
    for (size_t i = 0; i < clouds_.size(); ++i) {
        if (!slots_[i].alive) continue;
        clouds_[i]->visible = !layout_settings.layer_isolation_active_
                              || i == layout_settings.isolated_layer_idx_;
    }
}

glm::vec3 Scene::cloud_aabb_center(uint32_t cloud_idx) const {
    if (cloud_idx >= clouds_.size()) return glm::vec3(0);
    if (!slots_[cloud_idx].alive) return glm::vec3(0);
    const auto& pts = clouds_[cloud_idx]->points();
    if (pts.empty()) return glm::vec3(0);
    glm::vec3 mn = pts[0], mx = pts[0];
    for (const auto& p : pts) {
        mn = glm::min(mn, p);
        mx = glm::max(mx, p);
    }
    return (mn + mx) * 0.5f + slots_[cloud_idx].layout_offset;
}
// Undo / Redo
void Scene::push_undo(UndoEntry entry) {
    undo_stack_.push_back(std::move(entry));
    redo_stack_.clear();
}

void Scene::undo() {
    if (undo_stack_.empty()) return;

    auto entry  = std::move(undo_stack_.back());
    PointCloud* c = cloud(entry.cloud_id);
    if (c) {
        std::vector<glm::vec3> cur;
        cur.reserve(entry.indices.size());

        for (uint32_t idx : entry.indices) cur.push_back(c->get_point_by_original(idx));

        redo_stack_.push_back({entry.cloud_id, entry.indices, cur});
        for (size_t i = 0; i < entry.indices.size(); i++) {
            glm::vec3 d = entry.positions[i] - c->get_point_by_original(entry.indices[i]);
            c->select(entry.indices[i], false);
            c->apply_transform(glm::translate(glm::mat4(1.0f), d), entry.positions[i]);
        }

        c->clear_selection();
    }
    undo_stack_.pop_back();
    redo_stack_.push_back(std::move(entry));
}

void Scene::redo() {
    if (redo_stack_.empty()) return;

    auto entry  = std::move(redo_stack_.back());
    PointCloud* c = cloud(entry.cloud_id);
    if (c) {
        std::vector<glm::vec3> cur;
        cur.reserve(entry.indices.size());

        for (uint32_t idx : entry.indices) cur.push_back(c->get_point_by_original(idx));
        undo_stack_.push_back({entry.cloud_id, entry.indices, cur});

        for (size_t i = 0; i < entry.indices.size(); i++) {
            glm::vec3 d = entry.positions[i] - c->get_point_by_original(entry.indices[i]);
            c->select(entry.indices[i], false);
            c->apply_transform(glm::translate(glm::mat4(1.0f), d), entry.positions[i]);
        }
        c->clear_selection();
    }
    redo_stack_.pop_back();
    undo_stack_.push_back(std::move(entry));
}

void Scene::apply_state(const AppState &state) {
    render_settings.point_size = state.point_size;
    render_settings.bg_color = state.bg_color;
    render_settings.axis_color = state.axis_color;

    layout_settings.layout_scale_ = state.layout_scale;

    light.direction = state.direction;
    light.color = state.color;
    light.ambient = state.ambient;
    light.specular = state.specular;
    light.shininess = state.shininess;
}

void Scene::extract_state(AppState &state) const {
    state.point_size = render_settings.point_size;
    state.bg_color = render_settings.bg_color;
    state.axis_color = render_settings.axis_color;
    state.layout_scale = layout_settings.layout_scale_;
    state.direction = light.direction;
    state.color = light.color;
    state.ambient = light.ambient;
    state.specular = light.specular;
    state.shininess = light.shininess;
}

void Scene::attach(ipc::IPCRouter& router) {
    jobs_ = &router.jobs();

    // Receive
    router.on_layer_meta([this](ipc::LayerMetaMsg msg) {
        num_classes_ = static_cast<int>(msg.num_classes);

        // Build legend before moving msg
        legend_entries_.clear();
        std::unordered_map<std::string, glm::vec3> seen;
        for (size_t i = 0; i < msg.layer_types.size(); ++i) {
            const std::string& lt = msg.layer_types[i];
            if (seen.count(lt)) continue;

            glm::vec3 col = msg.colors[i];
            seen[lt] = col;

            legend_entries_.push_back({lt, col});
        }

        pending_layer_meta_ = std::move(msg);  // move AFTER reading because msg is undefined now.
    });

    router.on_point_cloud([this](ipc::PointCloudMsg msg) {
        if (!pending_layer_meta_) return;

        load_layers(std::move(msg.points),
                    std::move(pending_layer_meta_->names),
                    std::move(pending_layer_meta_->point_counts),
                    std::move(pending_layer_meta_->colors));

        pending_layer_meta_.reset();
    });

    router.on_vector_list(ipc::VectorListID::CONCEPT_SCORES,
        [this](ipc::VectorListMsg msg) { set_concept_scores(msg); });

    router.on_layout([this](ipc::LayoutMsg msg) {
        uint32_t id = msg.layout_id;
        layout_settings.layout_names_[id]     = msg.name;
        layout_settings.layout_axis_labels_[id]  = {msg.axis_x, msg.axis_y, msg.axis_z};
        layout_settings.received_layouts_[id] = std::move(msg.offsets);

        if (id == layout_settings.active_layout_id_)
            apply_layout(id);
    });

    // Send
    on_concept_select = [&router](int class_idx) {
        router.send_vector_list(ipc::VectorListID::CONCEPT_SELECT, 1,
            {static_cast<float>(class_idx)});
    };
    on_activation_request = [&router](std::vector<float>&& data) {
        router.send_vector_list(ipc::VectorListID::ACTIVATION_REQ, 3, std::move(data));
    };
}

// UI
void Scene::draw_ui_contents() {
    draw_ui_controls();
    ImGui::Separator();
    draw_ui_stats();
    draw_ui_rendering();
    draw_ui_lighting();
    draw_ui_layout();
    draw_ui_concept_heatmap();
    draw_ui_spatial_activation();
    draw_ui_buffer();
    ImGui::Separator();
    draw_ui_legend();
    draw_ui_layers();
    ImGui::Separator();
    draw_ui_generate();
}

bool Scene::draw_selection_ui_contents() {
    bool changed = false;
    if (ImGui::CollapsingHeader("Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("All (Ctrl+A)")) {
            for (size_t i = 0; i < cloud_count(); ++i) {
                if (cloud(i)) cloud(i)->select_all();
            }
            changed = has_any_selection();
        }

        ImGui::SameLine();
        if(ImGui::Button("Clear (Esc)")) {
            clear_all_selections();
        }

        ImGui::SameLine();
        if(ImGui::Button("Invert (Ctrl+I)")) {
            for(size_t i = 0; i < cloud_count(); ++i) {
                if (cloud(i)) cloud(i)->invert_selection();
            }
            changed = has_any_selection();
        }

        if(has_any_selection()) {
            glm::vec3 c = selection_centroid();
            ImGui::Text("Centroid (%.2f, %.2f, %.2f)", c.x, c.y, c.z);
        }
    }

    return changed;
}

// UI Sub-panels

void Scene::draw_ui_controls() {
    if (!ImGui::CollapsingHeader("Controls", ImGuiTreeNodeFlags_DefaultOpen)) return;
 
    ImGui::BulletText("Left-click: Select");
    ImGui::BulletText("Shift+click: Add to selection");
    ImGui::BulletText("Ctrl+click: Toggle");
    ImGui::BulletText("Right-drag+Alt: Box select");
    ImGui::BulletText("Middle-drag / Alt+Left: Orbit");
    ImGui::BulletText("Shift+Middle: Pan");
    ImGui::BulletText("Scroll: Zoom");
    ImGui::Separator();
    ImGui::BulletText("W / E / R: Translate / Rotate / Scale gizmo");
    ImGui::BulletText("Ctrl+A: Select all   Ctrl+I: Invert   Esc: Clear");
    ImGui::BulletText("Ctrl+Z / Ctrl+Y: Undo / Redo");
    ImGui::BulletText("F1 / F2: Orbit / Fly camera");
    ImGui::Separator();
    ImGui::BulletText("Fly mode — WASD: Move   Space/Ctrl: Up/Down   Shift: Sprint");
    ImGui::BulletText("Fly mode — Right-drag: Look");
}

void Scene::draw_ui_stats() {
    //stats are pre-computed in render(), we just read them here
    ImGui::Text("Points %zu    Selected: %zu    Clouds:%zu", 
        stats.total_points, stats.selected_points, live_cloud_count());

    if (render_settings.use_frustum_culling) {
        float pct = stats.total_points > 0
                    ? 100.0f * stats.visible_points / stats.total_points
                    : 0.0f;
        
        ImGui::Text("Visible: %zu / %zu (%.1f%%)    Chunks: %zu / %zu", 
            stats.visible_points, stats.total_points, pct, stats.visible_chunks, stats.total_chunks);
    }

    ImGui::Text("FPS: %.1f (%.2f ms)", 
        ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
}

void Scene::draw_ui_rendering() {
    if(!ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen)) return;

    ImGui::SliderFloat("Point Size", &render_settings.point_size, 1.0f, 50.0f);
    ImGui::Checkbox("Sphere Shading", &render_settings.fake_sphere_shading);
    ImGui::Checkbox("Soft Edges", &render_settings.soft_edges);
    ImGui::Checkbox("Frustum Culling", &render_settings.use_frustum_culling);
    ImGui::ColorEdit3("Selected", &render_settings.color_selected.x);

    ImGui::ColorEdit3("Background", &render_settings.bg_color.x);
    ImGui::ColorEdit3("Axis Labels", &render_settings.axis_color.x);
}

void Scene::draw_ui_lighting() {
    if (!ImGui::CollapsingHeader("Lighting"))
        return;
 
    if (ImGui::SliderFloat3("Direction", &light.direction.x, -1.0f, 1.0f)) {
        float len = glm::length(light.direction);
        if (len > 0.001f) light.direction /= len;
    }
    ImGui::ColorEdit3("Color",      &light.color.x);
    ImGui::SliderFloat("Ambient",   &light.ambient,   0.0f,   1.0f);
    ImGui::SliderFloat("Diffuse",   &light.diffuse,   0.0f,   1.0f);
    ImGui::SliderFloat("Specular",  &light.specular,  0.0f,   1.0f);
    ImGui::SliderFloat("Shininess", &light.shininess, 1.0f, 128.0f);
}
 
void Scene::draw_ui_layout() {
    if (!ImGui::CollapsingHeader("Layout"))
        return;
    
    if (layout_settings.received_layouts_.empty()) {
        ImGui::TextDisabled("Waiting for backend...");
        return;
    }

    // Layouts arrive one by one after connect. Switching between received ones stays possible
    const bool busy = jobs_ && (jobs_->busy(ipc::JobType::CKA_LAYOUT)
                                || jobs_->busy(ipc::JobType::TRAJECTORY_GEOMETRY)
                                || jobs_->busy(ipc::JobType::SENSITIVITY_LAYOUT)
                                || jobs_->busy(ipc::JobType::EV_LAYOUT)
                                || jobs_->busy(ipc::JobType::CORRELATION_LAYOUT)
                                );
    ui::BusySection section(busy, "Computing layouts...", ui::BusyInput::ALLOW);

    const std::string& active_name = layout_settings.layout_names_.count(layout_settings.active_layout_id_)
        ? layout_settings.layout_names_.at(layout_settings.active_layout_id_) : "Unknown";

    if (ImGui::BeginCombo("Layout##Combo", active_name.c_str())) {
        for (auto& [id, name] : layout_settings.layout_names_) {
            bool selected = (id == layout_settings.active_layout_id_);

            // Push a unique ID for each selectable item
            // Use the layout ID to ensure uniqueness
            ImGui::PushID(("layout_select_" + std::to_string(id)).c_str());

            if (ImGui::Selectable(name.c_str(), selected))
                apply_layout(id);

            if (selected)
                ImGui::SetItemDefaultFocus();

            ImGui::PopID();
        }
        ImGui::EndCombo();
    }

    const auto& labels = active_axis_labels();
    bool changed = false;
    changed |= ImGui::SliderFloat(
        (labels[0].empty() ? "X scale" : ("X: " + labels[0]).c_str()),
        &layout_settings.layout_scale_.x, 0.0f, 5.0f, "%.2f");
    changed |= ImGui::SliderFloat(
        (labels[1].empty() ? "Y scale" : ("Y: " + labels[1]).c_str()),
        &layout_settings.layout_scale_.y, 0.0f, 5.0f, "%.2f");
    changed |= ImGui::SliderFloat(
        (labels[2].empty() ? "Z scale" : ("Z: " + labels[2]).c_str()),
        &layout_settings.layout_scale_.z, 0.0f, 5.0f, "%.2f");

    if (changed)
        apply_layout(layout_settings.active_layout_id_);

    if (ImGui::Button("Reset scales")) {
        layout_settings.layout_scale_ = glm::vec3(1.0f);
        apply_layout(layout_settings.active_layout_id_);
    }
}

void Scene::draw_ui_concept_heatmap() {
    if (!ImGui::CollapsingHeader("Concept Heatmap"))
        return;
 
    if (num_classes_ == 0) {
        ImGui::TextDisabled("Waiting for backend...");
        return;
    }

    const bool busy = jobs_ && jobs_->busy(ipc::JobType::CONCEPT_SELECT);
    ui::BusySection section(busy, "Computing concept scores...", ui::BusyInput::BLOCK);
    if (jobs_ && jobs_->failed(ipc::JobType::CONCEPT_SELECT)) {
        ImGui::TextColored({1.0f, 0.4f, 0.4f, 1.0f}, "Last request failed. See backend log for details");
    }

    ImGui::TextDisabled("Colors all points by concept alignment.");
    ImGui::Separator();
 
    // Class selector
    int max_class = std::max(0, num_classes_ - 1);
    bool changed  = ImGui::SliderInt("Class##cav",
                                     &ui_.selected_class, 0, max_class);
 
    // Colormap combo
    ImGui::SetNextItemWidth(120.0f);
    int cmap_idx = static_cast<int>(ui_.colormap);
    if (ImGui::Combo("Colormap##cav", &cmap_idx,
                     ColorMaps::NAMES, IM_ARRAYSIZE(ColorMaps::NAMES)))
        ui_.colormap = static_cast<ColorMaps::ColorMapType>(cmap_idx);
 
    // Gradient scale bar
    ImGui::Spacing();
    ImGui::TextDisabled("-1");
    ImGui::SameLine();
 
    ImDrawList* dl    = ImGui::GetWindowDrawList();
    ImVec2      pos   = ImGui::GetCursorScreenPos();
    const float bar_w = 140.0f;
    const float bar_h = 12.0f;
    const int   steps = 64;
    const float sw    = bar_w / steps;
 
    for (int i = 0; i < steps; i++) {
        float t0 = static_cast<float>(i)     / steps * 2.0f - 1.0f;
        float t1 = static_cast<float>(i + 1) / steps * 2.0f - 1.0f;
 
        auto to_col = [&](float t) -> ImU32 {
            glm::vec3 c = glm::clamp(
                ColorMaps::apply(t, static_cast<int>(ui_.colormap)),
                0.0f, 1.0f);
            return IM_COL32(int(c.r * 255), int(c.g * 255), int(c.b * 255), 255);
        };
 
        dl->AddRectFilledMultiColor(
            ImVec2(pos.x +  i      * sw, pos.y),
            ImVec2(pos.x + (i + 1) * sw, pos.y + bar_h),
            to_col(t0), to_col(t1), to_col(t1), to_col(t0)
        );
    }
    ImGui::Dummy(ImVec2(bar_w, bar_h));
    ImGui::SameLine();
    ImGui::TextDisabled("+1");
 
    ImGui::Spacing();
 
    // Status
    if (heatmap_active_)
        ImGui::TextColored({0.4f, 0.9f, 0.4f, 1.0f}, "Heatmap active");
    else
        ImGui::TextDisabled("Heatmap off");
 
    ImGui::SameLine();
    if (ImGui::SmallButton("Clear##cav"))
        clear_concept_scores();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Restore solid layer colors");
 
    // Compute button — also fires when slider moves so the view updates live
    ImGui::BeginDisabled(!on_concept_select);
    if (ImGui::Button("Compute##cav") || changed)
        if (on_concept_select) on_concept_select(ui_.selected_class);
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Send class %d to backend for CAV scoring",
                          ui_.selected_class);
}
 
void Scene::draw_ui_spatial_activation() {
    if (!ImGui::CollapsingHeader("Spatial Activation"))
        return;
 
    // Spinner over the section while activation maps are computed...
    const bool busy = jobs_ && jobs_->busy(ipc::JobType::ACTIVATION_REQUEST);
    ui::BusySection section(busy, "Computing activation maps...", ui::BusyInput::BLOCK);

    // Count selected points
    size_t total_selected  = 0;
    size_t clouds_with_sel = 0;
    for (size_t i = 0; i < clouds_.size(); i++) {
        if (!slots_[i].alive) continue;
        size_t n = clouds_[i]->selection_count();
        if (n > 0) { total_selected += n; clouds_with_sel++; }
    }
 
    if (total_selected == 0)
        ImGui::TextDisabled("No points selected");
    else
        ImGui::Text("%zu point(s) across %zu cloud(s)",
                    total_selected, clouds_with_sel);
 
    // Aggregation mode
    bool is_avg = (ui_.aggregation == ActivationAggregation::Average);
    if (ImGui::RadioButton("Average##act", is_avg))
        ui_.aggregation = ActivationAggregation::Average;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Mean activation map across selected filters/layers.");
    ImGui::SameLine();
    if (ImGui::RadioButton("Maximum##act", !is_avg))
        ui_.aggregation = ActivationAggregation::Maximum;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Per-pixel maximum — shows full spatial extent.");
 
    // Request button
    ImGui::BeginDisabled(total_selected == 0 || !on_activation_request);
    if (ImGui::Button("Request Activation Map##act")) {
        float agg = (ui_.aggregation == ActivationAggregation::Average)
                    ? 0.0f : 1.0f;
 
        std::vector<float> data;
        data.reserve(total_selected * 3);
        for (size_t i = 0; i < clouds_.size(); i++) {
            if (!slots_[i].alive) continue;
            for (uint32_t orig_idx : clouds_[i]->selected_indices()) {
                data.push_back(static_cast<float>(i));
                data.push_back(static_cast<float>(orig_idx));
                data.push_back(agg);
            }
        }
        if (!data.empty())
            on_activation_request(std::move(data));
    }
    ImGui::EndDisabled();
 
    if (total_selected == 0)
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("Select points first");
}
 
void Scene::draw_ui_buffer() {
    if (!ImGui::CollapsingHeader("Buffer"))
        return;
 
    ImGui::Text("VBO capacity: %u pts", buffer_.capacity());
    ImGui::Text("VBO used:     %u pts", buffer_.used());
    ImGui::Text("Fragmentation: %.1f%%", buffer_.fragmentation() * 100.0f);
    ImGui::Text("Pending removals: %u",  removal_count_);
 
    int thresh = static_cast<int>(defrag_threshold);
    if (ImGui::SliderInt("Auto-defrag after N removals", &thresh, 0, 50))
        defrag_threshold = static_cast<uint32_t>(thresh);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("0 = manual only");
 
    if (ImGui::Button("Defrag now")) defrag();
}
 
void Scene::draw_ui_generate() {
    if (!ImGui::CollapsingHeader("Generate"))
        return;
 
    ImGui::SliderInt("Points",  &ui_.num_points,  1000, 10000000,
                     "%d", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Range", &ui_.point_range, 1.0f,    100.0f);
    ImGui::SliderInt("Seed",    &ui_.random_seed,    0,      9999);
 
    if (ImGui::Button("Add Cloud"))
        add_cloud(PointCloud::make_random(
                      ui_.num_points, ui_.point_range, ui_.random_seed), {});
}

void Scene::draw_ui_legend() {
    if (!ImGui::CollapsingHeader("Layer Types")) return;
    if (legend_entries_.empty()) {
        ImGui::TextDisabled("No layer type data");
        return;
    }
    for (const auto& entry : legend_entries_) {
        ImVec4 col(entry.color.r, entry.color.g, entry.color.b, 1.0f);
        ImGui::ColorButton(
            ("##leg_" + entry.name).c_str(), col,
            ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoDragDrop,
            ImVec2(16, 16)
        );
        ImGui::SameLine();
        ImGui::Text("%s", entry.name.c_str());
    }
}

void Scene::draw_ui_layers() {
    if (!ImGui::CollapsingHeader("Layers")) return;

    // Isolation status + clear button
    if (layout_settings.isolated_layer_idx_.has_value()) {
        ImGui::TextColored(ImVec4(1, 0.8f, 0.2f, 1),
                           "Isolated: %s",
                           clouds_[layout_settings.isolated_layer_idx_.value()]->name.c_str());
        ImGui::SameLine();
        if (ImGui::SmallButton("Show all"))
            set_layer_isolation(std::nullopt);
    } else {
        ImGui::TextDisabled("Click a layer to isolate it");
    }

    ImGui::Separator();
    ImGui::BeginChild("##layer_list", ImVec2(0, 300), false);
    for (size_t i = 0; i < clouds_.size(); ++i) {
        if (!slots_[i].alive) continue;

        bool is_isolated = layout_settings.isolated_layer_idx_.has_value()
                        && layout_settings.isolated_layer_idx_.value() == i;

        // Color swatch
        glm::vec3 c = clouds_[i]->color;
        ImGui::ColorButton(
            ("##col_" + std::to_string(i)).c_str(),
            ImVec4(c.r, c.g, c.b, 1.0f),
            ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoDragDrop,
            ImVec2(12, 12)
        );
        ImGui::SameLine();

        // Selectable row — highlighted if isolated
        if (ImGui::Selectable(clouds_[i]->name.c_str(), is_isolated,
                              0, ImVec2(0, 0))) {
            set_layer_isolation(is_isolated ? std::nullopt
                                           : std::optional<uint32_t>(i));
            if (layout_settings.isolated_layer_idx_.has_value()) {
                layout_settings.reframe_pending_ = true;
                layout_settings.reframe_target_  = cloud_aabb_center(layout_settings.isolated_layer_idx_.value());
            }
        }
    }
    ImGui::EndChild();
}
 
void Scene::apply_layout(uint32_t layout_id) {
    auto it = layout_settings.received_layouts_.find(layout_id);
    if (it == layout_settings.received_layouts_.end()) return;

    const auto& offsets = it->second;
    size_t li = 0;
    for (size_t i = 0; i < clouds_.size(); ++i) {
        if (!slots_[i].alive) continue;

        if (li < offsets.size())
            slots_[i].layout_offset = offsets[li++] * layout_settings.layout_scale_;
    }
    layout_settings.active_layout_id_ = layout_id;
}

// Shader setup
void Scene::setup_main_shader(const glm::mat4& view, const glm::mat4& proj,
                               const glm::vec3& camera_pos)
{
    main_shader_pipeline_.bind();

    main_vert_->set_int("u_Colormap", static_cast<int>(ui_.colormap));
    main_vert_->set_mat4("u_View",  view);
    main_vert_->set_mat4("u_Proj",  proj);
    main_vert_->set_float("u_PointSize",    render_settings.point_size);
    main_vert_->set_vec3("u_ColorSelected", render_settings.color_selected);

    // Tell the vertex shader whether to read attribute 1 (heatmap color) or
    // the per-draw-call color SSBO.  This is a plain int uniform updated every
    // frame so the change takes effect immediately without any other state change.
    main_vert_->set_int("u_HeatmapActive", heatmap_active_ ? 1 : 0);

    main_frag_->set_vec3("u_CameraPos",  camera_pos);
    main_frag_->set_vec3("u_LightDir",   glm::normalize(light.direction));
    main_frag_->set_vec3("u_LightColor", light.color);
    main_frag_->set_float("u_Ambient",   light.ambient);
    main_frag_->set_float("u_Diffuse",   light.diffuse);
    main_frag_->set_float("u_Specular",  light.specular);
    main_frag_->set_float("u_Shininess", light.shininess);
    main_frag_->set_int("u_FakeSphere",  render_settings.fake_sphere_shading ? 1 : 0);
    main_frag_->set_int("u_SoftEdges",   render_settings.soft_edges ? 1 : 0);
}

//TODO: in separate file?
void Scene::compile_shaders() {
    // Main vertex shader
    //
    // Attribute 0 ((aPos) vec3 world-space position)
    // Attribute 1 ((aScore) float raw heatmap score in [-1, 1])
    //
    // The colormap is applied per-vertex in the shader via u_Colormap, so the
    // heatmap VBO stores raw float scores, not pre-baked RGB colours.
    // Switching colormaps is free: just update u_Colormap, no VBO re-upload.
    //
    // Colormaps::GLSL_SOURCE is injected after the #version line and provides:
    //   vec3 apply_colormap(float score, int cmap)
    // which maps a score in [-1, 1] -> RGB using the selected colormap.
    const std::string main_vert =
        "#version 430 core\n"
        "#extension GL_ARB_shader_draw_parameters : require\n"
        + std::string(ColorMaps::GLSL_SOURCE) + R"(

        layout(location = 0) in vec3 aPos;
        layout(location = 1) in float aScore;   // raw [-1, 1] from heatmap VBO

        layout(std430, binding = 0) buffer SelectionBuffer {
            uint selectionMask[];
        };
        layout(std430, binding = 1) buffer LayerColors {
            vec4 layerColors[];
        };
        layout(std430, binding = 2) buffer LayerOffsets {
            vec4 layerOffsets[];
        };

        uniform int u_Colormap; //0=none, 1=viridis, 2=inferno, 3=coolwarm
        uniform mat4  u_View;
        uniform mat4  u_Proj;
        uniform float u_PointSize;
        uniform vec3  u_ColorSelected;
        uniform int   u_HeatmapActive;   // 0 = solid layer color, 1 = heatmap

        layout(location = 0) out vec3 vColor;
        layout(location = 1) out vec3 vWorldPos;
        out gl_PerVertex {
            vec4  gl_Position;
            float gl_PointSize;
        };

        void main() {
            uint wordIdx    = uint(gl_VertexID) / 32u;
            uint bitIdx     = uint(gl_VertexID) % 32u;
            bool isSelected = bool((selectionMask[wordIdx] >> bitIdx) & 1u);

            // Color priority:
            // 1. Selection highlight: always overrides everything else
            // 2. Heatmap color: if u_HeatmapActive == 1
            // 3. Per-layer solid color: default
            vec3 base_color = (u_HeatmapActive == 1)
                              ? apply_colormap(aScore, u_Colormap)
                              : layerColors[gl_DrawIDARB].rgb;
            vColor    = isSelected ? u_ColorSelected : base_color;
            vec3 worldPos = aPos + layerOffsets[gl_DrawIDARB].xyz;
            vWorldPos = worldPos;

            vec4 viewPos = u_View * vec4(worldPos, 1.0);
            gl_Position   = u_Proj * viewPos;
            float dist    = -viewPos.z;
            gl_PointSize  = u_PointSize * (10.0 / max(dist, 0.1));
        }
    )";

    const char* main_frag = R"(
        #version 430 core

        layout(location = 0) in vec3 vColor;
        layout(location = 1) in vec3 vWorldPos;
        out vec4 FragColor;

        uniform vec3  u_CameraPos;
        uniform vec3  u_LightDir;
        uniform vec3  u_LightColor;
        uniform float u_Ambient;
        uniform float u_Diffuse;
        uniform float u_Specular;
        uniform float u_Shininess;
        uniform int   u_FakeSphere;
        uniform int   u_SoftEdges;

        void main() {
            vec2  coord   = gl_PointCoord * 2.0 - 1.0;
            float dist_sq = dot(coord, coord);
            if (dist_sq > 1.0) discard;

            vec3  finalColor = vColor;
            float alpha      = 1.0;

            if (u_FakeSphere == 1) {
                float z      = sqrt(1.0 - dist_sq);
                vec3  normal = vec3(coord.x, -coord.y, z);
                vec3  ambient  = u_Ambient * u_LightColor;
                float diff     = max(dot(normal, u_LightDir), 0.0);
                vec3  diffuse  = u_Diffuse * diff * u_LightColor;
                vec3  halfDir  = normalize(u_LightDir + vec3(0.0, 0.0, 1.0));
                float spec     = pow(max(dot(normal, halfDir), 0.0), u_Shininess);
                vec3  specular = u_Specular * spec * u_LightColor;
                finalColor     = vColor * (ambient + diffuse) + specular;
            }

            if (u_SoftEdges == 1) {
                float d = sqrt(dist_sq);
                alpha = 1.0 - smoothstep(0.85, 1.0, d);
            }

            FragColor = vec4(finalColor, alpha);
        }
    )";

    main_frag_ = std::make_unique<Shader>(Shader::from_string(GL_FRAGMENT_SHADER, main_frag));
    main_vert_ = std::make_unique<Shader>(Shader::from_string(GL_VERTEX_SHADER,   main_vert.c_str()));
    main_shader_pipeline_.attach(*main_frag_).attach(*main_vert_);

    const char* pick_vert = R"(
        #version 430 core
        #extension GL_ARB_shader_draw_parameters : require

        layout(location = 0) in vec3 aPos;

        uniform mat4  u_VP;
        uniform mat4  u_View;
        uniform float u_PointSize;
        uniform uint  u_ChunkOffset;

        layout(location = 0) flat out uint vLocalID;
        //layout(std430, binding = 2) buffer LayerOffsets {
        //    vec4 layerOffsets[];
        //};
        uniform vec3 u_LayoutOffset;

        out gl_PerVertex { vec4 gl_Position; float gl_PointSize; };

        void main() {
            vLocalID     = uint(gl_VertexID) - u_ChunkOffset;
            //vec3 worldPos = aPos + layerOffsets[gl_DrawIDARB].xyz;
            vec3 worldPos = aPos + u_LayoutOffset;
            vec4 viewPos = u_View * vec4(worldPos, 1.0);
            gl_Position  = u_VP   * vec4(worldPos, 1.0);
            float dist   = -viewPos.z;
            gl_PointSize = u_PointSize * (10.0 / max(dist, 0.1));
        }
    )";

    const char* pick_frag = R"(
        #version 430 core
        layout(location = 0) flat in uint vLocalID;
        layout(location = 0) out uvec2 outID;
        uniform uint u_GroupID;
        void main() {
            vec2 coord = gl_PointCoord * 2.0 - 1.0;
            if (dot(coord, coord) > 1.0) discard;
            outID = uvec2(u_GroupID, vLocalID);
        }
    )";

    pick_frag_ = std::make_unique<Shader>(Shader::from_string(GL_FRAGMENT_SHADER, pick_frag));
    pick_vert_ = std::make_unique<Shader>(Shader::from_string(GL_VERTEX_SHADER,   pick_vert));
    pick_shader_pipeline_.attach(*pick_frag_).attach(*pick_vert_);
}
