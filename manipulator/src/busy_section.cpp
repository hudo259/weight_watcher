#include "busy_section.hpp"

#include <imgui.h>

namespace ui {

    namespace {
        // Note: These hard-coded values and the draw_overlay function have been generated with the help of Claude to make this spinner
        // look nice.

        constexpr float SPINNER_RADIUS    = 8.0f;
        constexpr float SPINNER_THICKNESS = 2.5f;
        constexpr float SPINNER_SPEED     = 6.0f;              // rad/s
        constexpr float PI                = 3.14159265f;      // IM_PI lives in imgui_internal.h
        constexpr float SPINNER_ARC       = 1.5f * PI;        // 270 deg of the circle visible
        constexpr int   SPINNER_SEGMENTS  = 24;
        constexpr ImU32 DIM_COLOR         = IM_COL32(0, 0, 0, 120);

        // Dimmed rect over [min, max] with a rotating arc and label centred in it
        void draw_overlay(ImVec2 min, ImVec2 max, const char* label) {
            ImDrawList*   dl    = ImGui::GetWindowDrawList();
            const ImU32   color = ImGui::GetColorU32(ImGuiCol_Text);
            const float   gap   = ImGui::GetStyle().ItemSpacing.x;

            dl->AddRectFilled(min, max, DIM_COLOR, ImGui::GetStyle().FrameRounding);

            // Centre spinner + label as one unit
            const ImVec2 text_size = ImGui::CalcTextSize(label);
            const float  total_w   = 2.0f * SPINNER_RADIUS + gap + text_size.x;
            const ImVec2 center(0.5f * (min.x + max.x - total_w) + SPINNER_RADIUS,
                                0.5f * (min.y + max.y));

            const float start = static_cast<float>(ImGui::GetTime()) * SPINNER_SPEED;
            dl->PathArcTo(center, SPINNER_RADIUS, start, start + SPINNER_ARC, SPINNER_SEGMENTS);
            dl->PathStroke(color, ImDrawFlags_None, SPINNER_THICKNESS);

            dl->AddText(ImVec2(center.x + SPINNER_RADIUS + gap, center.y - 0.5f * text_size.y),
                        color, label);
                

        }
    }

    BusySection::BusySection(bool busy, const char* label, BusyInput input)
        : busy_(busy),
          disabled_(busy && input == BusyInput::BLOCK),
          label_(label),
          right_edge_(ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x) {
        // Group so the section's bounding box is available afterwards
        ImGui::BeginGroup();
        ImGui::BeginDisabled(disabled_);
    }

    BusySection::~BusySection() {
        ImGui::EndDisabled();
        ImGui::EndGroup();

        if (!busy_) {return;}

        // Stretch to the full width: a group is only as wide as its widest item
        const ImVec2 min = ImGui::GetItemRectMin();
        const ImVec2 max(right_edge_, ImGui::GetItemRectMax().y);
        draw_overlay(min, max, label_);
    }

}
