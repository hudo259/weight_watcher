#pragma once

namespace ui {

    // Whether widgets under a 'busy' overlay still accept input
    enum class BusyInput {
        BLOCK, // request controls: avoid stacking up more backend jobs
        ALLOW, // display-only views: keep scrolling, zooming, toggles usable
    };

    // Scoped block of ImGui widgets taht waits on a backend job. while busy, it draws a dimmed overlay with a spinner and a label across the full width.
    //
    // e.g.:
    //  {
    //      ui::BusySection busy(is_busy, "Computing...", ui::BusyInput::BLOCK);
    //      ImGui::Button("Compute");
    //  }   // overlay drawn here, on top of the button
    class BusySection {
        public:
            BusySection(bool busy, const char* label, BusyInput input);
            ~BusySection();

            BusySection(const BusySection& bs) = delete;
            BusySection& operator=(const BusySection& bs) = delete;

        private:
            bool busy_;
            bool disabled_;
            const char* label_;
            float right_edge_; //screen x of the available region's right edge
    };

}
