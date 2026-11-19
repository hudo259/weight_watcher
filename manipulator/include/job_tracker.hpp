#pragma once

#include "ipc_messages.hpp"

#include <array>
#include <chrono>
#include <cstddef>

namespace ipc {

    // Counts backend jobs per type that are queued but not finished, fed by
    // JOB_STATUS messages. UI sections ask busy() to decide whether to show a spinner.
    //
    // e.g. QUEUED, QUEUED, DONE  ->  pending(CONCEPT_SELECT) == 1  ->  busy
    class JobTracker {
    public:
        void on_status(JobType job, JobStatus status);

        // Pending for longer than SHOW_DELAY and not stalled
        bool busy(JobType job) const;

        // Last finished job of this type raised in the backend
        bool failed(JobType job) const;

        // Pending, but no status for STALL_TIMEOUT (backend died or hung)
        bool stalled(JobType job) const;

    private:
        using Clock = std::chrono::steady_clock;

        // Fast jobs never show a spinner, so gizmo drags don't flicker
        static constexpr auto SHOW_DELAY    = std::chrono::milliseconds(250);
        static constexpr auto STALL_TIMEOUT = std::chrono::seconds(60);

        static constexpr size_t JOB_COUNT = static_cast<size_t>(JobType::COUNT);

        static size_t index(JobType job) { return static_cast<size_t>(job); }

        std::array<int, JOB_COUNT>               pending_{};
        std::array<Clock::time_point, JOB_COUNT> pending_since_{};
        std::array<Clock::time_point, JOB_COUNT> last_status_{};
        std::array<bool, JOB_COUNT>              failed_{};
    };
    
}
