#include "job_tracker.hpp"

namespace ipc {
    void JobTracker::on_status(JobType job, JobStatus status) {
        // Backend dropped its whole queue on (re)connect: nothing is pending anymore
        if (status == JobStatus::RESET) {
            pending_.fill(0);
            failed_.fill(false);
            return;
        }

        const size_t i = index(job);
        if (i >= JOB_COUNT) {
            return;   // job type from a newer backend
        }

        const auto now = Clock::now();
        last_status_[i] = now;

        if (status == JobStatus::QUEUED) {
            if (pending_[i] == 0) {
                pending_since_[i] = now;
            }
            pending_[i]++;
            return;
        }

        // Clamp at 0: a job that was running during a RESET still reports back
        pending_[i] = std::max(0, pending_[i] - 1);
        failed_[i]  = (status == JobStatus::FAILED);
    }

    bool JobTracker::busy(JobType job) const {
        const size_t i = index(job);
        if (i >= JOB_COUNT || pending_[i] == 0) {
            return false;
        }

        const auto now = Clock::now();
        if (now - last_status_[i] >= STALL_TIMEOUT) {
            return false;
        }

        return now - pending_since_[i] >= SHOW_DELAY;
    }

    bool JobTracker::failed(JobType job) const {
        const size_t i = index(job);
        return i < JOB_COUNT && failed_[i];
    }

    bool JobTracker::stalled(JobType job) const {
        const size_t i = index(job);
        if (i >= JOB_COUNT || pending_[i] == 0) {
            return false;
        }

        return Clock::now() - last_status_[i] >= STALL_TIMEOUT;
    }

}
    


