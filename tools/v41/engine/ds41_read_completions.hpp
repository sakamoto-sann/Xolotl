// A bounded completion queue: one notification per complete expert, never per partial read.
// Reader workers publish bytes through the mutex; only the engine thread calls CUDA.
#pragma once
#include <array>
#include <condition_variable>
#include <cstddef>
#include <mutex>

namespace strata::v41 {
template <size_t N> class Ds41ReadCompletions {
    std::array<int, N> left_ = {};
    std::array<int, N> ready_ = {};
    size_t size_ = 0, pushed_ = 0, popped_ = 0;
    std::mutex mu_;
    std::condition_variable cv_;
public:
    // Configure before submitting reads. A zero-part expert is already in the speculative stash.
    bool reset(size_t n, const std::array<int, N> & parts) {
        std::lock_guard<std::mutex> lock(mu_);
        if (n > N) return false;
        for (size_t i = 0; i < n; ++i) if (parts[i] < 0) return false;
        size_ = n; left_ = parts; pushed_ = popped_ = 0;
        for (size_t i = 0; i < n; ++i) if (!parts[i]) ready_[pushed_++] = (int) i;
        return true;
    }
    bool part_done(size_t i) {
        std::lock_guard<std::mutex> lock(mu_);
        if (i >= size_ || left_[i] <= 0) return false;
        if (--left_[i] == 0) { ready_[pushed_++] = (int) i; cv_.notify_one(); }
        return true;
    }
    int wait_next() {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return popped_ < pushed_ || popped_ == size_; });
        return popped_ < pushed_ ? ready_[popped_++] : -1;
    }
};
} // namespace strata::v41
