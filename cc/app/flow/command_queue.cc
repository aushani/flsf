#include "app/flow/command_queue.h"

#include <chrono>

using namespace std::chrono_literals;

namespace app {
namespace flow {

CommandQueue::CommandQueue() {
}

Command CommandQueue::Pop(int timeout_ms) {
  Command c(Type::NONE);

  {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait_for(lock, timeout_ms * 1ms);

    if (queue_.size() > 0) {
      c = queue_.front();
      queue_.pop();
    }
  }

  return c;
}

void CommandQueue::Push(const Command &c) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(c);
  }

  cond_.notify_one();
}

} // namespace flow
} // namespace app
