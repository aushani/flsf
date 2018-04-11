#include "app/flow/command_queue.h"

#include <chrono>

using namespace std::chrono_literals;

namespace app {
namespace flow {

CommandQueue::CommandQueue() {

}

bool CommandQueue::Push(char c) {
  switch(c) {
    case 'N':
    queue_.push(NEXT);
    return true;

    default:
    // No command
    break;
  }

  return false;
}

Command CommandQueue::Pop(int timeout_ms) {
  Command c = NONE;

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

void CommandQueue::Push(Command c) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(c);
  }

  cond_.notify_one();
}

} // namespace flow
} // namespace app
