#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

#include <boost/noncopyable.hpp>

#include "app/flow/command.h"

namespace app {
namespace flow {

class CommandQueue : boost::noncopyable {
 public:
  CommandQueue();

  void Push(const Command &c);
  Command Pop(int timeout_ms);

 private:
  std::queue<Command>     queue_;
  std::mutex              mutex_;
  std::condition_variable cond_;
};

} // namespace flow
} // namespace app
