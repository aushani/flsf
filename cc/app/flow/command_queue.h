#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

#include <boost/noncopyable.hpp>

namespace app {
namespace flow {

enum Command {
  NEXT,
  NONE,
};

class CommandQueue : boost::noncopyable {
 public:
  CommandQueue();

  bool Push(char c);
  Command Pop(int timeout_ms);

 private:
  std::queue<Command>     queue_;
  std::mutex              mutex_;
  std::condition_variable cond_;

  void Push(Command c);
};

} // namespace flow
} // namespace app
