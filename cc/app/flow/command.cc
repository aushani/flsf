#include "app/flow/command.h"

#include <boost/assert.hpp>

namespace app {
namespace flow {

Command::Command(const Type &type) :
 type_(type) {
}

void Command::SetClickPosition(double x, double y) {
  BOOST_ASSERT(type_ == CLICK_AT);

  click_pos_[0] = x;
  click_pos_[1] = y;
}

Type Command::GetCommandType() const {
  return type_;
}

double Command::GetClickX() const {
  BOOST_ASSERT(type_ == CLICK_AT);

  return click_pos_[0];
}

double Command::GetClickY() const {
  BOOST_ASSERT(type_ == CLICK_AT);

  return click_pos_[1];
}

} // namespace flow
} // namespace app
