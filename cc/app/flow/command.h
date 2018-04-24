#pragma once

namespace app {
namespace flow {

enum Type {
  NONE,
  NEXT,
  CLICK_AT,
};

class Command {
 public:
  Command(const Type &type);

  void SetClickPosition(double x, double y);

  Type GetCommandType() const;

  double GetClickX() const;
  double GetClickY() const;

 private:
  Type type_;

  double click_pos_[2];

};

} // namespace flow
} // namespace app
