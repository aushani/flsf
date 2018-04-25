#pragma once

namespace app {
namespace flow {

enum Type {
  NONE,
  NEXT,
  VIEW_MODE,
  CLICK_AT,
};

class Command {
 public:
  Command(const Type &type);

  void SetClickPosition(double x, double y);
  void SetViewMode(int view_mode);

  Type GetCommandType() const;

  double GetClickX() const;
  double GetClickY() const;

  int GetViewMode() const;

 private:
  Type type_;

  double click_pos_[2];

  int view_mode_;

};

} // namespace flow
} // namespace app
