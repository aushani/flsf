#include "library/evaluation/flow_image_evaluator.h"

#include <iostream>
#include <fstream>

#include <Eigen/Core>

#include "library/kitti/util.h"

namespace library {
namespace evaluation {

FlowImageEvaluator::FlowImageEvaluator(const kt::Tracklets &tracklets, const kt::CameraCal &cc, const std::vector<kt::Pose> &poses) :
 tracklets_(tracklets),
 camera_cal_(cc),
 poses_(poses) {
}

void FlowImageEvaluator::Evaluate(const fl::FlowImage &flow_image, int from, int to) {
  const kt::Pose pose1 = poses_[from];
  const kt::Pose pose2 = poses_[to];

  const double res = flow_image.GetResolution();

  for (int i=flow_image.MinX(); i<=flow_image.MaxX(); i++) {
    for (int j=flow_image.MinY(); j<=flow_image.MaxY(); j++) {
      double x = i*res;
      double y = j*res;
      double z = 0.0;

      Eigen::Vector2f pos1(x, y);

      // Check in camera view
      if (!camera_cal_.InCameraView(x, y, z)) {
        continue;
      }

      // Only valid flows
      if (!flow_image.GetFlowValid(i, j)) {
        continue;
      }

      // Get object class
      kt::ObjectClass c = kt::GetObjectTypeAtLocation(&tracklets_, pos1, from, res);

      if (c != kt::ObjectClass::CAR) {
        continue;
      }

      // Find true flow
      Eigen::Vector2f pos2 = kt::FindCorrespondingPosition(&tracklets_, pos1, from, to, pose1, pose2);

      Eigen::Vector2f true_flow = pos2 - pos1;

      double xf = flow_image.GetXFlow(i, j) * res;
      double yf = flow_image.GetYFlow(i, j) * res;
      Eigen::Vector2f pred_flow(xf, yf);

      Eigen::Vector2f err = true_flow - pred_flow;

      //printf("%d %d\n", i, j);
      //printf("true is %f %f\n", true_flow.x(), true_flow.y());
      //printf("pred is %f %f\n", pred_flow.x(), pred_flow.y());
      //printf("error is %f %f\n", err.x(), err.y());
      //printf("\n");

      total_err_ += err.norm();
      count_++;

      errors_.push_back(err.norm());
    }
  }

  printf("Mean norm error: %5.3f (%d samples)\n", total_err_ / count_, count_);
}

void FlowImageEvaluator::Clear() {
  total_err_ = 0.0;
  count_ = 0;

  errors_.clear();
}

void FlowImageEvaluator::WriteErrors(const fs::path &path) {
  std::ofstream file(path.string());

  for (const float err : errors_) {
    file << err << std::endl;
  }
}

} // namespace evaluation
} // namespace library
